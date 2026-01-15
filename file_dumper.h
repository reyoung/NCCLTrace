#pragma once

#include <concepts>
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <stop_token>
#include <functional>
#include <stdexcept>
#include <cstring>
#include <zstd.h>
#include "lock_free_queue.h"

namespace nccltrace {

// Concept for items that can be dumped
template<typename T>
concept DumpItem = requires(T item, std::ostream& out) {
    { item.dump(out) } -> std::same_as<void>;
};

template<DumpItem T>
class FileDumper {
public:
    // Constructor with filename and sleep duration (default 100ms)
    explicit FileDumper(
        const std::string& filename,
        std::chrono::milliseconds sleep_duration = std::chrono::milliseconds(100)
    )
        : filename_(filename + ".zst")  // Add .zst suffix
        , sleep_duration_(sleep_duration)
        , file_(nullptr)
        , cctx_(nullptr)
        , compression_buffer_size_(ZSTD_CStreamOutSize())
    {
        // Open file for writing
        file_ = std::fopen(filename_.c_str(), "wb");
        if (file_ == nullptr) {
            throw std::runtime_error("Failed to open file: " + filename_);
        }

        // Create zstd compression context
        cctx_ = ZSTD_createCCtx();
        if (cctx_ == nullptr) {
            std::fclose(file_);
            throw std::runtime_error("Failed to create ZSTD compression context");
        }

        // Set high compression level (19 is max for zstd, using 15 for good balance)
        size_t const result = ZSTD_CCtx_setParameter(cctx_, ZSTD_c_compressionLevel, 15);
        if (ZSTD_isError(result)) {
            ZSTD_freeCCtx(cctx_);
            std::fclose(file_);
            throw std::runtime_error("Failed to set compression level: " + std::string(ZSTD_getErrorName(result)));
        }

        // Allocate compression output buffer
        compression_buffer_.resize(compression_buffer_size_);

        // Start the dumper thread
        dumper_thread_ = std::jthread(
            std::bind_front(&FileDumper<T>::dumper_thread_func, this)
        );
    }

    // Destructor - stops the background thread
    ~FileDumper() {
        stop();
    }
    
    // Non-copyable
    FileDumper(const FileDumper&) = delete;
    FileDumper& operator=(const FileDumper&) = delete;
    
    // Movable
    FileDumper(FileDumper&& other) noexcept
        : filename_(std::move(other.filename_))
        , sleep_duration_(other.sleep_duration_)
        , queue_(std::move(other.queue_))
        , file_(other.file_)
        , cctx_(other.cctx_)
        , compression_buffer_(std::move(other.compression_buffer_))
        , compression_buffer_size_(other.compression_buffer_size_)
        , dumper_thread_(std::move(other.dumper_thread_))
        , stopped_(other.stopped_.load())
    {
        other.file_ = nullptr;
        other.cctx_ = nullptr;
        other.stopped_ = true;
    }
    
    FileDumper& operator=(FileDumper&& other) noexcept {
        if (this != &other) {
            stop();

            filename_ = std::move(other.filename_);
            sleep_duration_ = other.sleep_duration_;
            queue_ = std::move(other.queue_);
            file_ = other.file_;
            cctx_ = other.cctx_;
            compression_buffer_ = std::move(other.compression_buffer_);
            compression_buffer_size_ = other.compression_buffer_size_;
            dumper_thread_ = std::move(other.dumper_thread_);
            stopped_ = other.stopped_.load();

            other.file_ = nullptr;
            other.cctx_ = nullptr;
            other.stopped_ = true;
        }
        return *this;
    }
    
    // Push item into the queue
    void push(T item) {
        if (stopped_.load()) {
            throw std::runtime_error("FileDumper is stopped");
        }
        queue_.push(std::move(item));
    }
    
    // Stop the dumper thread
    void stop() {
        bool expected = false;
        if (stopped_.compare_exchange_strong(expected, true)) {
            dumper_thread_.request_stop();
            if (dumper_thread_.joinable()) {
                dumper_thread_.join();
            }
            
            // Free zstd context and close file
            if (cctx_ != nullptr) {
                ZSTD_freeCCtx(cctx_);
                cctx_ = nullptr;
            }
            if (file_ != nullptr) {
                std::fclose(file_);
                file_ = nullptr;
            }
        }
    }

private:
    // Helper function to write compressed data
    void write_compressed(const void* data, size_t size) {
        ZSTD_inBuffer input = { data, size, 0 };

        while (input.pos < input.size) {
            ZSTD_outBuffer output = { compression_buffer_.data(), compression_buffer_size_, 0 };

            size_t const result = ZSTD_compressStream2(cctx_, &output, &input, ZSTD_e_continue);
            if (ZSTD_isError(result)) {
                throw std::runtime_error("Compression error: " + std::string(ZSTD_getErrorName(result)));
            }

            if (output.pos > 0) {
                std::fwrite(compression_buffer_.data(), 1, output.pos, file_);
            }
        }
    }

    // Helper function to flush compressed data
    void flush_compressed() {
        ZSTD_outBuffer output = { compression_buffer_.data(), compression_buffer_size_, 0 };

        size_t remaining = ZSTD_flushStream(cctx_, &output);
        while (remaining > 0) {
            if (output.pos > 0) {
                std::fwrite(compression_buffer_.data(), 1, output.pos, file_);
            }
            output.pos = 0;
            remaining = ZSTD_flushStream(cctx_, &output);
        }

        if (output.pos > 0) {
            std::fwrite(compression_buffer_.data(), 1, output.pos, file_);
        }
        std::fflush(file_);
    }

    // Helper function to finish compression
    void finish_compressed() {
        ZSTD_outBuffer output = { compression_buffer_.data(), compression_buffer_size_, 0 };

        size_t remaining = ZSTD_endStream(cctx_, &output);
        while (remaining > 0) {
            if (output.pos > 0) {
                std::fwrite(compression_buffer_.data(), 1, output.pos, file_);
            }
            output.pos = 0;
            remaining = ZSTD_endStream(cctx_, &output);
        }

        if (output.pos > 0) {
            std::fwrite(compression_buffer_.data(), 1, output.pos, file_);
        }
    }

    // Background thread function
    void dumper_thread_func(std::stop_token stoken) {
        // Use a stringstream buffer for efficient serialization
        std::ostringstream buffer;

        while (!stoken.stop_requested()) {
            // Try to pop an item from the queue
            auto item = queue_.try_pop();
            
            if (item.has_value()) {
                // Dump the item to the string buffer
                buffer.str("");
                buffer.clear();
                item.value().dump(buffer);
                buffer << "\n";

                // Write compressed data
                if (file_ != nullptr && cctx_ != nullptr) {
                    std::string data = buffer.str();
                    write_compressed(data.c_str(), data.size());
                    // Flush periodically for better error recovery
                    flush_compressed();
                }
            } else {
                // No data available, sleep for the specified duration
                std::this_thread::sleep_for(sleep_duration_);
            }
        }
        
        // Process any remaining items in the queue before stopping
        while (true) {
            auto item = queue_.try_pop();
            if (!item.has_value()) {
                break;
            }
            
            buffer.str("");
            buffer.clear();
            item.value().dump(buffer);
            buffer << "\n";

            if (file_ != nullptr && cctx_ != nullptr) {
                std::string data = buffer.str();
                write_compressed(data.c_str(), data.size());
            }
        }

        // Finish compression
        if (file_ != nullptr && cctx_ != nullptr) {
            finish_compressed();
        }
    }
    
    std::string filename_;
    std::chrono::milliseconds sleep_duration_;
    LockFreeQueue<T> queue_;
    std::FILE* file_;
    ZSTD_CCtx* cctx_;
    std::vector<char> compression_buffer_;
    size_t compression_buffer_size_;
    std::jthread dumper_thread_;
    std::atomic<bool> stopped_{false};
};

} // namespace nccltrace
