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
#include <zlib.h>
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
        : filename_(filename + ".gz")  // Add .gz suffix
        , sleep_duration_(sleep_duration)
        , gz_file_(nullptr)
    {
        // Open gzip file for writing
        gz_file_ = gzopen(filename_.c_str(), "wb");
        if (gz_file_ == nullptr) {
            throw std::runtime_error("Failed to open gzip file: " + filename_);
        }

        // Set compression level (6 is default, 9 is max compression)
        gzsetparams(gz_file_, 6, Z_DEFAULT_STRATEGY);

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
        , gz_file_(other.gz_file_)
        , dumper_thread_(std::move(other.dumper_thread_))
        , stopped_(other.stopped_.load())
    {
        other.gz_file_ = nullptr;
        other.stopped_ = true;
    }
    
    FileDumper& operator=(FileDumper&& other) noexcept {
        if (this != &other) {
            stop();

            filename_ = std::move(other.filename_);
            sleep_duration_ = other.sleep_duration_;
            queue_ = std::move(other.queue_);
            gz_file_ = other.gz_file_;
            dumper_thread_ = std::move(other.dumper_thread_);
            stopped_ = other.stopped_.load();

            other.gz_file_ = nullptr;
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
            
            // Close the gzip file
            if (gz_file_ != nullptr) {
                gzclose(gz_file_);
                gz_file_ = nullptr;
            }
        }
    }

private:
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

                // Write compressed data to gzip file
                if (gz_file_ != nullptr) {
                    std::string data = buffer.str();
                    gzwrite(gz_file_, data.c_str(), data.size());
                    // Flush periodically for better error recovery
                    gzflush(gz_file_, Z_SYNC_FLUSH);
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

            if (gz_file_ != nullptr) {
                std::string data = buffer.str();
                gzwrite(gz_file_, data.c_str(), data.size());
            }
        }

        // Final flush
        if (gz_file_ != nullptr) {
            gzflush(gz_file_, Z_FINISH);
        }
    }
    
    std::string filename_;
    std::chrono::milliseconds sleep_duration_;
    LockFreeQueue<T> queue_;
    gzFile gz_file_;
    std::jthread dumper_thread_;
    std::atomic<bool> stopped_{false};
};

} // namespace nccltrace
