#pragma once

#include "comm_desc.h"
#include "lock_free_queue.h"
#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <cstdlib>
#include <string_view>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

namespace nccltrace {

// Concept for items that can be dumped
template <typename T>
concept DumpItem = requires(T item, std::ostream &out) {
  { item.dump(out) } -> std::same_as<void>;
};

template <DumpItem T> class FileDumper {
public:
  // Constructor with filename and sleep duration (default 100ms)
  explicit FileDumper(
      const std::string &filename,
      std::chrono::milliseconds sleep_duration = std::chrono::milliseconds(2))
      : sleep_duration_(sleep_duration) {
    const bool gzip_enabled = should_enable_gzip();
    filename_ = make_output_filename(filename, gzip_enabled);

    // Open file for writing (binary: msgpack bytes)
    file_.open(filename_, std::ios::out | std::ios::binary);
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename_);
    }

    // Setup output stream (optionally gzip)
    if (gzip_enabled) {
      out_.push(boost::iostreams::gzip_compressor());
    }
    out_.push(file_);

    // Start the dumper thread
    dumper_thread_ =
        std::jthread(std::bind_front(&FileDumper<T>::dumper_thread_func, this));
  }

  // Destructor - stops the background thread
  ~FileDumper() { stop(); }

  // Non-copyable
  FileDumper(const FileDumper &) = delete;
  FileDumper &operator=(const FileDumper &) = delete;

  // Movable
  FileDumper(FileDumper &&other) noexcept
      : filename_(std::move(other.filename_)),
        sleep_duration_(other.sleep_duration_), queue_(std::move(other.queue_)),
        file_(std::move(other.file_)), out_(std::move(other.out_)),
        dumper_thread_(std::move(other.dumper_thread_)),
        stopped_(other.stopped_.load()) {
    other.stopped_ = true;
  }

  FileDumper &operator=(FileDumper &&other) noexcept {
    if (this != &other) {
      stop();

      filename_ = std::move(other.filename_);
      sleep_duration_ = other.sleep_duration_;
      queue_ = std::move(other.queue_);
      file_ = std::move(other.file_);
      out_ = std::move(other.out_);
      dumper_thread_ = std::move(other.dumper_thread_);
      stopped_ = other.stopped_.load();

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

      // Finalize gzip stream (needs reset before closing underlying file)
      try {
        out_.flush();
        out_.reset();
      } catch (...) {
        // best-effort shutdown
      }

      // Close file
      if (file_.is_open()) {
        file_.close();
      }
    }
  }

  const std::string &filename() const { return filename_; }

private:
  static bool should_enable_gzip() {
    const char *v = std::getenv("NCCL_TRACER_DUMP_GZIP");
    // default: enabled
    if (!v || *v == '\0') {
      return true;
    }
    // "1" enables; "0" disables; anything else treated as enabled
    return !(std::string_view(v) == "0");
  }

  static std::string make_output_filename(const std::string &base,
                                          bool gzip_enabled) {
    return gzip_enabled ? (base + ".msgpack.gz") : (base + ".msgpack");
  }

  // Background thread function
  void dumper_thread_func(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
      // Try to pop an item from the queue
      auto item = queue_.try_pop();
      if (item.has_value()) {
        // Dump to output stream (file or gzip-wrapped)
        item.value().dump(out_);
      } else {
        // No data available
        out_.flush();
        std::this_thread::yield();
      }
    }

    // Process any remaining items in the queue before stopping
    while (true) {
      auto item = queue_.try_pop();
      if (!item.has_value()) {
        break;
      }
      item.value().dump(out_);
    }

    // Final flush
    out_.flush();
  }

  std::string filename_;
  std::chrono::milliseconds sleep_duration_;
  LockFreeQueue<T> queue_;
  std::ofstream file_;
  boost::iostreams::filtering_ostream out_;
  std::jthread dumper_thread_;
  std::atomic<bool> stopped_{false};
};

} // namespace nccltrace
