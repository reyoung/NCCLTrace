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
      : filename_(filename), sleep_duration_(sleep_duration) {
    // Open file for writing
    file_.open(filename_, std::ios::out);
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename_);
    }

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
        file_(std::move(other.file_)),
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

      // Close file
      if (file_.is_open()) {
        file_.close();
      }
    }
  }

private:
  // Background thread function
  void dumper_thread_func(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
      // Try to pop an item from the queue
      auto item = queue_.try_pop();
      if (item.has_value()) {
        // Dump directly to file
        item.value().dump(file_);
      } else {
        // No data available, sleep for the specified duration
        file_.flush();
        std::this_thread::yield();
      }
    }

    // Process any remaining items in the queue before stopping
    while (true) {
      auto item = queue_.try_pop();
      if (!item.has_value()) {
        break;
      }
      item.value().dump(file_);
    }

    // Final flush
    file_.flush();
  }

  std::string filename_;
  std::chrono::milliseconds sleep_duration_;
  LockFreeQueue<T> queue_;
  std::ofstream file_;
  std::jthread dumper_thread_;
  std::atomic<bool> stopped_{false};
};

} // namespace nccltrace
