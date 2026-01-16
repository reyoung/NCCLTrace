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
#include <string>
#include <string_view>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#if __has_include(<boost/iostreams/filter/bzip2.hpp>)
#include <boost/iostreams/filter/bzip2.hpp>
#define NCCLTRACE_HAS_BOOST_BZIP2 1
#else
#define NCCLTRACE_HAS_BOOST_BZIP2 0
#endif

#if __has_include(<boost/iostreams/filter/lzma.hpp>)
#include <boost/iostreams/filter/lzma.hpp>
#define NCCLTRACE_HAS_BOOST_LZMA 1
#else
#define NCCLTRACE_HAS_BOOST_LZMA 0
#endif

// Note: zstd filter availability depends on Boost version/build.
#if __has_include(<boost/iostreams/filter/zstd.hpp>)
#include <boost/iostreams/filter/zstd.hpp>
#define NCCLTRACE_HAS_BOOST_ZSTD 1
#else
#define NCCLTRACE_HAS_BOOST_ZSTD 0
#endif

namespace nccltrace {

// Concept for items that can be dumped
template <typename T>
concept DumpItem = requires(T item, std::ostream &out) {
  { item.dump(out) } -> std::same_as<void>;
};

template <DumpItem T> class FileDumper {
public:
  explicit FileDumper(
      const std::string &filename,
      std::chrono::milliseconds sleep_duration = std::chrono::milliseconds(2))
      : sleep_duration_(sleep_duration) {
    const auto format = resolve_format();
    filename_ = make_output_filename(filename, format);

    // Open file for writing (binary: msgpack bytes)
    file_.open(filename_, std::ios::out | std::ios::binary);
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename_);
    }

    // Setup output stream filter chain
    setup_output_filters(format);
    out_.push(file_);

    dumper_thread_ =
        std::jthread(std::bind_front(&FileDumper<T>::dumper_thread_func, this));
  }

  ~FileDumper() { stop(); }

  // Non-copyable
  FileDumper(const FileDumper &) = delete;
  FileDumper &operator=(const FileDumper &) = delete;

  // Non-movable (filtering_ostream is not movable; this object owns a thread)
  FileDumper(FileDumper &&) = delete;
  FileDumper &operator=(FileDumper &&) = delete;

  void push(T item) {
    if (stopped_.load()) {
      throw std::runtime_error("FileDumper is stopped");
    }
    queue_.push(std::move(item));
  }

  void stop() {
    bool expected = false;
    if (stopped_.compare_exchange_strong(expected, true)) {
      dumper_thread_.request_stop();
      if (dumper_thread_.joinable()) {
        dumper_thread_.join();
      }

      // Finalize filter chain (important for compressed formats)
      try {
        out_.flush();
        out_.reset();
      } catch (...) {
      }

      if (file_.is_open()) {
        file_.close();
      }
    }
  }

  const std::string &filename() const { return filename_; }

private:
  enum class DumpFormat {
    Plain,
    Gzip,
    Bz2,
    Lzma,
    Zstd,
  };

  static std::string_view to_lower(std::string_view s) {
    // small helper; avoid locale
    static thread_local std::string tmp;
    tmp.assign(s.begin(), s.end());
    for (char &c : tmp) {
      if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return std::string_view(tmp);
  }

  static DumpFormat parse_format(std::string_view v) {
    v = to_lower(v);
    if (v == "" || v == "gzip" || v == "gz") return DumpFormat::Gzip;
    if (v == "plain" || v == "none") return DumpFormat::Plain;
    if (v == "bz2" || v == "bzip2") return DumpFormat::Bz2;
    if (v == "lzma" || v == "xz") return DumpFormat::Lzma;
    if (v == "zstd" || v == "zst") return DumpFormat::Zstd;
    // unknown -> default
    return DumpFormat::Gzip;
  }

  static DumpFormat resolve_format() {
    // New var
    if (const char *v = std::getenv("NCCL_TRACER_DUMP_FORMAT")) {
      if (*v != '\0') {
        return parse_format(v);
      }
    }

    // Deprecated alias: NCCL_TRACER_DUMP_GZIP
    // If =0 => plain, else => gzip
    if (const char *v = std::getenv("NCCL_TRACER_DUMP_GZIP")) {
      if (*v != '\0' && std::string_view(v) == "0") {
        return DumpFormat::Plain;
      }
      return DumpFormat::Gzip;
    }

    // Default
    return DumpFormat::Gzip;
  }

  static std::string make_output_filename(const std::string &base,
                                          DumpFormat format) {
    switch (format) {
    case DumpFormat::Plain:
      return base + ".msgpack";
    case DumpFormat::Gzip:
      return base + ".msgpack.gz";
    case DumpFormat::Bz2:
      return base + ".msgpack.bz2";
    case DumpFormat::Lzma:
      return base + ".msgpack.xz";
    case DumpFormat::Zstd:
      return base + ".msgpack.zst";
    }
    return base + ".msgpack.gz";
  }

  void setup_output_filters(DumpFormat format) {
    switch (format) {
    case DumpFormat::Plain:
      return;
    case DumpFormat::Gzip:
      out_.push(boost::iostreams::gzip_compressor());
      return;
    case DumpFormat::Bz2:
#if NCCLTRACE_HAS_BOOST_BZIP2
      out_.push(boost::iostreams::bzip2_compressor());
#else
      std::cerr << "[nccltrace] NCCL_TRACER_DUMP_FORMAT=bz2 requested but "
                   "Boost bzip2 filter not available; falling back to gzip\n";
      out_.push(boost::iostreams::gzip_compressor());
#endif
      return;
    case DumpFormat::Lzma:
#if NCCLTRACE_HAS_BOOST_LZMA
      out_.push(boost::iostreams::lzma_compressor());
#else
      std::cerr << "[nccltrace] NCCL_TRACER_DUMP_FORMAT=lzma requested but "
                   "Boost lzma filter not available; falling back to gzip\n";
      out_.push(boost::iostreams::gzip_compressor());
#endif
      return;
    case DumpFormat::Zstd:
#if NCCLTRACE_HAS_BOOST_ZSTD
      out_.push(boost::iostreams::zstd_compressor());
#else
      std::cerr << "[nccltrace] NCCL_TRACER_DUMP_FORMAT=zstd requested but "
                   "Boost zstd filter not available; falling back to gzip\n";
      out_.push(boost::iostreams::gzip_compressor());
#endif
      return;
    }
  }

  void dumper_thread_func(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
      auto item = queue_.try_pop();
      if (item.has_value()) {
        item.value().dump(out_);
      } else {
        out_.flush();
        std::this_thread::yield();
      }
    }

    while (true) {
      auto item = queue_.try_pop();
      if (!item.has_value()) {
        break;
      }
      item.value().dump(out_);
    }

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
