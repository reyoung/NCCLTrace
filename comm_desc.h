#pragma once
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdint.h>
#include <string>
namespace nccltrace {
typedef enum {
  NCCL_LOG_NONE = 0,
  NCCL_LOG_VERSION = 1,
  NCCL_LOG_WARN = 2,
  NCCL_LOG_INFO = 3,
  NCCL_LOG_ABORT = 4,
  NCCL_LOG_TRACE = 5
} ncclDebugLogLevel;

typedef enum {
  NCCL_INIT = 0x1,
  NCCL_COLL = 0x2,
  NCCL_P2P = 0x4,
  NCCL_SHM = 0x8,
  NCCL_NET = 0x10,
  NCCL_GRAPH = 0x20,
  NCCL_TUNING = 0x40,
  NCCL_ENV = 0x80,
  NCCL_ALLOC = 0x100,
  NCCL_CALL = 0x200,
  NCCL_PROXY = 0x400,
  NCCL_NVLS = 0x800,
  NCCL_BOOTSTRAP = 0x1000,
  NCCL_REG = 0x2000,
  NCCL_PROFILE = 0x4000,
  NCCL_RAS = 0x8000,
  NCCL_ALL = ~0
} ncclDebugLogSubSys;
typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags,
                                  const char *file, int line, const char *fmt,
                                  ...);
struct CommDesc {
  std::string name_;
  uint64_t hash_;
  int n_nodes_;
  int n_ranks_;
  int rank_;
  ncclDebugLogger_t log_fn_;

  CommDesc(const char *name, uint64_t hash, int n_nodes, int n_ranks, int rank,
           ncclDebugLogger_t log_fn)
      : name_(name ? std::string(name) : ""), hash_(hash), n_nodes_(n_nodes),
        n_ranks_(n_ranks), rank_(rank), log_fn_(log_fn) {}

  template <class... Args>
  void log(ncclDebugLogLevel level, unsigned long flags, const char *file,
           int line, const char *fmt, Args... args) {
    if (log_fn_) {
      // always add NCCL_PRO
      log_fn_(level, flags | NCCL_PROFILE, file, line, fmt, args...);
    }
  }
};

} // namespace nccltrace

NLOHMANN_JSON_NAMESPACE_BEGIN

template <> struct adl_serializer<nccltrace::CommDesc> {
  static void to_json(nlohmann::json &j, const nccltrace::CommDesc &desc) {
    j = json{
        {"name", desc.name_},       {"hash", desc.hash_},
        {"n_nodes", desc.n_nodes_}, {"n_ranks", desc.n_ranks_},
        {"rank", desc.rank_},
    };
  }
};

template <typename T> struct adl_serializer<std::shared_ptr<T>> {
  static void to_json(json &j, const std::shared_ptr<T> &ptr) {
    if (ptr) {
      j = *ptr;
    } else {
      j = nullptr;
    }
  }
};

NLOHMANN_JSON_NAMESPACE_END

#define LOG_INFO(desc, flag, fmt, ...)                                         \
  desc->log(nccltrace::NCCL_LOG_INFO, flag, __FILE__, __LINE__, fmt,           \
            ##__VA_ARGS__)

#define LOG_TRACE(desc, flag, fmt, ...)                                        \
  desc->log(nccltrace::NCCL_LOG_TRACE, flag, __FILE__, __LINE__, fmt,          \
            ##__VA_ARGS__)

#define LOG_WARN(desc, flag, fmt, ...)                                         \
  desc->log(nccltrace::NCCL_LOG_WARN, flag, __FILE__, __LINE__, fmt,           \
            ##__VA_ARGS__)
