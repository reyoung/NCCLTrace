#include "comm_desc.h"
#include <memory>
#include <shared_mutex>
#include <unordered_map>

namespace nccltrace {

class CommDescDict {
public:
  std::shared_ptr<CommDesc> operator[](uint64_t comm_hash) {
    std::shared_lock read_lock(mutex_);
    if (auto it = descs_.find(comm_hash); it != descs_.end()) {
      return it->second;
    }
    return nullptr;
  }

  std::shared_ptr<CommDesc> create(const char *name, uint64_t hash, int n_nodes,
                                   int n_ranks, int rank,
                                   ncclDebugLogger_t log_fn) {
    std::unique_lock write_lock(mutex_);
    if (auto it = descs_.find(hash); it != descs_.end()) {
      return it->second;
    }
    auto desc =
        std::make_shared<CommDesc>(name, hash, n_nodes, n_ranks, rank, log_fn);
    descs_[hash] = desc;
    return desc;
  }

private:
  std::unordered_map<uint64_t, std::shared_ptr<CommDesc>> descs_;
  std::shared_mutex mutex_;
};

} // namespace nccltrace