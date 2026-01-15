#pragma once
#include "nlohmann/json.hpp"
#include <chrono>
namespace nccltrace {

struct Timepoint {
  uint64_t unix_nano_;

  Timepoint()
      : unix_nano_(std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count()) {}
};
} // namespace nccltrace

NLOHMANN_JSON_NAMESPACE_BEGIN
template <> struct adl_serializer<nccltrace::Timepoint> {
  static void to_json(json &j, const nccltrace::Timepoint &tp) {
    j = json{{"sec", tp.unix_nano_ / 1000000000},
             {"nsec", tp.unix_nano_ % 1000000000}};
  }
};
NLOHMANN_JSON_NAMESPACE_END