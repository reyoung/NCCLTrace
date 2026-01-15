#pragma once
#include "comm_desc.h"
#include "unix_nano_time.h"
#include <memory>
#include <variant>

namespace nccltrace {

struct BaseItem {
  std::shared_ptr<CommDesc> comm_;
  Timepoint timestamp_;
  std::optional<nlohmann::json> extra_;


  BaseItem& with_extra(std::string_view key, nlohmann::json val) {
    if (!extra_.has_value()) {
      extra_ = nlohmann::json::object();
    }
    (*extra_)[key] = std::move(val);
    return *this;
  }
};

struct CommInit : BaseItem {};
struct CommDestroy : BaseItem {};
struct ProfileGroupStart : BaseItem {
  int rank_;
  void *self_;
};
struct ProfileGroupStop : BaseItem {
  int rank_;
};

struct ProfileCollContext {
  int rank_;
  uint64_t seq_;
  std::string_view func_;
  void const *send_buff_;
  void *recv_buffer_;
  size_t count_;
  int root_;
  std::string_view data_type_;
  uint8_t n_channels_;
  uint8_t n_warps_;
  std::string_view algo_;
  std::string_view proto_;
};

struct ProfileCollStart : BaseItem {
  ProfileCollContext context_;
};

struct ProfileCollStop : BaseItem {
  ProfileCollContext context_;
};

struct ProfileP2pContext {
  int rank_;
  std::string_view func_;
  void *buff_;
  std::string_view data_type_;
  size_t count_;
  int peer_;
  uint8_t n_channels_;
};

struct ProfileP2pStart : BaseItem {
  ProfileP2pContext context_;
};

struct ProfileP2pStop : BaseItem {
  ProfileP2pContext context_;
};

struct ProfileKernelChContext {
  int rank_;
  uint8_t channel_id_;
  uint64_t p_timer_; // start timestamp from GPU globaltimer
};

struct ProfileKernelChStart : BaseItem {
  ProfileKernelChContext context_;
};

struct ProfileKernelChStop : BaseItem {
  ProfileKernelChContext context_;
  uint64_t stop_p_timer_; // stop timestamp from GPU globaltimer
};

struct ProfileProxyOpContext {
  int rank_;
  pid_t pid_;
  uint8_t channel_id_;
  int peer_;
  int n_steps_;
  int chunk_size_;
  int is_send_;
};

struct ProfileProxyOpStart : BaseItem {
  ProfileProxyOpContext context_;
};

struct ProfileProxyOpStop : BaseItem {
  ProfileProxyOpContext context_;
};

struct ProfileProxyOpStateRecord : BaseItem {
  ProfileProxyOpContext context_;
  std::string_view state_name_;  // State name as string
};

struct ProfileProxyStepContext {
  int rank_;
  int step_;
};

struct ProfileProxyStepStart : BaseItem {
  ProfileProxyStepContext context_;
};

struct ProfileProxyStepStop : BaseItem {
  ProfileProxyStepContext context_;
};

struct ProfileProxyStepStateRecord : BaseItem {
  ProfileProxyStepContext context_;
  std::string_view state_name_;  // State name as string
  size_t trans_size_;  // transferred size from state args
};

struct ProfileProxyCtrlContext {
  int rank_;
};

struct ProfileProxyCtrlStart : BaseItem {
  ProfileProxyCtrlContext context_;
};

struct ProfileProxyCtrlStop : BaseItem {
  ProfileProxyCtrlContext context_;
};

struct ProfileProxyCtrlStateRecord : BaseItem {
  ProfileProxyCtrlContext context_;
  std::string_view state_name_;  // State name as string
  int appended_proxy_ops_;  // Only valid for Append/AppendEnd states
};

} // namespace nccltrace

NLOHMANN_JSON_NAMESPACE_BEGIN

// Generic builder for all item types
template <typename T> class JsonBuilder {
  const T &item_;
  json j_;

public:
  explicit JsonBuilder(const T &item, const char *type) : item_(item) {
    if (item_.extra_.has_value()) {
      j_ = *(item_.extra_);
    }
    j_["comm"] = item_.comm_;
    j_["type"] = type;
    j_["timestamp"] = item_.timestamp_;

  }

  JsonBuilder &with_coll_context() {
    auto &ctx = item_.context_;
    j_["context"] = json{
        {"seq", ctx.seq_},
        {"func", ctx.func_},
        {"send_buff", uintptr_t(ctx.send_buff_)},
        {"recv_buffer", uintptr_t(ctx.recv_buffer_)},
        {"count", ctx.count_},
        {"root", ctx.root_},
        {"data_type", ctx.data_type_},
        {"n_channels", ctx.n_channels_},
        {"n_warps", ctx.n_warps_},
        {"algo", ctx.algo_},
        {"proto", ctx.proto_},
    };
    return *this;
  }

  JsonBuilder &with_p2p_context() {
    auto &ctx = item_.context_;
    j_["context"] = json{
        {"rank", ctx.rank_},
        {"func", ctx.func_},
        {"buff", uintptr_t(ctx.buff_)},
        {"data_type", ctx.data_type_},
        {"count", ctx.count_},
        {"peer", ctx.peer_},
        {"n_channels", ctx.n_channels_},
    };
    return *this;
  }

  JsonBuilder &with_kernel_ch_context() {
    auto &ctx = item_.context_;
    j_["context"] = json{
        {"rank", ctx.rank_},
        {"channel_id", ctx.channel_id_},
        {"p_timer", ctx.p_timer_},
    };
    return *this;
  }

  JsonBuilder &with_proxy_op_context() {
    auto &ctx = item_.context_;
    j_["context"] = json{
        {"rank", ctx.rank_},
        {"pid", ctx.pid_},
        {"channel_id", ctx.channel_id_},
        {"peer", ctx.peer_},
        {"n_steps", ctx.n_steps_},
        {"chunk_size", ctx.chunk_size_},
        {"is_send", ctx.is_send_},
    };
    return *this;
  }

  JsonBuilder &with_proxy_step_context() {
    auto &ctx = item_.context_;
    j_["context"] = json{
        {"rank", ctx.rank_},
        {"step", ctx.step_},
    };
    return *this;
  }

  JsonBuilder &with_proxy_ctrl_context() {
    auto &ctx = item_.context_;
    j_["context"] = json{
        {"rank", ctx.rank_},
    };
    return *this;
  }

  JsonBuilder &with_proxy_ctrl_state() {
    j_["state"] = item_.state_name_;
    j_["appended_proxy_ops"] = item_.appended_proxy_ops_;
    return *this;
  }

  JsonBuilder &with_state_and_trans_size() {
    j_["state"] = item_.state_name_;
    j_["trans_size"] = item_.trans_size_;
    return *this;
  }

  JsonBuilder &with_state_name() {
    j_["state"] = item_.state_name_;
    return *this;
  }

  JsonBuilder &with_stop_p_timer() {
    j_["stop_p_timer"] = item_.stop_p_timer_;
    return *this;
  }

  json build() { return std::move(j_); }
};

template <> struct adl_serializer<nccltrace::CommInit> {
  static void to_json(json &j, const nccltrace::CommInit &init) {
    j = JsonBuilder(init, "comm_init").build();
  }
};

template <> struct adl_serializer<nccltrace::CommDestroy> {
  static void to_json(json &j, const nccltrace::CommDestroy &destroy) {
    j = JsonBuilder(destroy, "comm_destroy").build();
  }
};

template <> struct adl_serializer<std::monostate> {
  static void to_json(json &j, const std::monostate &ms) {
    j = json{{"type", "none"}};
  }
};

template <> struct adl_serializer<nccltrace::ProfileGroupStart> {
  static void to_json(json &j, const nccltrace::ProfileGroupStart &start) {
    j = JsonBuilder(start, "profile_group_start").build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileGroupStop> {
  static void to_json(json &j, const nccltrace::ProfileGroupStop &stop) {
    j = JsonBuilder(stop, "profile_group_stop").build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileCollStart> {
  static void to_json(json &j, const nccltrace::ProfileCollStart &start) {
    j = JsonBuilder(start, "profile_coll_start").with_coll_context().build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileCollStop> {
  static void to_json(json &j, const nccltrace::ProfileCollStop &stop) {
    j = JsonBuilder(stop, "profile_coll_stop").with_coll_context().build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileP2pStart> {
  static void to_json(json &j, const nccltrace::ProfileP2pStart &start) {
    j = JsonBuilder(start, "profile_p2p_start").with_p2p_context().build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileP2pStop> {
  static void to_json(json &j, const nccltrace::ProfileP2pStop &stop) {
    j = JsonBuilder(stop, "profile_p2p_stop").with_p2p_context().build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileKernelChStart> {
  static void to_json(json &j, const nccltrace::ProfileKernelChStart &start) {
    j = JsonBuilder(start, "profile_kernel_ch_start")
            .with_kernel_ch_context()
            .build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileKernelChStop> {
  static void to_json(json &j, const nccltrace::ProfileKernelChStop &stop) {
    j = JsonBuilder(stop, "profile_kernel_ch_stop")
            .with_kernel_ch_context()
            .with_stop_p_timer()
            .build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileProxyOpStart> {
  static void to_json(json &j, const nccltrace::ProfileProxyOpStart &start) {
    j = JsonBuilder(start, "profile_proxy_op_start")
            .with_proxy_op_context()
            .build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileProxyOpStop> {
  static void to_json(json &j, const nccltrace::ProfileProxyOpStop &stop) {
    j = JsonBuilder(stop, "profile_proxy_op_stop")
            .with_proxy_op_context()
            .build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileProxyOpStateRecord> {
  static void to_json(json &j, const nccltrace::ProfileProxyOpStateRecord &record) {
    j = JsonBuilder(record, "profile_proxy_op_state_record")
            .with_proxy_op_context()
            .with_state_name()
            .build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileProxyStepStart> {
  static void to_json(json &j, const nccltrace::ProfileProxyStepStart &start) {
    j = JsonBuilder(start, "profile_proxy_step_start")
            .with_proxy_step_context()
            .build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileProxyStepStop> {
  static void to_json(json &j, const nccltrace::ProfileProxyStepStop &stop) {
    j = JsonBuilder(stop, "profile_proxy_step_stop")
            .with_proxy_step_context()
            .build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileProxyStepStateRecord> {
  static void to_json(json &j, const nccltrace::ProfileProxyStepStateRecord &record) {
    j = JsonBuilder(record, "profile_proxy_step_state_record")
            .with_proxy_step_context()
            .with_state_and_trans_size()
            .build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileProxyCtrlStart> {
  static void to_json(json &j, const nccltrace::ProfileProxyCtrlStart &start) {
    j = JsonBuilder(start, "profile_proxy_ctrl_start")
            .with_proxy_ctrl_context()
            .build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileProxyCtrlStop> {
  static void to_json(json &j, const nccltrace::ProfileProxyCtrlStop &stop) {
    j = JsonBuilder(stop, "profile_proxy_ctrl_stop")
            .with_proxy_ctrl_context()
            .build();
  }
};

template <> struct adl_serializer<nccltrace::ProfileProxyCtrlStateRecord> {
  static void to_json(json &j, const nccltrace::ProfileProxyCtrlStateRecord &record) {
    j = JsonBuilder(record, "profile_proxy_ctrl_state_record")
            .with_proxy_ctrl_context()
            .with_proxy_ctrl_state()
            .build();
  }
};

NLOHMANN_JSON_NAMESPACE_END

namespace nccltrace {
using TraceDumpItemVar =
    std::variant<CommInit, CommDestroy, ProfileGroupStart, ProfileGroupStop,
                 ProfileCollStart, ProfileCollStop, ProfileP2pStart,
                 ProfileP2pStop, ProfileKernelChStart, ProfileKernelChStop,
                 ProfileProxyOpStart, ProfileProxyOpStop,
                 ProfileProxyOpStateRecord, ProfileProxyStepStart,
                 ProfileProxyStepStop, ProfileProxyStepStateRecord,
                 ProfileProxyCtrlStart, ProfileProxyCtrlStop,
                 ProfileProxyCtrlStateRecord, std::monostate>;
class TraceDumpItem {
public:
  TraceDumpItemVar item_;
  TraceDumpItem(TraceDumpItemVar var) : item_(std::move(var)) {}
  TraceDumpItem() = default;

  void dump(std::ostream &out) const {
    std::visit(
        [&out](const auto &arg) {
          nlohmann::json j = arg;
          out << j;
        },
        item_);
  }
};
} // namespace nccltrace