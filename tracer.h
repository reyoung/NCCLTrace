#pragma once
#include "comm_desc_dict.h"
#include "dump_items.h"
#include "file_dumper.h"
#include "free_reference_pool.h"

#include <iostream>
#include <memory>
#include <stdlib.h>
namespace nccltrace {

enum {
  ncclProfileGroup = (1 << 0),         // group event type
  ncclProfileColl = (1 << 1),          // host collective call event type
  ncclProfileP2p = (1 << 2),           // host point-to-point call event type
  ncclProfileProxyOp = (1 << 3),       // proxy operation event type
  ncclProfileProxyStep = (1 << 4),     // proxy step event type
  ncclProfileProxyCtrl = (1 << 5),     // proxy control event type
  ncclProfileKernelCh = (1 << 6),      // kernel channel event type
  ncclProfileNetPlugin = (1 << 7),     // network plugin-defined, events
  ncclProfileGroupApi = (1 << 8),      // Group API events
  ncclProfileCollApi = (1 << 9),       // Collective API events
  ncclProfileP2pApi = (1 << 10),       // Point-to-Point API events
  ncclProfileKernelLaunch = (1 << 11), // Kernel launch events
  // CE events (v6)
  ncclProfileCeColl = (1 << 12),  // CE collective operation
  ncclProfileCeSync = (1 << 13),  // CE synchronization operation
  ncclProfileCeBatch = (1 << 14), // CE batch operation
};

using Dumper = FileDumper<TraceDumpItem>;

struct CommContext;
struct ProfileGroupContext {
  int rank_;
  void *parent_;
};

struct BaseEventHandle {
  CommContext *context_;
  std::variant<std::monostate, ProfileGroupContext, ProfileCollContext,
               ProfileP2pContext, ProfileKernelChContext, ProfileProxyOpContext,
               ProfileProxyStepContext, ProfileProxyCtrlContext>
      event_context_;

  void *parent_{nullptr};

  void set_parent(void *parent);

  void reset(CommContext *ctx) {
    context_ = ctx;
    event_context_.emplace<std::monostate>();
    parent_ = nullptr;
  }

  void on_event_stop();

  void on_profile_group_start(void *parent, int rank);
  void on_coll_start(ProfileCollContext ctx);
  void on_p2p_start(ProfileP2pContext ctx);
  void on_kernel_ch_start(ProfileKernelChContext ctx);
  void on_proxy_op_start(ProfileProxyOpContext ctx);
  void on_proxy_step_start(ProfileProxyStepContext ctx);
  void on_proxy_ctrl_start(ProfileProxyCtrlContext ctx);
  void on_obj_free();
  void on_obj_required_again();

  template <typename T>
  void report(T desc);
};

std::optional<nlohmann::json> build_parent(const FreeRefPool<BaseEventHandle>::Item& ev_handle) {
  if (ev_handle.parent_ == nullptr) {
    return std::nullopt;
  }
  nlohmann::json j;
  auto p = reinterpret_cast<FreeRefPool<BaseEventHandle>::Item*>(ev_handle.parent_);
  j["id"] = p->rand_id_;
  j["parent"] = build_parent(*p);
  return j;
}


struct CommContext {
  std::weak_ptr<CommDesc> self_;
  std::weak_ptr<Dumper> dumper_;
  FreeRefPool<BaseEventHandle> pool_;

  using EventHandle = FreeRefPool<BaseEventHandle>::Item;

  CommContext(std::weak_ptr<CommDesc> desc, std::weak_ptr<Dumper> dumper)
      : self_(desc), dumper_(dumper),
        pool_([this](BaseEventHandle *handle) { handle->reset(this); }) {}

  template <typename T> void report(T desc, EventHandle* event_handle) {
    if (event_handle != nullptr) {
      desc.with_extra("id", event_handle->rand_id_);
      desc.with_extra("parent", build_parent(*event_handle));
    }

    auto comm = self_.lock();
    if (!comm) {
      return;
    }
    desc.comm_ = comm;
    auto dumper = dumper_.lock();
    if (!dumper) {
      return;
    }
    dumper->push(TraceDumpItem(std::move(desc)));
  }
};


using EventHandle = CommContext::EventHandle;
template <typename T>
void BaseEventHandle::report(T desc) {
  context_->report(std::move(desc), reinterpret_cast<EventHandle*>(this));
}

inline void BaseEventHandle::on_event_stop() {
  std::visit(
      [this](auto &&ev_ctx) {
        using T = std::decay_t<decltype(ev_ctx)>;
        if constexpr (std::is_same_v<T, ProfileGroupContext>) {
          report(ProfileGroupStop{});
        } else if constexpr (std::is_same_v<T, ProfileCollContext>) {
          report(ProfileCollStop{ .context_ = ev_ctx});
        } else if constexpr (std::is_same_v<T, ProfileP2pContext>) {
          report(ProfileP2pStop{.context_ = ev_ctx});
        } else if constexpr (std::is_same_v<T, ProfileProxyOpContext>) {
          report(ProfileProxyOpStop{.context_ = ev_ctx});
        } else if constexpr (std::is_same_v<T, ProfileProxyStepContext>) {
          report(ProfileProxyStepStop{.context_ = ev_ctx});
        } else if constexpr (std::is_same_v<T, ProfileProxyCtrlContext>) {
          report(ProfileProxyCtrlStop{.context_ = ev_ctx});
        }
        // Note: ProfileKernelChContext is handled in on_event_state_record_v4
        // when ncclProfilerKernelChStop state is received
      },
      event_context_);
}

inline void BaseEventHandle::on_profile_group_start(void *parent, int rank) {
  report(ProfileGroupStart{});
  event_context_ = ProfileGroupContext{.rank_ = rank, .parent_ = parent};
}

inline void BaseEventHandle::on_coll_start(ProfileCollContext ctx) {
  event_context_ = ctx;
  report(ProfileCollStart{ .context_ = ctx});
}

inline void BaseEventHandle::on_p2p_start(ProfileP2pContext ctx) {
  event_context_ = ctx;
  report(ProfileP2pStart{ .context_ = ctx});
}

inline void BaseEventHandle::on_kernel_ch_start(ProfileKernelChContext ctx) {
  event_context_ = ctx;
  report(ProfileKernelChStart{.context_ = ctx});
}

inline void BaseEventHandle::on_proxy_op_start(ProfileProxyOpContext ctx) {
  event_context_ = ctx;
  report(ProfileProxyOpStart{.context_ = ctx});
}

inline void BaseEventHandle::on_proxy_step_start(ProfileProxyStepContext ctx) {
  event_context_ = ctx;
  report(ProfileProxyStepStart{.context_ = ctx});
}

inline void BaseEventHandle::on_proxy_ctrl_start(ProfileProxyCtrlContext ctx) {
  event_context_ = ctx;
  report(ProfileProxyCtrlStart{.context_ = ctx});
}

inline void BaseEventHandle::set_parent(void *parent) {
  parent_ = parent;
  if (parent != nullptr) { // ref the parent object
    auto p = static_cast<EventHandle *>(parent);
    auto comm = this->context_->self_.lock();
    static_cast<EventHandle *>(parent)->ref();
  }
}
inline void BaseEventHandle::on_obj_free() {
  auto comm = this->context_->self_.lock();
  if (parent_ == nullptr) {
    return;
  }
  static_cast<EventHandle *>(parent_)->deref();
}

inline void BaseEventHandle::on_obj_required_again() {
  auto p = this->parent_;
  if (p == nullptr) {
    return;
  }
  // broadcast reference chain
  static_cast<EventHandle *>(p)->ref();
}

#define CTX_LOG_INFO(ctx, flag, fmt, ...)                                      \
  do {                                                                         \
    if (auto comm = (ctx).self_.lock(); comm) {                                \
      LOG_INFO(comm, flag, fmt, ##__VA_ARGS__);                                \
    }                                                                          \
  } while (0)

#define CTX_LOG_TRACE(ctx, flag, fmt, ...)                                     \
  do {                                                                         \
    if (auto comm = (ctx).self_.lock(); comm) {                                \
      LOG_TRACE(comm, flag, fmt, ##__VA_ARGS__);                               \
    }                                                                          \
  } while (0)

#define CTX_LOG_WARN(ctx, flag, fmt, ...)                                      \
  do {                                                                         \
    if (auto comm = (ctx).self_.lock(); comm) {                                \
      LOG_WARN(comm, flag, fmt, ##__VA_ARGS__);                                \
    }                                                                          \
  } while (0)

typedef struct {
  uint8_t type;    // event type descriptor: ncclProfileColl, ...
  void *parentObj; // pointer to the profiler parent object (for coll is the
                   // group)
  int rank;        // originating rank
  union {
    struct {
      uint64_t seqNumber;
      const char *func;
      void const *sendBuff;
      void *recvBuff;
      size_t count;
      int root;
      const char *datatype;
      uint8_t nChannels;
      uint8_t nWarps;
      const char *algo;
      const char *proto;
    } coll;

    struct {
      const char *func;
      void *buff;
      const char *datatype;
      size_t count;
      int peer;
      uint8_t nChannels;
    } p2p;

    struct {
      pid_t pid;         // pid of the originating process
      uint8_t channelId; // channel id for this proxy operation
      int peer;          // remote rank for send/recv
      int nSteps;        // number of steps for this proxy operation
      int chunkSize;     // amount of data transferred by this proxy operation
      int isSend;
    } proxyOp;

    struct {
      int step;
    } proxyStep;

    struct {
      uint8_t channelId;
      uint64_t pTimer; // start timestamp from GPU globaltimer
    } kernelCh;

    struct {
      int64_t id;
      void *data;
    } netPlugin;
  };
} ncclProfilerEventDescr_v4_t;

typedef enum {
  ncclProfilerProxyOpSendPosted = 0,      // deprecated in v4
  ncclProfilerProxyOpSendRemFifoWait = 1, // deprecated in v4
  ncclProfilerProxyOpSendTransmitted = 2, // deprecated in v4
  ncclProfilerProxyOpSendDone = 3,        // deprecated in v4
  ncclProfilerProxyOpRecvPosted = 4,      // deprecated in v4
  ncclProfilerProxyOpRecvReceived = 5,    // deprecated in v4
  ncclProfilerProxyOpRecvTransmitted = 6, // deprecated in v4
  ncclProfilerProxyOpRecvDone = 7,        // deprecated in v4
  ncclProfilerProxyOpInProgress_v4 = 19,

  /* Legacy proxy profiler states */
  ncclProfilerProxyStepSendGPUWait = 8,
  ncclProfilerProxyStepSendPeerWait_v4 = 20,
  ncclProfilerProxyStepSendWait = 9,
  ncclProfilerProxyStepRecvWait = 10,
  ncclProfilerProxyStepRecvFlushWait = 11,
  ncclProfilerProxyStepRecvGPUWait = 12,

  /* Legacy proxy control states */
  ncclProfilerProxyCtrlIdle = 13,
  ncclProfilerProxyCtrlActive = 14,
  ncclProfilerProxyCtrlSleep = 15,
  ncclProfilerProxyCtrlWakeup = 16,
  ncclProfilerProxyCtrlAppend = 17,
  ncclProfilerProxyCtrlAppendEnd = 18,

  /* Network defined events states */
  ncclProfilerNetPluginUpdate = 21,

  /* Kernel event states */
  ncclProfilerKernelChStop = 22,

  /* Group API States */
  ncclProfilerEndGroupApiStart = 23,
  ncclProfilerBeginGroupApiEnd = 24
} ncclProfilerEventState_t;

typedef ncclProfilerEventState_t ncclProfilerEventState_v4_t;

/**
 * @brief Get the name of ncclProfilerEventState_t enum value
 *
 * Uses static storage for string literals to avoid memory management issues.
 * Returns a string_view pointing to compile-time constants.
 *
 * @param state The event state enum value
 * @return std::string_view The name of the state
 */
inline std::string_view get_e_state_name(ncclProfilerEventState_v4_t state) {
  // Static storage for state name strings
  static constexpr const char* STATE_KERNEL_CH_STOP = "ncclProfilerKernelChStop";
  static constexpr const char* STATE_PROXY_OP_IN_PROGRESS = "ncclProfilerProxyOpInProgress_v4";
  static constexpr const char* STATE_PROXY_STEP_SEND_GPU_WAIT = "ncclProfilerProxyStepSendGPUWait";
  static constexpr const char* STATE_PROXY_STEP_SEND_PEER_WAIT = "ncclProfilerProxyStepSendPeerWait_v4";
  static constexpr const char* STATE_PROXY_STEP_SEND_WAIT = "ncclProfilerProxyStepSendWait";
  static constexpr const char* STATE_PROXY_STEP_RECV_WAIT = "ncclProfilerProxyStepRecvWait";
  static constexpr const char* STATE_PROXY_STEP_RECV_FLUSH_WAIT = "ncclProfilerProxyStepRecvFlushWait";
  static constexpr const char* STATE_PROXY_STEP_RECV_GPU_WAIT = "ncclProfilerProxyStepRecvGPUWait";
  static constexpr const char* STATE_PROXY_CTRL_IDLE = "ncclProfilerProxyCtrlIdle";
  static constexpr const char* STATE_PROXY_CTRL_ACTIVE = "ncclProfilerProxyCtrlActive";
  static constexpr const char* STATE_PROXY_CTRL_SLEEP = "ncclProfilerProxyCtrlSleep";
  static constexpr const char* STATE_PROXY_CTRL_WAKEUP = "ncclProfilerProxyCtrlWakeup";
  static constexpr const char* STATE_PROXY_CTRL_APPEND = "ncclProfilerProxyCtrlAppend";
  static constexpr const char* STATE_PROXY_CTRL_APPEND_END = "ncclProfilerProxyCtrlAppendEnd";
  static constexpr const char* STATE_UNKNOWN = "UnknownState";

  switch (state) {
    case ncclProfilerKernelChStop:
      return STATE_KERNEL_CH_STOP;
    case ncclProfilerProxyOpInProgress_v4:
      return STATE_PROXY_OP_IN_PROGRESS;
    case ncclProfilerProxyStepSendGPUWait:
      return STATE_PROXY_STEP_SEND_GPU_WAIT;
    case ncclProfilerProxyStepSendPeerWait_v4:
      return STATE_PROXY_STEP_SEND_PEER_WAIT;
    case ncclProfilerProxyStepSendWait:
      return STATE_PROXY_STEP_SEND_WAIT;
    case ncclProfilerProxyStepRecvWait:
      return STATE_PROXY_STEP_RECV_WAIT;
    case ncclProfilerProxyStepRecvFlushWait:
      return STATE_PROXY_STEP_RECV_FLUSH_WAIT;
    case ncclProfilerProxyStepRecvGPUWait:
      return STATE_PROXY_STEP_RECV_GPU_WAIT;
    case ncclProfilerProxyCtrlIdle:
      return STATE_PROXY_CTRL_IDLE;
    case ncclProfilerProxyCtrlActive:
      return STATE_PROXY_CTRL_ACTIVE;
    case ncclProfilerProxyCtrlSleep:
      return STATE_PROXY_CTRL_SLEEP;
    case ncclProfilerProxyCtrlWakeup:
      return STATE_PROXY_CTRL_WAKEUP;
    case ncclProfilerProxyCtrlAppend:
      return STATE_PROXY_CTRL_APPEND;
    case ncclProfilerProxyCtrlAppendEnd:
      return STATE_PROXY_CTRL_APPEND_END;
    default:
      return STATE_UNKNOWN;
  }
}

typedef union {
  struct {
    size_t transSize;
  } proxyStep;

  struct {
    int appendedProxyOps;
  } proxyCtrl;

  struct {
    void *data;
  } netPlugin;

  struct {
    uint64_t pTimer;
  } kernelCh;
} ncclProfilerEventStateArgs_v4_t;

class Tracer {
public:
  struct CommInitResult {
    std::unique_ptr<CommContext> context_;
    int activation_mask_;
  };

  CommInitResult on_comm_init(const char *commName, uint64_t commHash,
                              int nNodes, int nranks, int rank,
                              ncclDebugLogger_t logfn) {
    init();
    auto desc = comm_desc_dict_->create(commName, commHash, nNodes, nranks,
                                        rank, logfn);

    LOG_TRACE(desc, NCCL_INIT, "comm %s:%ld inited", commName, commHash);

    auto comm_ctx = std::make_unique<CommContext>(desc, file_dumper_);

    comm_ctx->report(CommInit{}, nullptr);

    return CommInitResult{
        .context_ = std::move(comm_ctx),
        .activation_mask_ = activate_mask_,
    };
  }

  static void on_comm_finalize(void *context) {
    auto sh_ctx = reinterpret_cast<std::shared_ptr<CommContext> *>(context);
    auto &ctx = **sh_ctx;
    CTX_LOG_INFO(ctx, 0, "comm finalized ", comm->hash_);
    ctx.report(CommDestroy{}, nullptr);
    delete sh_ctx;
  }

  static void on_event_start_v4(void *context, void **eHandle,
                                ncclProfilerEventDescr_v4_t *eDescr) {
    auto &sh_ctx = *static_cast<std::shared_ptr<CommContext> *>(context);
    auto hdl = sh_ctx->pool_.create(
      sh_ctx->self_.lock());
    hdl->set_parent(eDescr->parentObj);
    *eHandle = hdl;

    switch (eDescr->type) {
    case ncclProfileGroup:
      CTX_LOG_TRACE(*sh_ctx, 0, "group start %p, hdl %p", eDescr, hdl);
      hdl->on_profile_group_start(eDescr->parentObj, eDescr->rank);
      break;
    case ncclProfileColl:
      CTX_LOG_TRACE(*sh_ctx, 0, "event profile coll, parent %p",
                    eDescr->parentObj);
      hdl->on_coll_start(asProfileContextV4(eDescr));
      break;
    case ncclProfileP2p:
      CTX_LOG_TRACE(*sh_ctx, 0, "event profile p2p start, parent %p",
                    eDescr->parentObj);
      hdl->on_p2p_start(asProfileP2pContextV4(eDescr));
      break;
    case ncclProfileKernelCh:
      CTX_LOG_TRACE(*sh_ctx, 0, "event profile kernel ch start, parent %p",
                    eDescr->parentObj);
      hdl->on_kernel_ch_start(asProfileKernelChContextV4(eDescr));
      break;
    case ncclProfileProxyOp:
      CTX_LOG_TRACE(*sh_ctx, 0, "event profile proxy op start, parent %p",
                    eDescr->parentObj);
      hdl->on_proxy_op_start(asProfileProxyOpContextV4(eDescr));
      break;
    case ncclProfileProxyStep:
      CTX_LOG_TRACE(*sh_ctx, 0, "event profile proxy step start, parent %p",
                    eDescr->parentObj);
      hdl->on_proxy_step_start(asProfileProxyStepContextV4(eDescr));
      break;
    case ncclProfileProxyCtrl:
      CTX_LOG_TRACE(*sh_ctx, 0, "event profile proxy ctrl start, parent %p",
                    eDescr->parentObj);
      hdl->on_proxy_ctrl_start(asProfileProxyCtrlContextV4(eDescr));
      break;
    default:
      CTX_LOG_INFO(*sh_ctx, 0, "event profile type %d started",
                   int(eDescr->type));
    }
  }

  static void on_event_stop_v4(void *handle) {
    auto hdl = static_cast<EventHandle *>(handle);
    hdl->on_event_stop();
    hdl->deref();
  }

  static void
  on_event_state_record_v4(void *eHandle, ncclProfilerEventState_v4_t eState,
                           ncclProfilerEventStateArgs_v4_t *eStateArgs) {
    auto hdl = static_cast<EventHandle *>(eHandle);

    if (eState == ncclProfilerKernelChStop) {
      // Handle KernelCh stop event
      if (std::holds_alternative<ProfileKernelChContext>(hdl->event_context_)) {
        auto &ctx = std::get<ProfileKernelChContext>(hdl->event_context_);
        uint64_t stop_p_timer = eStateArgs ? eStateArgs->kernelCh.pTimer : 0;
        hdl->report(ProfileKernelChStop{
            .context_ = ctx,
            .stop_p_timer_ = stop_p_timer,
        });
        CTX_LOG_TRACE(
            *hdl->context_, 0,
            "kernel ch stop: channel_id=%u, start_timer=%lu, stop_timer=%lu",
            ctx.channel_id_, ctx.p_timer_, stop_p_timer);
      }
    } else if (eState == ncclProfilerProxyOpInProgress_v4) {
      // Handle ProxyOp InProgress event
      if (std::holds_alternative<ProfileProxyOpContext>(hdl->event_context_)) {
        auto &ctx = std::get<ProfileProxyOpContext>(hdl->event_context_);
        std::string_view state_name = get_e_state_name(eState);
        hdl->report(ProfileProxyOpStateRecord{
            .context_ = ctx,
            .state_name_ = state_name,
        });
        CTX_LOG_TRACE(*hdl->context_, 0,
                     "proxy op state: rank=%d, channel_id=%u, peer=%d, state=%s",
                     ctx.rank_, ctx.channel_id_, ctx.peer_, state_name.data());
      }
    } else if (eState == ncclProfilerProxyStepSendGPUWait ||
               eState == ncclProfilerProxyStepSendPeerWait_v4 ||
               eState == ncclProfilerProxyStepSendWait ||
               eState == ncclProfilerProxyStepRecvWait ||
               eState == ncclProfilerProxyStepRecvFlushWait ||
               eState == ncclProfilerProxyStepRecvGPUWait) {
      // Handle ProxyStep state record events
      if (std::holds_alternative<ProfileProxyStepContext>(hdl->event_context_)) {
        auto &ctx = std::get<ProfileProxyStepContext>(hdl->event_context_);
        size_t trans_size = eStateArgs ? eStateArgs->proxyStep.transSize : 0;
        std::string_view state_name = get_e_state_name(eState);
        hdl->report(ProfileProxyStepStateRecord{
            .context_ = ctx,
            .state_name_ = state_name,
            .trans_size_ = trans_size,
        });
        CTX_LOG_TRACE(*hdl->context_, 0,
                     "proxy step state: step=%d, state=%s, trans_size=%zu",
                     ctx.step_, state_name.data(), trans_size);
      }
    } else if (eState == ncclProfilerProxyCtrlIdle ||
               eState == ncclProfilerProxyCtrlActive ||
               eState == ncclProfilerProxyCtrlSleep ||
               eState == ncclProfilerProxyCtrlWakeup ||
               eState == ncclProfilerProxyCtrlAppend ||
               eState == ncclProfilerProxyCtrlAppendEnd) {
      // Handle ProxyCtrl state record events
      if (std::holds_alternative<ProfileProxyCtrlContext>(hdl->event_context_)) {
        auto &ctx = std::get<ProfileProxyCtrlContext>(hdl->event_context_);
        int appended_ops = eStateArgs ? eStateArgs->proxyCtrl.appendedProxyOps : 0;
        std::string_view state_name = get_e_state_name(eState);
        hdl->report(ProfileProxyCtrlStateRecord{
            .context_ = ctx,
            .state_name_ = state_name,
            .appended_proxy_ops_ = appended_ops,
        });
        CTX_LOG_TRACE(*hdl->context_, 0,
                     "proxy ctrl state: rank=%d, state=%s, appended_ops=%d",
                     ctx.rank_, state_name.data(), appended_ops);
      }
    } else {
      CTX_LOG_INFO(*hdl->context_, 0, "unhandled event state recorded state=%d", eState);
    }
  }

private:
  static ProfileCollContext
  asProfileContextV4(ncclProfilerEventDescr_v4_t *desc) {
    if (desc->type != ncclProfileColl) {
      throw std::runtime_error("Invalid event type for ProfileCollContext");
    }
    return ProfileCollContext{
        .rank_ = desc->rank,
        .seq_ = desc->coll.seqNumber,
        .func_ = desc->coll.func,
        .send_buff_ = desc->coll.sendBuff,
        .recv_buffer_ = desc->coll.recvBuff,
        .count_ = desc->coll.count,
        .root_ = desc->coll.root,
        .data_type_ = desc->coll.datatype,
        .n_channels_ = desc->coll.nChannels,
        .n_warps_ = desc->coll.nWarps,
        .algo_ = desc->coll.algo,
        .proto_ = desc->coll.proto,
    };
  }

  static ProfileP2pContext
  asProfileP2pContextV4(ncclProfilerEventDescr_v4_t *desc) {
    if (desc->type != ncclProfileP2p) {
      throw std::runtime_error("Invalid event type for ProfileP2pContext");
    }
    return ProfileP2pContext{
        .rank_ = desc->rank,
        .func_ = desc->p2p.func,
        .buff_ = desc->p2p.buff,
        .data_type_ = desc->p2p.datatype,
        .count_ = desc->p2p.count,
        .peer_ = desc->p2p.peer,
        .n_channels_ = desc->p2p.nChannels,
    };
  }

  static ProfileKernelChContext
  asProfileKernelChContextV4(ncclProfilerEventDescr_v4_t *desc) {
    if (desc->type != ncclProfileKernelCh) {
      throw std::runtime_error("Invalid event type for ProfileKernelChContext");
    }
    return ProfileKernelChContext{
        .rank_ = desc->rank,
        .channel_id_ = desc->kernelCh.channelId,
        .p_timer_ = desc->kernelCh.pTimer,
    };
  }

  static ProfileProxyOpContext
  asProfileProxyOpContextV4(ncclProfilerEventDescr_v4_t *desc) {
    if (desc->type != ncclProfileProxyOp) {
      throw std::runtime_error("Invalid event type for ProfileProxyOpContext");
    }
    return ProfileProxyOpContext{
        .rank_ = desc->rank,
        .pid_ = desc->proxyOp.pid,
        .channel_id_ = desc->proxyOp.channelId,
        .peer_ = desc->proxyOp.peer,
        .n_steps_ = desc->proxyOp.nSteps,
        .chunk_size_ = desc->proxyOp.chunkSize,
        .is_send_ = desc->proxyOp.isSend,
    };
  }

  static ProfileProxyStepContext
  asProfileProxyStepContextV4(ncclProfilerEventDescr_v4_t *desc) {
    if (desc->type != ncclProfileProxyStep) {
      throw std::runtime_error("Invalid event type for ProfileProxyStepContext");
    }
    return ProfileProxyStepContext{
        .rank_ = desc->rank,
        .step_ = desc->proxyStep.step,
    };
  }

  static ProfileProxyCtrlContext
  asProfileProxyCtrlContextV4(ncclProfilerEventDescr_v4_t *desc) {
    if (desc->type != ncclProfileProxyCtrl) {
      throw std::runtime_error("Invalid event type for ProfileProxyCtrlContext");
    }
    return ProfileProxyCtrlContext{
        .rank_ = desc->rank,
    };
  }

  void init() {
    std::call_once(init_once_, [this]() {
      auto fn = std::getenv("NCCL_TRACER_DUMP_FILE_NAME");
      if (fn == nullptr) {
        throw std::runtime_error(
            "NCCL_TRACER_DUMP_FILE_NAME environment variable not set");
      }
      char hostname[256];
      if (gethostname(hostname, sizeof(hostname)) != 0) {
        throw std::runtime_error("Failed to get hostname");
      }

      // real fn is ${fn}.${hostname}.${pid}
      std::string real_fn = std::string(fn) + "." + std::string(hostname) +
                            "." + std::to_string(getpid());
      file_dumper_ = std::make_shared<FileDumper<TraceDumpItem>>(real_fn);
      comm_desc_dict_ = std::make_shared<CommDescDict>();

      auto mask = std::getenv("NCCL_TRACER_ACTIVATE_MASK");
      if (mask == nullptr) {
        activate_mask_ = ncclProfileP2p | ncclProfileColl | ncclProfileProxyStep;
      } else {
        activate_mask_ = atoi(mask);
      }
    });
  }
  std::shared_ptr<Dumper> file_dumper_;
  std::shared_ptr<CommDescDict> comm_desc_dict_;
  std::once_flag init_once_;
  int activate_mask_{};
};

} // namespace nccltrace
