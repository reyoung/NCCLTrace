
#include "tracer.h"
#include <mutex>

namespace nccltrace {
typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7,
  ncclNumResults = 8
} ncclResult_t;
static Tracer tracer_;
static ncclResult_t init_v4(void **context, int *eActivationMask,
                            const char *commName, uint64_t commHash, int nNodes,
                            int nranks, int rank, ncclDebugLogger_t logfn) {
  auto res =
      tracer_.on_comm_init(commName, commHash, nNodes, nranks, rank, logfn);
  *eActivationMask = res.activation_mask_;
  *context = new std::shared_ptr(std::move(res.context_));
  return ncclSuccess;
}
static ncclResult_t finalize_v4(void *context) {
  Tracer::on_comm_finalize(context);
  return ncclSuccess;
}

//  - eHandle: return event handle for supplied event descriptor object
static ncclResult_t startEvent_v4(void *context, void **eHandle,
                                  ncclProfilerEventDescr_v4_t *eDescr) {
  Tracer::on_event_start_v4(context, eHandle, eDescr);
  return ncclSuccess;
}

ncclResult_t stopEvent_v4(void *eHandle) {
  Tracer::on_event_stop_v4(eHandle);
  return ncclSuccess;
}
ncclResult_t recordEventStateV4(void *eHandle,
                                ncclProfilerEventState_v4_t eState,
                                ncclProfilerEventStateArgs_v4_t *eStateArgs) {
  Tracer::on_event_state_record_v4(eHandle, eState, eStateArgs);
  return ncclSuccess;
}

typedef struct {
  const char *name;

  // init - initialize the profiler plugin
  // Input
  //  - context        : opaque profiler context object for separating profiler
  //  behavior across comms
  //  - commName       : user assigned communicator name
  //  - commHash       : communicator id
  //  - nNodes         : number of nodes in communicator
  //  - nranks         : number of ranks in communicator
  //  - rank           : rank identifier in communicator
  //  - logfn          : logger function
  // Output
  //  - eActivationMask: bitmask of active events set by the plugin
  ncclResult_t (*init)(void **context, int *eActivationMask,
                       const char *commName, uint64_t commHash, int nNodes,
                       int nranks, int rank, ncclDebugLogger_t logfn);

  // startEvent - initialize and start a new event for the supplied event
  // descriptor inside the eventset Input
  //  - context: opaque profiler context object
  //  - eDescr : pointer to ncclProfilerEventDescr_t object
  // Output
  //  - eHandle: return event handle for supplied event descriptor object
  ncclResult_t (*startEvent)(void *context, void **eHandle,
                             ncclProfilerEventDescr_v4_t *eDescr);

  // stopEvent - stop/finalize an event inside and event set
  // Input
  //  - eHandle: handle to event object
  ncclResult_t (*stopEvent)(void *eHandle);

  // recordEventState - record event state transitions and event attribute
  // updates Input
  //  - eHandle   : handle to event object created through startEvent
  //  - eStateArgs: optional argument used to capture event attribute updates
  //  associated with the state transition
  //  - eState    : event state transition
  ncclResult_t (*recordEventState)(void *eHandle,
                                   ncclProfilerEventState_v4_t eState,
                                   ncclProfilerEventStateArgs_v4_t *eStateArgs);

  // finalize - finalize the profiler plugin
  // Input
  //  - context: opaque profiler context object
  ncclResult_t (*finalize)(void *context);
} ncclProfiler_v4_t;

} // namespace nccltrace

extern "C" {
nccltrace::ncclProfiler_v4_t ncclProfiler_v4{
    "NCCLTracer",
    nccltrace::init_v4,
    nccltrace::startEvent_v4,
    nccltrace::stopEvent_v4,
    nccltrace::recordEventStateV4,
    nccltrace::finalize_v4,
};
}