# NCCLTrace

A high-performance NCCL profiler plugin that traces and logs NCCL communication events with minimal overhead. The plugin uses lock-free data structures and compressed output to efficiently capture detailed information about NCCL collective and point-to-point operations.

## Features

- **NCCL Profiler Plugin**: Implements the NCCL v4 profiler plugin interface
- **Lock-Free Architecture**: Uses lock-free queues for minimal performance impact
- **High Compression Output**: Automatically compresses trace logs using zstd with high compression level (15)
- **Comprehensive Event Tracking**: Captures various NCCL events including:
  - Collective operations (AllReduce, Broadcast, etc.)
  - Point-to-point operations (Send/Recv)
  - Kernel channel operations
  - Proxy operations and steps
  - Group API calls
- **JSON Format**: Outputs structured JSON logs for easy parsing and analysis
- **Hierarchical Event Tracking**: Maintains parent-child relationships between events

## Architecture

The project consists of several key components:

- **`FileDumper`**: A templated class that uses a background thread to asynchronously dump traced events to compressed files
- **`LockFreeQueue`**: A lock-free FIFO queue implementation for thread-safe, wait-free data passing
- **`Tracer`**: The main tracer class that handles NCCL profiler callbacks
- **`CommContext`**: Per-communicator context that manages event lifecycle
- **`FreeReferencePool`**: Memory pool for efficient event handle allocation

## Requirements

- CMake 3.24 or higher
- C++23 compatible compiler
- libzstd library
- NCCL (for runtime usage)
- PyTorch with NCCL backend (for testing)

## Building

```bash
# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make -j$(nproc)

# Run tests
ctest
```

The build produces:
- `libnccltrace.so`: The main NCCL profiler plugin library
- Test executables for various components

## Usage

### Basic Usage

To use the NCCLTrace plugin with your NCCL application:

```bash
# Set the profiler plugin
export NCCL_PROFILER_PLUGIN=/path/to/libnccltrace.so

# Enable NCCL profiling
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=PROFILE

# Set output file path (optional, defaults to current directory)
export NCCL_TRACER_DUMP_FILE_NAME=/path/to/output_log

# Run your NCCL application
./your_nccl_application
```

The plugin will create zstd compressed trace files with `.zst` extension (e.g., `output_log.rank_0.zst`, `output_log.rank_1.zst`).

### Example: Tracing AllReduce Operations

```bash
#!/bin/bash
cd pytests
export NCCL_PROFILER_PLUGIN=../build/libnccltrace.so
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=PROFILE
export NCCL_TRACER_DUMP_FILE_NAME=$PWD/all_reduce_log

# Clean up old logs
rm -rf all_reduce_log*

# Run PyTorch multi-GPU AllReduce test
torchrun --nproc_per_node=2 --standalone all_reduce.py
```

This will trace NCCL AllReduce operations and save the logs to `all_reduce_log.rank_0.gz` and `all_reduce_log.rank_1.gz`.

### Example: Tracing Point-to-Point Operations

```bash
#!/bin/bash
cd pytests
export NCCL_PROFILER_PLUGIN=../build/libnccltrace.so
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=PROFILE
export NCCL_TRACER_DUMP_FILE_NAME=$PWD/p2p_log

# Clean up old logs
rm -rf p2p_log*

# Run PyTorch P2P test
torchrun --nproc_per_node=2 --standalone p2p.py
```

This will trace NCCL point-to-point (Send/Recv) operations.

## Output Format

The trace logs are written as newline-delimited JSON (NDJSON) with gzip compression. Each line contains a JSON object representing a traced event.

Example event structure:
```json
{
  "type": "ProfileCollStart",
  "timestamp": 1234567890123456,
  "comm": {
    "name": "my_comm",
    "hash": 123456789,
    "rank": 0,
    "n_ranks": 2,
    "n_nodes": 1
  },
  "context": {
    "rank": 0,
    "seq": 1,
    "func": "AllReduce",
    "count": 4096,
    "algo": "Ring",
    "proto": "Simple"
  },
  "id": "unique-event-id",
  "parent": {
    "id": "parent-event-id",
    "parent": null
  }
}
```

To read the compressed logs:
```bash
# View the raw JSON
zstdcat output_log.rank_0.zst

# Pretty print with jq
zstdcat output_log.rank_0.zst | jq .

# Or decompress first
zstd -d output_log.rank_0.zst
cat output_log.rank_0
```

## Testing

The project includes comprehensive tests:

```bash
cd build

# Run all tests
ctest

# Run specific tests
./nccltrace_lockfree_queue_test
./nccltrace_file_dumper_test
./nccltrace_comm_desc_test
./nccltrace_free_reference_pool_test
```

## Configuration

### Environment Variables

- `NCCL_PROFILER_PLUGIN`: Path to the `libnccltrace.so` library (required)
- `NCCL_DEBUG`: NCCL debug level (set to `INFO` or higher to enable tracing)
- `NCCL_DEBUG_SUBSYS`: NCCL debug subsystems (must include `PROFILE`)
- `NCCL_TRACER_DUMP_FILE_NAME`: Base path for output log files (default: `nccl_trace_log`)

### Event Filtering

By default, the plugin traces the following event types:
- Group API events
- Collective API events
- Collective operations
- Point-to-point operations
- Kernel channel events
- Proxy operations

The activation mask is set in the `Tracer::on_comm_init()` method and can be customized in the source code.

## Performance Considerations

- **Lock-Free Design**: The use of lock-free queues minimizes contention and overhead
- **Asynchronous I/O**: Background threads handle file writing to avoid blocking the main application
- **High Compression**: Zstd compression with level 15 significantly reduces disk I/O and storage requirements while maintaining good performance
- **Memory Pool**: Event handles are recycled using a free reference pool to reduce allocation overhead

## Project Structure

```
.
├── CMakeLists.txt              # Build configuration
├── nccltrace.cpp               # NCCL profiler plugin entry point
├── tracer.h                    # Main tracer implementation
├── file_dumper.h               # Asynchronous file writer with compression
├── lock_free_queue.h           # Lock-free queue implementation
├── lock_free_queue.cpp
├── dump_items.h                # Event data structures and serialization
├── comm_desc.h                 # Communicator descriptor
├── comm_desc_dict.h            # Communicator dictionary
├── free_reference_pool.h       # Memory pool for event handles
├── unix_nano_time.h            # High-resolution timestamp utilities
├── *_test.cpp                  # Unit tests
├── pytests/                    # Integration tests
│   ├── run_allreduce.sh        # AllReduce test script
│   ├── run_p2p.sh              # P2P test script
│   ├── all_reduce.py           # PyTorch AllReduce test
│   └── p2p.py                  # PyTorch P2P test
└── 3rd/                        # Third-party dependencies
    ├── catch2/                 # Testing framework
    └── json/                   # JSON library (nlohmann/json)
```

## License

Please refer to the project repository for license information.

## Contributing

Contributions are welcome! Please ensure that:
1. All tests pass before submitting changes
2. New features include appropriate tests
3. Code follows the existing style conventions

## Troubleshooting

### Plugin Not Loading
- Ensure `NCCL_PROFILER_PLUGIN` points to the correct `.so` file
- Verify that NCCL version supports the profiler plugin interface (v4)
- Check that `NCCL_DEBUG_SUBSYS` includes `PROFILE`

### No Output Files
- Verify write permissions for the output directory
- Check that `NCCL_DEBUG` is set to `INFO` or higher
- Ensure the application actually makes NCCL calls

### Compressed Files Are Corrupt
- The plugin flushes data periodically; corruption may indicate a crash
- Check system logs for out-of-memory or segmentation fault errors

## References

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL Profiler Plugin Interface](https://github.com/NVIDIA/nccl)

