#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <thread>
#include <vector>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <sstream>
#include <cstdlib>
#include "file_dumper.h"

using namespace nccltrace;

// Test struct that implements DumpItem concept
struct TestItem {
    int id;
    std::string message;
    
    TestItem(int i = 0, std::string msg = "") : id(i), message(std::move(msg)) {}
    
    void dump(std::ostream& out) const {
        out << "Item " << id << ": " << message << "\n";
    }
};

// Another test struct to verify concept works with different types
struct LogEntry {
    std::string timestamp;
    std::string level;
    std::string content;
    
    void dump(std::ostream& out) const {
        out << "[" << timestamp << "] [" << level << "] " << content << "\n";
    }
};

namespace {
struct ScopedEnvVar {
    std::string name;
    std::string old_value;
    bool had_old{false};

    explicit ScopedEnvVar(std::string n) : name(std::move(n)) {
        const char* v = std::getenv(name.c_str());
        if (v) {
            had_old = true;
            old_value = v;
        }
    }

    void set(const std::string& v) {
        ::setenv(name.c_str(), v.c_str(), 1);
    }

    void unset() {
        ::unsetenv(name.c_str());
    }

    ~ScopedEnvVar() {
        if (had_old) {
            ::setenv(name.c_str(), old_value.c_str(), 1);
        } else {
            ::unsetenv(name.c_str());
        }
    }
};

std::string read_file_content(const std::string& filename, bool binary = false) {
    std::ifstream file(filename, binary ? std::ios::binary : std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void cleanup_test_file(const std::string& filename) {
    std::filesystem::remove(filename);
}

void cleanup_both_outputs(const std::string& base) {
    cleanup_test_file(base + ".msgpack");
    cleanup_test_file(base + ".msgpack.gz");
}
}

TEST_CASE("FileDumper - Basic functionality", "[file_dumper]") {
    ScopedEnvVar gzip_env("NCCL_TRACER_DUMP_GZIP");
    gzip_env.set("0"); // disable gzip for these text-content tests

    const std::string test_base = "test_dumper_basic";
    const std::string test_file = test_base + ".msgpack";
    cleanup_both_outputs(test_base);

    SECTION("Constructor opens file successfully") {
        FileDumper<TestItem> dumper(test_base);
        std::filesystem::path p(test_file);
        REQUIRE(std::filesystem::exists(p));
    }
    
    SECTION("Push and dump single item") {
        {
            FileDumper<TestItem> dumper(test_base);
            dumper.push(TestItem(1, "First message"));
            
            // Give dumper time to process
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        
        std::string content = read_file_content(test_file);
        REQUIRE(content.find("Item 1: First message") != std::string::npos);
    }
    
    SECTION("Push and dump multiple items") {
        {
            FileDumper<TestItem> dumper(test_base, std::chrono::milliseconds(50));

            for (int i = 0; i < 10; ++i) {
                dumper.push(TestItem(i, "Message " + std::to_string(i)));
            }
            
            // Give dumper time to process all items
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        std::string content = read_file_content(test_file);
        
        // Check that all items were dumped
        for (int i = 0; i < 10; ++i) {
            std::string expected = "Item " + std::to_string(i) + ": Message " + std::to_string(i);
            REQUIRE(content.find(expected) != std::string::npos);
        }
    }
    
    cleanup_both_outputs(test_base);
}

TEST_CASE("FileDumper - Concurrent pushes", "[file_dumper]") {
    ScopedEnvVar gzip_env("NCCL_TRACER_DUMP_GZIP");
    gzip_env.set("0");

    const std::string test_base = "test_dumper_concurrent";
    const std::string test_file = test_base + ".msgpack";
    cleanup_both_outputs(test_base);

    const int num_threads = 4;
    const int items_per_thread = 25;
    
    {
        FileDumper<TestItem> dumper(test_base, std::chrono::milliseconds(10));

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&dumper, t, items_per_thread]() {
                for (int i = 0; i < items_per_thread; ++i) {
                    int id = t * items_per_thread + i;
                    dumper.push(TestItem(id, "Thread " + std::to_string(t) + " Item " + std::to_string(i)));
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Give dumper time to process all items
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    
    std::string content = read_file_content(test_file);
    
    // Check that all items from all threads were dumped
    int total_items = num_threads * items_per_thread;
    for (int i = 0; i < total_items; ++i) {
        std::string id_str = "Item " + std::to_string(i) + ":";
        REQUIRE(content.find(id_str) != std::string::npos);
    }
    
    cleanup_both_outputs(test_base);
}

TEST_CASE("FileDumper - Different DumpItem types", "[file_dumper]") {
    ScopedEnvVar gzip_env("NCCL_TRACER_DUMP_GZIP");
    gzip_env.set("0");

    const std::string test_base = "test_dumper_types";
    const std::string test_file = test_base + ".msgpack";
    cleanup_both_outputs(test_base);

    SECTION("LogEntry type") {
        {
            FileDumper<LogEntry> dumper(test_base, std::chrono::milliseconds(50));

            dumper.push(LogEntry{"2024-01-14 10:00:00", "INFO", "System started"});
            dumper.push(LogEntry{"2024-01-14 10:00:01", "WARNING", "Low memory"});
            dumper.push(LogEntry{"2024-01-14 10:00:02", "ERROR", "Connection failed"});
            
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        
        std::string content = read_file_content(test_file);
        REQUIRE(content.find("[2024-01-14 10:00:00] [INFO] System started") != std::string::npos);
        REQUIRE(content.find("[2024-01-14 10:00:01] [WARNING] Low memory") != std::string::npos);
        REQUIRE(content.find("[2024-01-14 10:00:02] [ERROR] Connection failed") != std::string::npos);
    }
    
    cleanup_both_outputs(test_base);
}

TEST_CASE("FileDumper - Stop functionality", "[file_dumper]") {
    ScopedEnvVar gzip_env("NCCL_TRACER_DUMP_GZIP");
    gzip_env.set("0");

    const std::string test_base = "test_dumper_stop";
    const std::string test_file = test_base + ".msgpack";
    cleanup_both_outputs(test_base);

    SECTION("Stop processes remaining items") {
        FileDumper<TestItem> dumper(test_base, std::chrono::milliseconds(100));

        // Push multiple items
        for (int i = 0; i < 5; ++i) {
            dumper.push(TestItem(i, "Message " + std::to_string(i)));
        }
        
        // Stop the dumper (should process all remaining items)
        dumper.stop();
        
        std::string content = read_file_content(test_file);
        
        // All items should be in the file
        for (int i = 0; i < 5; ++i) {
            std::string expected = "Item " + std::to_string(i) + ": Message " + std::to_string(i);
            REQUIRE(content.find(expected) != std::string::npos);
        }
    }
    
    SECTION("Push after stop throws exception") {
        FileDumper<TestItem> dumper(test_base);
        dumper.stop();
        
        REQUIRE_THROWS_AS(dumper.push(TestItem(1, "Should fail")), std::runtime_error);
    }
    
    cleanup_both_outputs(test_base);
}

TEST_CASE("FileDumper - Sleep duration", "[file_dumper]") {
    ScopedEnvVar gzip_env("NCCL_TRACER_DUMP_GZIP");
    gzip_env.set("0");

    const std::string test_base = "test_dumper_sleep";
    const std::string test_file = test_base + ".msgpack";
    cleanup_both_outputs(test_base);

    SECTION("Custom sleep duration") {
        // Test with a very short sleep duration
        FileDumper<TestItem> dumper(test_base, std::chrono::milliseconds(10));

        dumper.push(TestItem(1, "Test"));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        std::string content = read_file_content(test_file);
        REQUIRE(content.find("Item 1: Test") != std::string::npos);
    }
    
    cleanup_both_outputs(test_base);
}

TEST_CASE("FileDumper - File append mode", "[file_dumper]") {
    ScopedEnvVar gzip_env("NCCL_TRACER_DUMP_GZIP");
    gzip_env.set("0");

    const std::string test_base = "test_dumper_append";
    const std::string test_file = test_base + ".msgpack";
    cleanup_both_outputs(test_base);

    // First dumper instance
    {
        FileDumper<TestItem> dumper1(test_base);
        dumper1.push(TestItem(1, "First dumper"));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    // Second dumper instance (should append to the same file)
    {
        FileDumper<TestItem> dumper2(test_base);
        dumper2.push(TestItem(2, "Second dumper"));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    std::string content = read_file_content(test_file);
    REQUIRE(content.find("Item 1: First dumper") != std::string::npos);
    REQUIRE(content.find("Item 2: Second dumper") != std::string::npos);
    
    cleanup_both_outputs(test_base);
}

TEST_CASE("FileDumper - gzip toggle + filename suffix", "[file_dumper]") {
    const std::string test_base = "test_dumper_gzip_toggle";
    cleanup_both_outputs(test_base);

    SECTION("Default is gzip enabled, suffix .msgpack.gz and gzip header") {
        ScopedEnvVar gzip_env("NCCL_TRACER_DUMP_GZIP");
        gzip_env.unset();

        {
            FileDumper<TestItem> dumper(test_base);
            dumper.push(TestItem(1, "Hello"));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        REQUIRE(std::filesystem::exists(test_base + ".msgpack.gz"));
        REQUIRE_FALSE(std::filesystem::exists(test_base + ".msgpack"));

        // gzip magic bytes: 1f 8b
        auto bytes = read_file_content(test_base + ".msgpack.gz", /*binary*/true);
        REQUIRE(bytes.size() >= 2);
        REQUIRE(static_cast<unsigned char>(bytes[0]) == 0x1f);
        REQUIRE(static_cast<unsigned char>(bytes[1]) == 0x8b);
    }

    SECTION("When NCCL_TRACER_DUMP_GZIP=0, suffix .msgpack and plaintext dump") {
        ScopedEnvVar gzip_env("NCCL_TRACER_DUMP_GZIP");
        gzip_env.set("0");

        {
            FileDumper<TestItem> dumper(test_base);
            dumper.push(TestItem(2, "World"));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        REQUIRE(std::filesystem::exists(test_base + ".msgpack"));
        REQUIRE_FALSE(std::filesystem::exists(test_base + ".msgpack.gz"));

        auto content = read_file_content(test_base + ".msgpack");
        REQUIRE(content.find("Item 2: World") != std::string::npos);
    }

    cleanup_both_outputs(test_base);
}
