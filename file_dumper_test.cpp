#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <thread>
#include <vector>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <zstd.h>
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

// Helper function to read zstd compressed file content
std::string read_file_content(const std::string& filename) {
    // Read compressed file
    std::string compressed_filename = filename + ".zst";
    std::ifstream file(compressed_filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + compressed_filename);
    }
    
    // Read all compressed data
    std::vector<char> compressed_data((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());
    file.close();

    if (compressed_data.empty()) {
        return "";
    }

    // Get decompressed size
    unsigned long long const decompressed_size = ZSTD_getFrameContentSize(
        compressed_data.data(), compressed_data.size());

    if (decompressed_size == ZSTD_CONTENTSIZE_ERROR ||
        decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
        throw std::runtime_error("Failed to determine decompressed size");
    }

    // Decompress data
    std::vector<char> decompressed_data(decompressed_size);
    size_t const result = ZSTD_decompress(
        decompressed_data.data(), decompressed_data.size(),
        compressed_data.data(), compressed_data.size());

    if (ZSTD_isError(result)) {
        throw std::runtime_error("Decompression error: " + std::string(ZSTD_getErrorName(result)));
    }

    return std::string(decompressed_data.begin(), decompressed_data.end());
}

// Helper function to clean up test files
void cleanup_test_file(const std::string& filename) {
    std::filesystem::remove(filename + ".zst");
}

TEST_CASE("FileDumper - Basic functionality", "[file_dumper]") {
    const std::string test_file = "test_dumper_basic.log";
    cleanup_test_file(test_file);
    
    SECTION("Constructor opens file successfully") {
        FileDumper<TestItem> dumper(test_file);
        std::filesystem::path p(test_file);
        REQUIRE(std::filesystem::exists(p));
    }
    
    SECTION("Push and dump single item") {
        {
            FileDumper<TestItem> dumper(test_file);
            dumper.push(TestItem(1, "First message"));
            
            // Give dumper time to process
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        
        std::string content = read_file_content(test_file);
        REQUIRE(content.find("Item 1: First message") != std::string::npos);
    }
    
    SECTION("Push and dump multiple items") {
        {
            FileDumper<TestItem> dumper(test_file, std::chrono::milliseconds(50));
            
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
    
    cleanup_test_file(test_file);
}

TEST_CASE("FileDumper - Concurrent pushes", "[file_dumper]") {
    const std::string test_file = "test_dumper_concurrent.log";
    cleanup_test_file(test_file);
    
    const int num_threads = 4;
    const int items_per_thread = 25;
    
    {
        FileDumper<TestItem> dumper(test_file, std::chrono::milliseconds(10));
        
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
    
    cleanup_test_file(test_file);
}

TEST_CASE("FileDumper - Different DumpItem types", "[file_dumper]") {
    const std::string test_file = "test_dumper_types.log";
    cleanup_test_file(test_file);
    
    SECTION("LogEntry type") {
        {
            FileDumper<LogEntry> dumper(test_file, std::chrono::milliseconds(50));
            
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
    
    cleanup_test_file(test_file);
}

TEST_CASE("FileDumper - Stop functionality", "[file_dumper]") {
    const std::string test_file = "test_dumper_stop.log";
    cleanup_test_file(test_file);
    
    SECTION("Stop processes remaining items") {
        FileDumper<TestItem> dumper(test_file, std::chrono::milliseconds(100));
        
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
        FileDumper<TestItem> dumper(test_file);
        dumper.stop();
        
        REQUIRE_THROWS_AS(dumper.push(TestItem(1, "Should fail")), std::runtime_error);
    }
    
    cleanup_test_file(test_file);
}

TEST_CASE("FileDumper - Sleep duration", "[file_dumper]") {
    const std::string test_file = "test_dumper_sleep.log";
    cleanup_test_file(test_file);
    
    SECTION("Custom sleep duration") {
        // Test with a very short sleep duration
        FileDumper<TestItem> dumper(test_file, std::chrono::milliseconds(10));
        
        dumper.push(TestItem(1, "Test"));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        std::string content = read_file_content(test_file);
        REQUIRE(content.find("Item 1: Test") != std::string::npos);
    }
    
    cleanup_test_file(test_file);
}

TEST_CASE("FileDumper - File append mode", "[file_dumper]") {
    const std::string test_file = "test_dumper_append.log";
    cleanup_test_file(test_file);
    
    // First dumper instance
    {
        FileDumper<TestItem> dumper1(test_file);
        dumper1.push(TestItem(1, "First dumper"));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    // Second dumper instance (should append to the same file)
    {
        FileDumper<TestItem> dumper2(test_file);
        dumper2.push(TestItem(2, "Second dumper"));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    std::string content = read_file_content(test_file);
    REQUIRE(content.find("Item 1: First dumper") != std::string::npos);
    REQUIRE(content.find("Item 2: Second dumper") != std::string::npos);
    
    cleanup_test_file(test_file);
}
