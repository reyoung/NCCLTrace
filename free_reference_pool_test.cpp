#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <vector>
#include <map>
#include <chrono>
#include <memory>
#include "free_reference_pool.h"

using namespace nccltrace;
using namespace std::chrono_literals;

namespace {
inline std::shared_ptr<CommDesc> null_comm() { return {}; }
}

// Test struct for pooling
struct TestData {
    int value;
    std::string name;
    bool initialized;

    TestData() : value(0), name(""), initialized(false) {}
};

// Reset function for TestData
void reset_test_data(TestData* data) {
    data->value = 0;
    data->name = "";
    data->initialized = true;
}

TEST_CASE("FreeRefPool - Basic creation and reference counting", "[FreeRefPool]") {
    FreeRefPool<TestData> pool(reset_test_data, 100ms);

    SECTION("Create item has ref_count 1") {
        auto* item = pool.create(null_comm());
        REQUIRE(item != nullptr);
        REQUIRE(item->ref_count_.load() == 1);
        REQUIRE(item->initialized == true);
    }

    SECTION("Multiple creates return different items initially") {
        auto* item1 = pool.create(null_comm());
        auto* item2 = pool.create(null_comm());
        auto* item3 = pool.create(null_comm());

        REQUIRE(item1 != nullptr);
        REQUIRE(item2 != nullptr);
        REQUIRE(item3 != nullptr);
        REQUIRE(item1 != item2);
        REQUIRE(item2 != item3);
        REQUIRE(item1 != item3);
    }
}

TEST_CASE("FreeRefPool - Reference counting operations", "[FreeRefPool]") {
    FreeRefPool<TestData> pool(reset_test_data, 100ms);

    SECTION("ref() increments count") {
        auto* item = pool.create(null_comm());
        REQUIRE(item->ref_count_.load() == 1);

        item->ref();
        REQUIRE(item->ref_count_.load() == 2);

        item->ref();
        REQUIRE(item->ref_count_.load() == 3);
    }

    SECTION("deref() decrements count") {
        auto* item = pool.create(null_comm());
        item->ref();
        item->ref();
        REQUIRE(item->ref_count_.load() == 3);

        item->deref();
        REQUIRE(item->ref_count_.load() == 2);

        item->deref();
        REQUIRE(item->ref_count_.load() == 1);
    }

    SECTION("deref() to 0 records release time") {
        auto* item = pool.create(null_comm());
        REQUIRE(item->ref_count_.load() == 1);

        auto before = std::chrono::steady_clock::now();
        item->deref();
        auto after = std::chrono::steady_clock::now();

        REQUIRE(item->ref_count_.load() == 0);
        REQUIRE(item->last_released_time_ >= before);
        REQUIRE(item->last_released_time_ <= after);
    }
}

TEST_CASE("FreeRefPool - Item reuse with delay", "[FreeRefPool]") {
    FreeRefPool<TestData> pool(reset_test_data, 100ms);

    SECTION("Item is reused after min_reuse_interval") {
        auto* item1 = pool.create(null_comm());
        item1->value = 42;
        item1->name = "test";
        auto* addr1 = item1;

        // Release the item
        item1->deref();
        REQUIRE(item1->ref_count_.load() == 0);

        // Wait for reuse interval to pass
        std::this_thread::sleep_for(150ms);

        // Create again should reuse the same item
        auto* item2 = pool.create(null_comm());
        REQUIRE(item2 == addr1);
        REQUIRE(item2->ref_count_.load() == 1);
        // Reset function should have been called
        REQUIRE(item2->value == 0);
        REQUIRE(item2->name == "");
        REQUIRE(item2->initialized == true);
    }

    SECTION("Item is NOT reused before min_reuse_interval") {
        auto* item1 = pool.create(null_comm());
        auto* addr1 = item1;

        // Release the item
        item1->deref();

        // Immediately try to create (before interval)
        std::this_thread::sleep_for(10ms); // Small delay but less than 100ms
        auto* item2 = pool.create(null_comm());

        // Should get a different item
        REQUIRE(item2 != addr1);
    }
}

TEST_CASE("FreeRefPool - Pool growth", "[FreeRefPool]") {
    FreeRefPool<TestData> pool(reset_test_data, 100ms);

    SECTION("Pool grows when no free slots available") {
        std::vector<TestData*> items;

        // Create multiple items without releasing
        for (int i = 0; i < 10; ++i) {
            auto* item = pool.create(null_comm());
            REQUIRE(item != nullptr);
            items.push_back(item);
        }

        // All items should be different
        for (size_t i = 0; i < items.size(); ++i) {
            for (size_t j = i + 1; j < items.size(); ++j) {
                REQUIRE(items[i] != items[j]);
            }
        }
    }
}

TEST_CASE("FreeRefPool - Reset function is called", "[FreeRefPool]") {
    int reset_call_count = 0;

    auto custom_reset = [&reset_call_count](TestData* data) {
        reset_call_count++;
        data->value = 999;
        data->name = "reset";
        data->initialized = true;
    };

    FreeRefPool<TestData> pool(custom_reset, 100ms);

    SECTION("Reset called on first create") {
        reset_call_count = 0;
        auto* item = pool.create(null_comm());

        REQUIRE(reset_call_count == 1);
        REQUIRE(item->value == 999);
        REQUIRE(item->name == "reset");
    }

    SECTION("Reset called on reuse") {
        reset_call_count = 0;

        auto* item1 = pool.create(null_comm());
        REQUIRE(reset_call_count == 1);

        item1->value = 42;
        item1->name = "modified";
        item1->deref();

        std::this_thread::sleep_for(150ms);

        auto* item2 = pool.create(null_comm());
        REQUIRE(reset_call_count == 2);
        REQUIRE(item2->value == 999);
        REQUIRE(item2->name == "reset");
    }
}

TEST_CASE("FreeRefPool - Thread safety", "[FreeRefPool][multithread]") {
    FreeRefPool<TestData> pool(reset_test_data, 10ms);

    SECTION("Concurrent creates are thread-safe") {
        constexpr int num_threads = 10;
        constexpr int items_per_thread = 100;

        std::vector<std::thread> threads;
        std::vector<std::vector<TestData*>> all_items(num_threads);

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&pool, &all_items, t]() {
                for (int i = 0; i < items_per_thread; ++i) {
                    auto* item = pool.create(null_comm());
                    REQUIRE(item != nullptr);
                    REQUIRE(item->ref_count_.load() >= 1);
                    all_items[t].push_back(item);

                    // Simulate some work
                    item->value = t * items_per_thread + i;
                    std::this_thread::sleep_for(1ms);

                    item->deref();
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // All operations completed successfully without crashes
        REQUIRE(true);
    }
}

TEST_CASE("FreeRefPool - Multiple references", "[FreeRefPool]") {
    FreeRefPool<TestData> pool(reset_test_data, 100ms);

    SECTION("Item with multiple refs is not reused") {
        auto* item = pool.create(null_comm());
        item->ref(); // Add extra reference
        item->ref(); // Add another reference

        REQUIRE(item->ref_count_.load() == 3);

        // Release two references
        item->deref();
        item->deref();
        REQUIRE(item->ref_count_.load() == 1);

        std::this_thread::sleep_for(150ms);

        // Should not be reused yet (still has 1 reference)
        auto* item2 = pool.create(null_comm());
        REQUIRE(item2 != item);

        // Now release the last reference
        item->deref();
        REQUIRE(item->ref_count_.load() == 0);

        std::this_thread::sleep_for(150ms);

        // Now it should be reusable
        auto* item3 = pool.create(null_comm());
        // item3 might be item or item2 depending on search order
        // Just verify it's valid
        REQUIRE(item3 != nullptr);
    }
}

TEST_CASE("FreeRefPool - Custom types", "[FreeRefPool]") {
    struct ComplexData {
        std::vector<int> numbers;
        std::map<std::string, int> data_map;
        int counter;
    };

    auto reset_complex = [](ComplexData* data) {
        data->numbers.clear();
        data->data_map.clear();
        data->counter = 0;
    };

    FreeRefPool<ComplexData> pool(reset_complex, 50ms);

    SECTION("Complex types work correctly") {
        auto* item = pool.create(null_comm());
        REQUIRE(item != nullptr);
        REQUIRE(item->numbers.empty());
        REQUIRE(item->data_map.empty());
        REQUIRE(item->counter == 0);

        item->numbers = {1, 2, 3, 4, 5};
        item->data_map["key"] = 100;
        item->counter = 42;

        item->deref();
        std::this_thread::sleep_for(100ms);

        auto* item2 = pool.create(null_comm());
        REQUIRE(item2 == item);
        REQUIRE(item2->numbers.empty());
        REQUIRE(item2->data_map.empty());
        REQUIRE(item2->counter == 0);
    }
}

TEST_CASE("FreeRefPool - Stress test", "[FreeRefPool][stress]") {
    FreeRefPool<TestData> pool(reset_test_data, 10ms);

    SECTION("Many creates and releases") {
        for (int round = 0; round < 5; ++round) {
            std::vector<TestData*> items;

            // Create many items
            for (int i = 0; i < 50; ++i) {
                auto* item = pool.create(null_comm());
                REQUIRE(item != nullptr);
                item->value = i;
                items.push_back(item);
            }

            // Release half
            for (int i = 0; i < 25; ++i) {
                items[i]->deref();
            }

            std::this_thread::sleep_for(20ms);

            // Create more (should reuse released ones)
            for (int i = 0; i < 25; ++i) {
                auto* item = pool.create(null_comm());
                REQUIRE(item != nullptr);
            }

            // Release remaining from first batch
            for (int i = 25; i < 50; ++i) {
                items[i]->deref();
            }

            std::this_thread::sleep_for(20ms);
        }

        REQUIRE(true); // Test completed without issues
    }
}

TEST_CASE("FreeRefPool - CRTP on_obj_free callback", "[FreeRefPool][crtp]") {
    // Test data type with on_obj_free() callback
    struct DataWithCallback {
        int value;
        int* free_counter; // Pointer to track callback calls

        DataWithCallback() : value(0), free_counter(nullptr) {}

        void on_obj_free() {
            if (free_counter) {
                (*free_counter)++;
            }
        }
    };

    auto reset_func = [](DataWithCallback* data) {
        data->value = 0;
    };

    FreeRefPool<DataWithCallback> pool(reset_func, 50ms);

    SECTION("on_obj_free is called when ref_count drops to 0") {
        int free_count = 0;

        auto* item = pool.create(null_comm());
        item->value = 42;
        item->free_counter = &free_count;

        REQUIRE(free_count == 0);
        REQUIRE(item->ref_count_.load() == 1);

        // Deref to 0 should trigger on_obj_free
        item->deref();

        REQUIRE(item->ref_count_.load() == 0);
        REQUIRE(free_count == 1);
    }

    SECTION("on_obj_free is called each time ref_count drops to 0") {
        int free_count = 0;

        auto* item = pool.create(null_comm());
        item->free_counter = &free_count;

        // First drop to 0
        item->deref();
        REQUIRE(free_count == 1);

        std::this_thread::sleep_for(100ms);

        // Reuse the item
        auto* item2 = pool.create(null_comm());
        REQUIRE(item2 == item); // Should be the same item
        item2->free_counter = &free_count;

        // Second drop to 0
        item2->deref();
        REQUIRE(free_count == 2);
    }

    SECTION("on_obj_free is not called for intermediate deref") {
        int free_count = 0;

        auto* item = pool.create(null_comm());
        item->free_counter = &free_count;
        item->ref(); // ref_count = 2
        item->ref(); // ref_count = 3

        REQUIRE(item->ref_count_.load() == 3);

        // Deref but not to 0
        item->deref(); // ref_count = 2
        REQUIRE(free_count == 0);

        item->deref(); // ref_count = 1
        REQUIRE(free_count == 0);

        // Only when dropping to 0
        item->deref(); // ref_count = 0
        REQUIRE(free_count == 1);
    }

    SECTION("Types without on_obj_free still work") {
        // TestData doesn't have on_obj_free, should compile and work fine
        FreeRefPool<TestData> simple_pool(reset_test_data, 50ms);

        auto* item = simple_pool.create(null_comm());
        REQUIRE(item != nullptr);
        item->value = 123;

        item->deref();
        REQUIRE(item->ref_count_.load() == 0);
        // No crash, on_obj_free is simply not called
    }
}

TEST_CASE("FreeRefPool - CRTP on_obj_required_again callback", "[FreeRefPool][crtp]") {
    // Test data type with on_obj_required_again() callback
    struct DataWithRequireCallback {
        int value;
        int* require_counter; // Pointer to track callback calls

        DataWithRequireCallback() : value(0), require_counter(nullptr) {}

        void on_obj_required_again() {
            if (require_counter) {
                (*require_counter)++;
            }
        }
    };

    auto reset_func = [](DataWithRequireCallback* data) {
        data->value = 0;
    };

    FreeRefPool<DataWithRequireCallback> pool(reset_func, 50ms);

    SECTION("on_obj_required_again is NOT called during initial create") {
        int require_count = 0;

        auto* item = pool.create(null_comm());
        item->require_counter = &require_count;

        // Should NOT be called during create (ref_count goes from 0 to 1 via fetch_add)
        REQUIRE(require_count == 0);
        REQUIRE(item->ref_count_.load() == 1);
    }

    SECTION("on_obj_required_again is called when ref() from 0") {
        int require_count = 0;

        auto* item = pool.create(null_comm());
        item->require_counter = &require_count;

        // Release to 0
        item->deref();
        REQUIRE(item->ref_count_.load() == 0);
        REQUIRE(require_count == 0);

        // Now ref() from 0 should trigger callback
        item->ref();
        REQUIRE(item->ref_count_.load() == 1);
        REQUIRE(require_count == 1);
    }

    SECTION("on_obj_required_again is NOT called for non-zero ref()") {
        int require_count = 0;

        auto* item = pool.create(null_comm());
        item->require_counter = &require_count;

        REQUIRE(item->ref_count_.load() == 1);
        REQUIRE(require_count == 0);

        // ref() from 1 to 2, should NOT trigger callback
        item->ref();
        REQUIRE(item->ref_count_.load() == 2);
        REQUIRE(require_count == 0);

        // ref() from 2 to 3, should NOT trigger callback
        item->ref();
        REQUIRE(item->ref_count_.load() == 3);
        REQUIRE(require_count == 0);
    }

    SECTION("on_obj_required_again is called multiple times") {
        int require_count = 0;

        auto* item = pool.create(null_comm());
        item->require_counter = &require_count;

        // First cycle: deref to 0, then ref
        item->deref();
        REQUIRE(require_count == 0);
        item->ref();
        REQUIRE(require_count == 1);

        // Second cycle: deref to 0, then ref
        item->deref();
        REQUIRE(require_count == 1);
        item->ref();
        REQUIRE(require_count == 2);

        // Third cycle
        item->deref();
        item->ref();
        REQUIRE(require_count == 3);
    }

    SECTION("Both callbacks work together") {
        struct DataWithBothCallbacks {
            int* require_counter;
            int* free_counter;

            DataWithBothCallbacks() : require_counter(nullptr), free_counter(nullptr) {}

            void on_obj_required_again() {
                if (require_counter) (*require_counter)++;
            }

            void on_obj_free() {
                if (free_counter) (*free_counter)++;
            }
        };

        auto reset_both = [](DataWithBothCallbacks* data) {};
        FreeRefPool<DataWithBothCallbacks> both_pool(reset_both, 50ms);

        int require_count = 0;
        int free_count = 0;

        auto* item = both_pool.create(null_comm());
        item->require_counter = &require_count;
        item->free_counter = &free_count;

        // Initial state
        REQUIRE(require_count == 0);
        REQUIRE(free_count == 0);

        // Deref to 0 triggers on_obj_free
        item->deref();
        REQUIRE(require_count == 0);
        REQUIRE(free_count == 1);

        // Ref from 0 triggers on_obj_required_again
        item->ref();
        REQUIRE(require_count == 1);
        REQUIRE(free_count == 1);

        // Another cycle
        item->deref();
        REQUIRE(require_count == 1);
        REQUIRE(free_count == 2);

        item->ref();
        REQUIRE(require_count == 2);
        REQUIRE(free_count == 2);
    }
}

