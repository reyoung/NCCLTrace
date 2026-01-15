#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <thread>
#include <vector>
#include <chrono>
#include "lock_free_queue.h"

using namespace nccltrace;

TEST_CASE("LockFreeQueue - Basic Operations", "[lockfree][basic]") {
    LockFreeQueue<int> queue;

    SECTION("Empty queue returns nullopt from try_pop") {
        auto result = queue.try_pop();
        REQUIRE_FALSE(result.has_value());
    }

    SECTION("Push and try_pop single item") {
        queue.push(42);
        auto result = queue.try_pop();
        REQUIRE(result.has_value());
        REQUIRE(result.value() == 42);
    }

    SECTION("Push multiple items and pop them") {
        queue.push(1);
        queue.push(2);
        queue.push(3);

        auto r1 = queue.try_pop();
        auto r2 = queue.try_pop();
        auto r3 = queue.try_pop();

        REQUIRE(r1.has_value());
        REQUIRE(r2.has_value());
        REQUIRE(r3.has_value());
        REQUIRE(r1.value() == 1);
        REQUIRE(r2.value() == 2);
        REQUIRE(r3.value() == 3);
    }

    SECTION("Try_pop on empty queue after exhausting") {
        queue.push(100);
        queue.try_pop();

        auto result = queue.try_pop();
        REQUIRE_FALSE(result.has_value());
    }
}

TEST_CASE("LockFreeQueue - Blocking Pop", "[lockfree][blocking]") {
    LockFreeQueue<int> queue;

    SECTION("Pop blocks until item is available") {
        bool popped = false;
        int value = 0;

        std::thread producer([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            queue.push(123);
        });

        std::thread consumer([&]() {
            value = queue.pop();
            popped = true;
        });

        producer.join();
        consumer.join();

        REQUIRE(popped);
        REQUIRE(value == 123);
    }

    SECTION("Multiple concurrent pops") {
        const int num_items = 100;
        std::vector<std::thread> producers;
        std::vector<std::thread> consumers;
        std::atomic<int> sum_produced(0);
        std::atomic<int> sum_consumed(0);

        for (int i = 0; i < num_items; ++i) {
            producers.emplace_back([&, i]() {
                queue.push(i);
                sum_produced += i;
            });
        }

        for (int i = 0; i < num_items; ++i) {
            consumers.emplace_back([&]() {
                int value = queue.pop();
                sum_consumed += value;
            });
        }

        for (auto& t : producers) t.join();
        for (auto& t : consumers) t.join();

        REQUIRE(sum_produced.load() == sum_consumed.load());
    }
}

TEST_CASE("LockFreeQueue - Concurrent Operations", "[lockfree][concurrent]") {
    LockFreeQueue<int> queue;

    SECTION("Multiple producers and consumers") {
        const int num_producers = 4;
        const int num_consumers = 4;
        const int items_per_producer = 100;

        std::vector<std::thread> producers;
        std::vector<std::thread> consumers;
        std::atomic<int> items_consumed(0);

        // Producers
        for (int p = 0; p < num_producers; ++p) {
            producers.emplace_back([&, p]() {
                for (int i = 0; i < items_per_producer; ++i) {
                    queue.push(p * items_per_producer + i);
                }
            });
        }

        // Consumers
        for (int c = 0; c < num_consumers; ++c) {
            consumers.emplace_back([&]() {
                int count = 0;
                while (count < num_producers * items_per_producer / num_consumers) {
                    auto result = queue.try_pop();
                    if (result.has_value()) {
                        count++;
                    } else {
                        std::this_thread::sleep_for(std::chrono::microseconds(10));
                    }
                }
                items_consumed += count;
            });
        }

        for (auto& t : producers) t.join();
        for (auto& t : consumers) t.join();

        REQUIRE(items_consumed.load() == num_producers * items_per_producer);
    }

    SECTION("Stress test - many operations") {
        const int total_items = 10000;
        std::atomic<int> producer_count(0);
        std::atomic<int> consumer_count(0);

        std::thread producer([&]() {
            for (int i = 0; i < total_items; ++i) {
                queue.push(i);
                producer_count++;
            }
        });

        std::thread consumer([&]() {
            int count = 0;
            while (count < total_items) {
                auto result = queue.try_pop();
                if (result.has_value()) {
                    count++;
                    consumer_count++;
                }
            }
        });

        producer.join();
        consumer.join();

        REQUIRE(producer_count.load() == total_items);
        REQUIRE(consumer_count.load() == total_items);
    }
}

TEST_CASE("LockFreeQueue - Type Tests", "[lockfree][types]") {
    SECTION("String type") {
        LockFreeQueue<std::string> queue;
        
        queue.push("hello");
        queue.push("world");
        
        auto r1 = queue.try_pop();
        auto r2 = queue.try_pop();
        
        REQUIRE(r1.has_value());
        REQUIRE(r2.has_value());
        REQUIRE(r1.value() == "hello");
        REQUIRE(r2.value() == "world");
    }

    SECTION("Move-only type") {
        LockFreeQueue<std::unique_ptr<int>> queue;
        
        auto p1 = std::make_unique<int>(42);
        auto p2 = std::make_unique<int>(100);
        
        queue.push(std::move(p1));
        queue.push(std::move(p2));
        
        auto r1 = queue.try_pop();
        auto r2 = queue.try_pop();
        
        REQUIRE(r1.has_value());
        REQUIRE(r2.has_value());
        REQUIRE(*r1.value() == 42);
        REQUIRE(*r2.value() == 100);
    }
}

TEST_CASE("LockFreeQueue - Ordering", "[lockfree][ordering]") {
    LockFreeQueue<int> queue;

    SECTION("FIFO ordering") {
        for (int i = 0; i < 100; ++i) {
            queue.push(i);
        }

        for (int i = 0; i < 100; ++i) {
            auto result = queue.try_pop();
            REQUIRE(result.has_value());
            REQUIRE(result.value() == i);
        }
    }
}

TEST_CASE("LockFreeQueue - Destructor", "[lockfree][destructor]") {
    SECTION("Queue destruction cleans up all nodes") {
        {
            LockFreeQueue<int> queue;
            for (int i = 0; i < 1000; ++i) {
                queue.push(i);
            }
        }
        // Queue should be destroyed without issues
        REQUIRE(true);
    }
}

TEST_CASE("LockFreeQueue - Size", "[lockfree][size]") {
    LockFreeQueue<int> queue;

    SECTION("Empty queue has size 0") {
        REQUIRE(queue.size() == 0);
    }

    SECTION("Size increases with push") {
        REQUIRE(queue.size() == 0);
        queue.push(1);
        REQUIRE(queue.size() == 1);
        queue.push(2);
        REQUIRE(queue.size() == 2);
        queue.push(3);
        REQUIRE(queue.size() == 3);
    }

    SECTION("Size decreases with try_pop") {
        queue.push(1);
        queue.push(2);
        queue.push(3);
        REQUIRE(queue.size() == 3);

        queue.try_pop();
        REQUIRE(queue.size() == 2);

        queue.try_pop();
        REQUIRE(queue.size() == 1);

        queue.try_pop();
        REQUIRE(queue.size() == 0);
    }

    SECTION("Size with mixed operations") {
        queue.push(1);
        queue.push(2);
        REQUIRE(queue.size() == 2);

        queue.try_pop();
        REQUIRE(queue.size() == 1);

        queue.push(3);
        queue.push(4);
        REQUIRE(queue.size() == 3);

        queue.try_pop();
        queue.try_pop();
        REQUIRE(queue.size() == 1);
    }

    SECTION("Size with concurrent operations") {
        const int num_threads = 4;
        const int items_per_thread = 100;

        std::vector<std::thread> producers;
        for (int t = 0; t < num_threads; ++t) {
            producers.emplace_back([&queue, items_per_thread, t]() {
                for (int i = 0; i < items_per_thread; ++i) {
                    queue.push(t * items_per_thread + i);
                }
            });
        }

        for (auto& thread : producers) {
            thread.join();
        }

        // Size should be approximately num_threads * items_per_thread
        // Allow some tolerance due to concurrent operations
        size_t final_size = queue.size();
        REQUIRE(final_size == num_threads * items_per_thread);

        // Consume all items
        int consumed = 0;
        while (queue.try_pop().has_value()) {
            consumed++;
        }

        REQUIRE(consumed == num_threads * items_per_thread);
        REQUIRE(queue.size() == 0);
    }

    SECTION("Try_pop on empty queue doesn't affect size") {
        REQUIRE(queue.size() == 0);
        auto result = queue.try_pop();
        REQUIRE_FALSE(result.has_value());
        REQUIRE(queue.size() == 0);
    }
}

