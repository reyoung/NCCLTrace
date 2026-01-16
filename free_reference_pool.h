#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <type_traits>
#include <vector>
#include "comm_desc.h"

namespace nccltrace {

/**
 * @brief Base class for items stored in FreeRefPool (CRTP pattern)
 *
 * Provides reference counting and tracking of when the item was last released.
 * Uses atomic operations for thread-safe reference counting.
 *
 * CRTP feature: If the derived class implements on_obj_free(), it will be
 * called when ref_count drops to 0. This allows custom cleanup logic before
 * reuse.
 *
 * @tparam Derived The derived class (CRTP parameter)
 */
template <typename Derived> struct PoolItem {
  std::atomic<int> ref_count_{0}; // Current number of references to this item

  // Timestamp when ref_count dropped to 0 (last release time)
  std::chrono::steady_clock::time_point last_released_time_;

  int64_t rand_id_{0};

  /**
   * @brief Increment reference count
   * Thread-safe increment operation.
   * If ref_count was 0, calls on_obj_required_again() if implemented (CRTP).
   */
  void ref() {
    int prev = ref_count_.fetch_add(1);
    if (prev == 0) {
      // static_assert(has_on_obj_required_again<Derived>::value);
      // Object is being acquired again from released state
      // Call on_obj_required_again() if it exists in Derived class (CRTP)
      if constexpr (has_on_obj_required_again<Derived>::value) {
        static_cast<Derived *>(this)->on_obj_required_again();
      }
    }
  }

  /**
   * @brief Decrement reference count
   * When the count reaches 0, records the release timestamp for reuse logic.
   * If Derived class has on_obj_free() method, it will be called.
   */
  void deref() {
    int prev = ref_count_.fetch_sub(1);
    if (prev == 1) {
      // Last reference released, record the time for min_reuse_interval check
      last_released_time_ = std::chrono::steady_clock::now();

      // Call on_obj_free() if it exists in Derived class (CRTP)
      if constexpr (has_on_obj_free<Derived>::value) {
        static_cast<Derived *>(this)->on_obj_free();
      }
    }
  }

private:
  // SFINAE helper to detect if Derived has on_obj_free() method
  template <typename T, typename = void>
  struct has_on_obj_free : std::false_type {};

  template <typename T>
  struct has_on_obj_free<T,
                         std::void_t<decltype(std::declval<T>().on_obj_free())>>
      : std::true_type {};

  // SFINAE helper to detect if Derived has on_obj_required_again() method
  template <typename T, typename = void>
  struct has_on_obj_required_again : std::false_type {};

  template <typename T>
  struct has_on_obj_required_again<
      T, std::void_t<decltype(std::declval<T>().on_obj_required_again())>>
      : std::true_type {};
};

/**
 * @brief Object pool with reference counting and delayed reuse
 *
 * FreeRefPool manages a pool of objects with the following features:
 * 1. Reference counting: Objects are only reusable when ref_count == 0
 * 2. Delayed reuse: Objects can only be reused after min_reuse_interval has
 * passed
 * 3. Auto-growth: Pool doubles in size when no free slots are available
 * 4. Thread-safe: Uses mutex to protect pool operations
 *
 * Usage pattern:
 *   - create(): Get a fresh or reused object from the pool (ref_count = 1)
 *   - item->ref(): Add a reference when passing to another owner
 *   - item->deref(): Release a reference when done
 *   - When ref_count reaches 0, the item can be reused after min_reuse_interval
 *
 * @tparam T The type of object to pool (must be default constructible)
 */
template <typename T> class FreeRefPool {
public:
  /**
   * @brief Pool item that combines user type T with PoolItem functionality
   * Inherits both T (user data) and PoolItem (reference counting with CRTP).
   * The Item itself is passed as the CRTP parameter to PoolItem.
   */
  struct Item : T, PoolItem<Item> {};

  /**
   * @brief Construct a new FreeRefPool
   *
   * @param reset Function to reset an item before reuse
   *              Can be a function pointer, lambda, or any callable object
   *              Called with the T* pointer to reinitialize the object
   * @param min_reuse_interval Minimum time before a released item can be reused
   *                           Default: 60 seconds
   *                           Prevents premature reuse that could cause issues
   */
  explicit FreeRefPool(std::function<void(T *)> reset,
                       std::chrono::nanoseconds min_reuse_interval =
                           std::chrono::milliseconds(60000))
      : reset_(std::move(reset)), min_reuse_interval_(min_reuse_interval),
        engine_(std::random_device()()) {}

  /**
   * @brief Create or reuse an item from the pool
   *
   * Search strategy:
   * 1. Search from free_idx_ to end for a reusable item
   * 2. If not found, search from beginning to free_idx_
   * 3. If still not found, double the pool size
   *
   * A reusable item must satisfy:
   * - ref_count == 0 (no active references)
   * - Time since last release >= min_reuse_interval
   *
   * @return Item* Pointer to a fresh/reused item with ref_count = 1
   * @throws std::runtime_error if internal error occurs (null item)
   */
  Item *create(std::shared_ptr<CommDesc> comm) {
    auto now = std::chrono::steady_clock::now();

    // Lambda to check if an item is reusable
    auto free_slot = [this, &now](std::unique_ptr<Item> &item) {
      if (!item) {
        throw std::runtime_error("null item, internal error");
      }

      // Check both ref_count and time since last release
      if (now - item->last_released_time_ >= min_reuse_interval_) {
        return item->ref_count_.load() == 0;
      }
      return false;
    };

    std::unique_lock lock(mutex_);

    // First search: from free_idx_ to end
    auto it = std::find_if(items_.begin() + free_idx_, items_.end(), free_slot);
    if (it != items_.end()) {
      return create_with_free_slot(it - items_.begin());
    }

    // Second search: from beginning to free_idx_
    it = std::find_if(items_.begin(), items_.begin() + free_idx_, free_slot);
    if (it != items_.begin() + free_idx_) {
      return create_with_free_slot(it - items_.begin());
    }

    // No free slot found, grow the pool (double the size)
    size_t old_size = items_.size();
    size_t new_size = old_size == 0 ? 1 : old_size * 2;
    LOG_INFO(comm, 0, "resize comm buffer to size %lld", new_size);
    items_.resize(new_size);
    for (size_t i = old_size; i < new_size; ++i) {
      items_[i] = std::make_unique<Item>();
    }
    return create_with_free_slot(old_size);
  }

private:
  /**
   * @brief Initialize an item at the given offset for use
   *
   * @param offset Index in the items_ vector
   * @return Item* Pointer to the initialized item (ref_count = 1)
   */
  Item *create_with_free_slot(size_t offset) {
    auto &item = items_[offset];
    // Directly set ref_count to 1 using atomic fetch_add to avoid
    // triggering on_obj_required_again callback during pool creation
    item->ref_count_.fetch_add(1);
    reset_(item.get());         // Reset the item to initial state
    item->rand_id_ = engine_(); // Assign a random ID
    free_idx_ = offset + 1;     // Update hint for next search
    return item.get();
  }

  std::function<void(T *)> reset_; // Function to reset items before reuse
  std::chrono::nanoseconds min_reuse_interval_; // Minimum time before reuse
  std::default_random_engine engine_;
  std::mutex mutex_;                         // Protects items_ and free_idx_
  std::vector<std::unique_ptr<Item>> items_; // Pool of items
  size_t free_idx_{0}; // Hint for where to start searching
};

} // namespace nccltrace