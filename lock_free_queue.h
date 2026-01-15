#pragma once

#include <optional>
#include <atomic>

namespace nccltrace {

template <typename T>
class LockFreeQueue {
public:
  LockFreeQueue();
  ~LockFreeQueue();

  /**
   * Push item into queue.
   * Busy wait until success.
   * Item is moved into queue.
   */
  void push(T item);

  /**
   * Non-blocking pop operation.
   * Returns std::nullopt if queue is empty.
   * Otherwise returns the moved item.
   */
  std::optional<T> try_pop();

  /**
   * Blocking pop operation.
   * Busy wait until item is available.
   * Returns the moved item.
   */
  T pop();

private:
  struct Node {
    T data;
    std::atomic<Node*> next;
    
    Node(T&& d) : data(std::move(d)), next(nullptr) {}
  };

  std::atomic<Node*> head_;
  std::atomic<Node*> tail_;
};

template <typename T>
LockFreeQueue<T>::LockFreeQueue() 
    : head_(nullptr), tail_(nullptr) {
  Node* dummy = new Node(T());
  head_.store(dummy, std::memory_order_relaxed);
  tail_.store(dummy, std::memory_order_relaxed);
}

template <typename T>
LockFreeQueue<T>::~LockFreeQueue() {
  // Clean up all nodes
  Node* current = head_.load(std::memory_order_relaxed);
  while (current != nullptr) {
    Node* next = current->next.load(std::memory_order_relaxed);
    delete current;
    current = next;
  }
}

template <typename T>
void LockFreeQueue<T>::push(T item) {
  Node* new_node = new Node(std::move(item));
  
  while (true) {
    Node* last = tail_.load(std::memory_order_acquire);
    Node* next = last->next.load(std::memory_order_acquire);
    
    // Check if tail is still valid
    if (last == tail_.load(std::memory_order_acquire)) {
      if (next == nullptr) {
        // Try to link new node
        if (std::atomic_compare_exchange_weak_explicit(
            &last->next, &next, new_node,
            std::memory_order_release, std::memory_order_relaxed)) {
          // Successfully linked, now try to advance tail
          std::atomic_compare_exchange_strong_explicit(
              &tail_, &last, new_node,
              std::memory_order_release, std::memory_order_relaxed);
          return;
        }
      } else {
        // Tail is lagging, help advance it
        std::atomic_compare_exchange_strong_explicit(
            &tail_, &last, next,
            std::memory_order_release, std::memory_order_relaxed);
      }
    }
  }
}

template <typename T>
std::optional<T> LockFreeQueue<T>::try_pop() {
  while (true) {
    Node* first = head_.load(std::memory_order_acquire);
    Node* last = tail_.load(std::memory_order_acquire);
    Node* next = first->next.load(std::memory_order_acquire);
    
    if (first == head_.load(std::memory_order_acquire)) {
      // Check if queue is empty
      if (first == last) {
        if (next == nullptr) {
          // Empty queue
          return std::nullopt;
        }
        // Tail is lagging, help advance it
        std::atomic_compare_exchange_strong_explicit(
            &tail_, &last, next,
            std::memory_order_release, std::memory_order_relaxed);
      } else {
        // Queue has data, try to dequeue
        T data(std::move(next->data));
        if (std::atomic_compare_exchange_weak_explicit(
            &head_, &first, next,
            std::memory_order_release, std::memory_order_relaxed)) {
          // Successfully dequeued, delete old dummy node
          delete first;
          return data;
        }
        // CAS failed, retry
      }
    }
  }
}

template <typename T>
T LockFreeQueue<T>::pop() {
  while (true) {
    auto result = try_pop();
    if (result.has_value()) {
      return std::move(result.value());
    }
    // Queue is empty, busy wait
  }
}

}
