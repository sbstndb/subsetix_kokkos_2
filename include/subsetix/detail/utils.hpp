// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <string>
#include <algorithm>

namespace subsetix {
namespace detail {

// ============================================================================
// Memory utilities
// ============================================================================

/**
 * @brief Ensure a Kokkos View has at least the required capacity.
 *
 * If the current capacity is less than required_size, the view is
 * reallocated. Content is NOT preserved (for scratch buffers).
 */
template <class ViewType>
inline void ensure_view_capacity(ViewType& view,
                                 std::size_t required_size,
                                 const std::string& label) {
  if (view.extent(0) < required_size) {
    view = ViewType(label, required_size);
  }
}

// ============================================================================
// Scan utilities
// ============================================================================

/**
 * @brief Perform an exclusive scan for CSR row_ptr, returning the total.
 *
 * For each i in [0, n), writes row_ptr(i) = sum of counts(0..i-1).
 * Also writes row_ptr(n) = total.
 * Returns the total sum of all counts.
 */
template <typename T, class CountView, class IndexView>
T exclusive_scan_csr_row_ptr(
    const std::string& label,
    std::size_t n,
    const CountView& counts,
    IndexView& row_ptr) {
  if (n == 0) {
    Kokkos::deep_copy(Kokkos::subview(row_ptr, 0), T(0));
    return T(0);
  }

  // Create subviews with exact size n
  auto counts_sub = Kokkos::subview(counts, std::make_pair(std::size_t(0), n));
  auto row_ptr_sub = Kokkos::subview(row_ptr, std::make_pair(std::size_t(0), n));

  // Exclusive scan into row_ptr[0..n)
  Kokkos::Experimental::exclusive_scan(
      Kokkos::DefaultExecutionSpace(), counts_sub, row_ptr_sub, T(0));

  // Compute total and set row_ptr[n]
  T total = T(0);
  Kokkos::parallel_reduce(
      label + "_total",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i, T& sum) {
        sum += static_cast<T>(counts(i));
      },
      total);

  Kokkos::deep_copy(Kokkos::subview(row_ptr, n), total);
  return total;
}

// ============================================================================
// Binary search utilities
// ============================================================================

/**
 * @brief Find a row index by (y,z) coordinates using binary search.
 *
 * @param rows The view of row keys (sorted by y, then z)
 * @param num_rows Number of rows in the view
 * @param y The y-coordinate to search for
 * @param z The z-coordinate to search for
 * @return The index of the row if found, -1 otherwise
 */
template <class RowKeyView>
KOKKOS_INLINE_FUNCTION
int find_row_by_yz(const RowKeyView& rows, std::size_t num_rows, Coord y, Coord z) {
  std::size_t lo = 0;
  std::size_t hi = num_rows;

  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    const RowKey key = rows(mid);

    if (key.y < y || (key.y == y && key.z < z)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  if (lo < num_rows) {
    const RowKey key = rows(lo);
    if (key.y == y && key.z == z) {
      return static_cast<int>(lo);
    }
  }

  return -1;
}

/**
 * @brief Extract interval ranges for two rows given their indices.
 *
 * Helper for binary CSR operations to get begin/end indices from CSR row_ptr.
 */
struct RowRanges {
  std::size_t begin_a = 0;
  std::size_t end_a = 0;
  std::size_t begin_b = 0;
  std::size_t end_b = 0;

  KOKKOS_INLINE_FUNCTION
  bool both_empty() const {
    return begin_a == end_a && begin_b == end_b;
  }

  KOKKOS_INLINE_FUNCTION
  bool a_empty() const {
    return begin_a == end_a;
  }

  KOKKOS_INLINE_FUNCTION
  bool b_empty() const {
    return begin_b == end_b;
  }
};

template <class RowPtrViewA, class RowPtrViewB>
KOKKOS_FORCEINLINE_FUNCTION
RowRanges extract_row_ranges(int ia, int ib,
                              const RowPtrViewA& row_ptr_a,
                              const RowPtrViewB& row_ptr_b) {
  RowRanges r;
  if (ia >= 0) {
    const std::size_t row_a = static_cast<std::size_t>(ia);
    r.begin_a = row_ptr_a(row_a);
    r.end_a = row_ptr_a(row_a + 1);
  }
  if (ib >= 0) {
    const std::size_t row_b = static_cast<std::size_t>(ib);
    r.begin_b = row_ptr_b(row_b);
    r.end_b = row_ptr_b(row_b + 1);
  }
  return r;
}

} // namespace detail
} // namespace subsetix
