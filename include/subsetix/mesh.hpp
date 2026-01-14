// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstddef>

namespace subsetix {

// Basic coordinate type for cell indices
using Coord = int32_t;

/**
 * @brief Half-open interval [begin, end) on the X axis.
 *
 * Invariant: begin < end
 */
struct Interval {
  Coord begin = 0;  // Inclusive
  Coord end = 0;    // Exclusive

  KOKKOS_INLINE_FUNCTION
  Coord size() const { return end - begin; }

  KOKKOS_INLINE_FUNCTION
  bool empty() const { return begin >= end; }
};

/**
 * @brief Row key for 3D sparse structure (Y and Z axes).
 *
 * Rows are identified by their (y, z) coordinates. The X-axis data
 * for each row is stored as a list of intervals.
 */
struct RowKey {
  Coord y = 0;
  Coord z = 0;

  KOKKOS_INLINE_FUNCTION
  bool operator==(const RowKey& other) const {
    return y == other.y && z == other.z;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const RowKey& other) const {
    return !(*this == other);
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<(const RowKey& other) const {
    if (y != other.y) {
      return y < other.y;
    }
    return z < other.z;
  }
};

/**
 * @brief CSR-based 3D mesh representation using interval sets.
 *
 * This is a compressed sparse row (CSR) representation where:
 * - row_keys stores the (y,z) coordinates of non-empty rows (sorted)
 * - row_ptr stores offsets into the intervals array for each row
 * - intervals stores [begin, end) X-intervals for each row
 *
 * Invariants:
 * - row_keys.extent(0) == num_rows
 * - row_ptr.extent(0) == num_rows + 1
 * - intervals.extent(0) >= num_intervals
 * - For each row, intervals are sorted and non-overlapping
 * - row_keys are sorted in lexicographic order (y first, then z)
 */
template <class MemorySpace>
class Mesh3D {
public:
  using RowKeyView = Kokkos::View<RowKey*, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, MemorySpace>;
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;

  RowKeyView row_keys;     // [num_rows] - (y,z) coordinates
  IndexView row_ptr;       // [num_rows + 1] - CSR offsets
  IntervalView intervals;  // [num_intervals] - X-intervals

  std::size_t num_rows = 0;
  std::size_t num_intervals = 0;

  KOKKOS_INLINE_FUNCTION
  Mesh3D() = default;

  KOKKOS_INLINE_FUNCTION
  Mesh3D(const Mesh3D&) = default;

  KOKKOS_INLINE_FUNCTION
  Mesh3D& operator=(const Mesh3D&) = default;
};

// Primary type aliases for common memory spaces
using Mesh3DDevice = Mesh3D<Kokkos::DefaultExecutionSpace::memory_space>;
using Mesh3DHost = Mesh3D<Kokkos::HostSpace>;

} // namespace subsetix
