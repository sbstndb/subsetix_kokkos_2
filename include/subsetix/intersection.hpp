// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <subsetix/mesh.hpp>
#include <subsetix/detail/utils.hpp>
#include <stdexcept>
#include <array>

namespace subsetix {

// ============================================================================
// Context for intersection operations (provides workspace buffers)
// ============================================================================

/**
 * @brief Workspace for mesh intersection operations.
 *
 * Reusable buffers to avoid repeated allocations during intersection.
 */
struct IntersectionWorkspace {
  // Buffer counts
  static constexpr std::size_t NUM_SIZE_T_BUFS = 2;

  // Generic buffers
  std::array<Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>, NUM_SIZE_T_BUFS> size_t_bufs_;

  // Accessors that ensure capacity
  Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>
  get_size_t_buf(std::size_t idx, std::size_t size) {
    detail::ensure_view_capacity(size_t_bufs_[idx], size,
                                 "intersection_ws_size_t_" + std::to_string(idx));
    return size_t_bufs_[idx];
  }

  // Convenience wrappers
  Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>
  get_size_t_buf_0(std::size_t size) { return get_size_t_buf(0, size); }
  Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>
  get_size_t_buf_1(std::size_t size) { return get_size_t_buf(1, size); }
};

// ============================================================================
// Row-level intersection algorithm
// ============================================================================

namespace detail {

/**
 * @brief Core row intersection algorithm (two-pointer merge).
 *
 * When CountOnly=true, only counts intervals without writing.
 * When CountOnly=false, writes intervals to intervals_out.
 *
 * Algorithm:
 * 1. Advance through both interval lists simultaneously
 * 2. For overlapping intervals, compute intersection [max(begin), min(end))
 * 3. Add intersection if non-empty
 * 4. Advance the interval that ends first
 */
template <bool CountOnly, class IntervalViewIn, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
std::size_t row_intersection_impl(const IntervalViewIn& intervals_a,
                                  std::size_t begin_a,
                                  std::size_t end_a,
                                  const IntervalViewIn& intervals_b,
                                  std::size_t begin_b,
                                  std::size_t end_b,
                                  const IntervalViewOut& intervals_out,
                                  std::size_t out_offset) {
  std::size_t ia = begin_a;
  std::size_t ib = begin_b;
  std::size_t count = 0;

  while (ia < end_a && ib < end_b) {
    const auto a = intervals_a(ia);
    const auto b = intervals_b(ib);

    // Compute intersection: [max(begin), min(end))
    const Coord start = (a.begin > b.begin) ? a.begin : b.begin;
    const Coord end = (a.end < b.end) ? a.end : b.end;

    // Add non-empty intersection
    if (start < end) {
      if constexpr (!CountOnly) {
        intervals_out(out_offset + count) = Interval{start, end};
      }
      ++count;
    }

    // Advance the interval that ends first
    if (a.end < b.end) {
      ++ia;
    } else if (b.end < a.end) {
      ++ib;
    } else {
      ++ia;
      ++ib;
    }
  }

  return count;
}

/**
 * @brief Allocate an empty mesh with given capacity.
 */
inline Mesh3DDevice allocate_mesh_device(std::size_t row_capacity,
                                         std::size_t interval_capacity) {
  Mesh3DDevice dev;

  if (row_capacity > 0) {
    dev.row_keys = typename Mesh3DDevice::RowKeyView(
        "mesh_row_keys", row_capacity);
    dev.row_ptr = typename Mesh3DDevice::IndexView(
        "mesh_row_ptr", row_capacity + 1);
  }
  if (interval_capacity > 0) {
    dev.intervals = typename Mesh3DDevice::IntervalView(
        "mesh_intervals", interval_capacity);
  }

  dev.num_rows = 0;
  dev.num_intervals = 0;

  return dev;
}

} // namespace detail

// ============================================================================
// Public intersection API
// ============================================================================

/**
 * @brief Compute the intersection of two meshes.
 *
 * Returns a new mesh containing only the cells that exist in BOTH input meshes.
 * The output mesh uses CSR format with:
 * - Only rows present in both inputs
 * - Only intervals that overlap in both inputs
 *
 * @param A First input mesh
 * @param B Second input mesh
 * @param ws Workspace for temporary buffers (created if empty)
 * @return Intersection mesh
 */
inline Mesh3DDevice intersect_meshes(const Mesh3DDevice& A,
                                     const Mesh3DDevice& B,
                                     IntersectionWorkspace& ws) {
  // Handle empty inputs
  if (A.num_rows == 0 || B.num_rows == 0) {
    return Mesh3DDevice{};
  }

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;

  // Use the smaller set for iteration (optimization)
  const bool small_is_a = (num_rows_a <= num_rows_b);
  const std::size_t n_small = small_is_a ? num_rows_a : num_rows_b;
  const std::size_t n_big = small_is_a ? num_rows_b : num_rows_a;

  auto rows_small = small_is_a ? rows_a : rows_b;
  auto rows_big = small_is_a ? rows_b : rows_a;

  // Allocate temporary buffers for row matching
  // These are local to this function and will be cleaned up properly
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> flags;
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> tmp_idx_a;
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> tmp_idx_b;
  Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space> positions;
  Kokkos::View<RowKey*, Kokkos::DefaultExecutionSpace::memory_space> out_rows;
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> out_idx_a;
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> out_idx_b;

  flags = Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space>("intersection_flags", n_small);
  tmp_idx_a = Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space>("intersection_tmp_a", n_small);
  tmp_idx_b = Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space>("intersection_tmp_b", n_small);
  positions = Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>("intersection_positions", n_small);

  // Step 1: For each row of the smaller set, binary-search in the larger set
  Kokkos::parallel_for(
      "intersection_row_map",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_small),
      KOKKOS_LAMBDA(const std::size_t i) {
        const RowKey key = rows_small(i);
        const int idx_big = detail::find_row_by_yz(rows_big, n_big, key.y, key.z);
        if (idx_big >= 0) {
          flags(i) = 1;
          tmp_idx_a(i) = small_is_a ? static_cast<int>(i) : idx_big;
          tmp_idx_b(i) = small_is_a ? idx_big : static_cast<int>(i);
        } else {
          flags(i) = 0;
          tmp_idx_a(i) = -1;
          tmp_idx_b(i) = -1;
        }
      });

  Kokkos::fence();

  // Step 2: Manual exclusive scan on flags to compute positions
  // Using parallel_scan instead of exclusive_scan to avoid subview issues
  Kokkos::View<std::size_t, Kokkos::DefaultExecutionSpace::memory_space> num_rows_out_view("num_rows_out");
  Kokkos::parallel_scan(
      "intersection_row_scan",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_small),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = static_cast<std::size_t>(flags(i));
        if (final_pass) {
          positions(i) = update;
          if (i + 1 == n_small) {
            num_rows_out_view() = update + count;
          }
        }
        update += count;
      });

  Kokkos::fence();

  std::size_t num_rows_out_host = 0;
  Kokkos::deep_copy(num_rows_out_host, num_rows_out_view);
  const std::size_t num_rows_out = num_rows_out_host;

  if (num_rows_out == 0) {
    return Mesh3DDevice{};
  }

  // Step 3: Allocate output buffers (local, will stay valid for this function)
  out_rows = Kokkos::View<RowKey*, Kokkos::DefaultExecutionSpace::memory_space>("intersection_out_rows", num_rows_out);
  out_idx_a = Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space>("intersection_out_idx_a", num_rows_out);
  out_idx_b = Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space>("intersection_out_idx_b", num_rows_out);

  // Step 4: Compact matching rows into output
  Kokkos::parallel_for(
      "intersection_row_compact",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_small),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (!flags(i)) {
          return;
        }
        const std::size_t pos = positions(i);
        out_rows(pos) = rows_small(i);
        out_idx_a(pos) = tmp_idx_a(i);
        out_idx_b(pos) = tmp_idx_b(i);
      });

  Kokkos::fence();

  // At this point, we have the row mapping in out_rows, out_idx_a, out_idx_b
  // These views will remain valid for the rest of this function

  // Allocate output mesh
  auto out = detail::allocate_mesh_device(
      num_rows_out,
      A.num_intervals + B.num_intervals);  // Worst case

  // Copy row keys
  Kokkos::deep_copy(out.row_keys, out_rows);

  // Get workspace buffer for row interval counts
  auto row_counts = ws.get_size_t_buf_0(num_rows_out);

  // Capture views for lambda
  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  // COUNT phase: count intervals per row
  Kokkos::parallel_for(
      "intersection_count",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = out_idx_a(i);
        const int ib = out_idx_b(i);

        if (ia < 0 || ib < 0) {
          row_counts(i) = 0;
          return;
        }

        const auto r = detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

        if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
          row_counts(i) = 0;
          return;
        }

        row_counts(i) = detail::row_intersection_impl<true>(
            intervals_a, r.begin_a, r.end_a,
            intervals_b, r.begin_b, r.end_b,
            Kokkos::View<Interval*, Kokkos::DefaultExecutionSpace::memory_space>(), 0);
      });

  Kokkos::fence();

  // SCAN phase: compute row_ptr offsets
  // Using parallel_scan directly to avoid subview issues
  Kokkos::View<std::size_t, Kokkos::DefaultExecutionSpace::memory_space> total_view("total_intervals");
  Kokkos::parallel_scan(
      "intersection_scan",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = row_counts(i);
        if (final_pass) {
          out.row_ptr(i) = update;
          if (i + 1 == num_rows_out) {
            out.row_ptr(num_rows_out) = update + count;
            total_view() = update + count;
          }
        }
        update += count;
      });

  Kokkos::fence();

  std::size_t num_intervals_host = 0;
  Kokkos::deep_copy(num_intervals_host, total_view);
  out.num_intervals = num_intervals_host;
  out.num_rows = num_rows_out;

  if (out.num_intervals == 0) {
    return Mesh3DDevice{};  // Empty mesh when no intervals intersect
  }

  auto intervals_out = out.intervals;
  auto row_ptr_out = out.row_ptr;

  // FILL phase: write intersected intervals
  Kokkos::parallel_for(
      "intersection_fill",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = out_idx_a(i);
        const int ib = out_idx_b(i);

        if (ia < 0 || ib < 0) {
          return;
        }

        const auto r = detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

        if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
          return;
        }

        detail::row_intersection_impl<false>(
            intervals_a, r.begin_a, r.end_a,
            intervals_b, r.begin_b, r.end_b,
            intervals_out, row_ptr_out(i));
      });

  Kokkos::fence();

  // COMPACT phase: Remove rows with no intersecting intervals
  // This is necessary because rows were selected based on (y,z) matching,
  // but some rows may have no X-interval overlaps
  if (out.num_intervals == 0) {
    return Mesh3DDevice{};
  }

  // Mark rows that have intervals
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> has_intervals("has_intervals", num_rows_out);
  Kokkos::parallel_for(
      "intersection_mark_rows",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        has_intervals(i) = (out.row_ptr(i) < out.row_ptr(i + 1)) ? 1 : 0;
      });

  Kokkos::fence();

  // Scan to compute new positions
  Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space> new_positions("new_positions", num_rows_out);
  Kokkos::View<std::size_t, Kokkos::DefaultExecutionSpace::memory_space> final_num_rows_view("final_num_rows");
  Kokkos::parallel_scan(
      "intersection_compact_scan",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = static_cast<std::size_t>(has_intervals(i));
        if (final_pass) {
          new_positions(i) = update;
          if (i + 1 == num_rows_out) {
            final_num_rows_view() = update + count;
          }
        }
        update += count;
      });

  Kokkos::fence();

  std::size_t final_num_rows = 0;
  Kokkos::deep_copy(final_num_rows, final_num_rows_view);

  // If all rows have intervals, return as-is
  if (final_num_rows == num_rows_out) {
    return out;
  }

  // Allocate compacted output
  auto compacted = detail::allocate_mesh_device(final_num_rows, out.num_intervals);

  // Copy row keys for non-empty rows
  Kokkos::parallel_for(
      "intersection_compact_rows",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (has_intervals(i)) {
          const std::size_t new_idx = new_positions(i);
          compacted.row_keys(new_idx) = out.row_keys(i);
        }
      });

  // Compute new row_ptr
  Kokkos::parallel_for(
      "intersection_compact_row_ptr",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, final_num_rows + 1),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (i == 0) {
          compacted.row_ptr(0) = 0;
        } else {
          // Find the i-th non-empty row
          std::size_t count = 0;
          for (std::size_t j = 0; j < num_rows_out; ++j) {
            if (has_intervals(j)) {
              if (count == i) {
                compacted.row_ptr(i) = out.row_ptr(j);
                break;
              }
              ++count;
            }
          }
        }
      });

  // Copy intervals
  Kokkos::parallel_for(
      "intersection_compact_intervals",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, out.num_intervals),
      KOKKOS_LAMBDA(const std::size_t i) {
        compacted.intervals(i) = out.intervals(i);
      });

  Kokkos::fence();

  compacted.num_rows = final_num_rows;
  compacted.num_intervals = out.num_intervals;

  return compacted;
}

/**
 * @brief Convenience overload that creates a local workspace.
 */
inline Mesh3DDevice intersect_meshes(const Mesh3DDevice& A,
                                     const Mesh3DDevice& B) {
  IntersectionWorkspace ws;
  return intersect_meshes(A, B, ws);
}

// ============================================================================
// Conversion between memory spaces
// ============================================================================

/**
 * @brief Convert a mesh between memory spaces (e.g., Device -> Host).
 */
template <class ToSpace, class FromSpace>
inline Mesh3D<ToSpace> mesh_to(const Mesh3D<FromSpace>& src) {
  Mesh3D<ToSpace> dst;

  if (src.num_rows == 0) {
    return dst;
  }

  dst.num_rows = src.num_rows;
  dst.num_intervals = src.num_intervals;

  // Allocate and copy views
  dst.row_keys = Kokkos::create_mirror_view_and_copy(
      ToSpace{}, src.row_keys);
  dst.row_ptr = Kokkos::create_mirror_view_and_copy(
      ToSpace{}, src.row_ptr);
  dst.intervals = Kokkos::create_mirror_view_and_copy(
      ToSpace{}, src.intervals);

  return dst;
}

} // namespace subsetix
