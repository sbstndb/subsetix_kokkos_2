// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <subsetix/mesh.hpp>
#include <subsetix/detail/utils.hpp>

namespace subsetix {

namespace detail {

/**
 * @brief Core row intersection algorithm (two-pointer merge).
 *
 * When CountOnly=true, only counts intervals without writing.
 * When CountOnly=false, writes intervals to intervals_out.
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

} // namespace detail

/**
 * @brief Compute the intersection of two meshes.
 *
 * Returns a new mesh containing only the cells that exist in BOTH input meshes.
 *
 * Algorithm:
 * 1. Row mapping - find common (Y,Z) rows via binary search
 * 2. Count - count intersecting X-intervals per row
 * 3. Scan - compute CSR offsets
 * 4. Fill - write intersected intervals
 * 5. Compact - filter rows with no intersections
 *
 * @param A First input mesh
 * @param B Second input mesh
 * @return Intersection mesh
 */
inline Mesh3DDevice intersect_meshes(const Mesh3DDevice& A,
                                     const Mesh3DDevice& B) {
  if (A.num_rows == 0 || B.num_rows == 0) {
    return Mesh3DDevice{};
  }

  // Allocate temporary buffers
  const std::size_t num_rows_a = A.num_rows;
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> flags("flags", num_rows_a);
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> tmp_idx_a("tmp_idx_a", num_rows_a);
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> tmp_idx_b("tmp_idx_b", num_rows_a);
  Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space> positions("positions", num_rows_a);

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;
  const std::size_t num_rows_b = B.num_rows;

  // Phase 1: Row mapping - find rows of A that exist in B
  Kokkos::parallel_for(
      "intersection_row_map",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        const RowKey key = rows_a(i);
        const int idx_b = detail::find_row_by_yz(rows_b, num_rows_b, key.y, key.z);
        if (idx_b >= 0) {
          flags(i) = 1;
          tmp_idx_a(i) = static_cast<int>(i);
          tmp_idx_b(i) = idx_b;
        } else {
          flags(i) = 0;
          tmp_idx_a(i) = -1;
          tmp_idx_b(i) = -1;
        }
      });

  Kokkos::fence();

  // Scan to count matching rows and compute positions
  Kokkos::View<std::size_t, Kokkos::DefaultExecutionSpace::memory_space> num_rows_out_view("num_rows_out");
  Kokkos::parallel_scan(
      "intersection_row_scan",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = static_cast<std::size_t>(flags(i));
        if (final_pass) {
          positions(i) = update;
          if (i + 1 == num_rows_a) {
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

  // Allocate output buffers for row mapping
  Kokkos::View<RowKey*, Kokkos::DefaultExecutionSpace::memory_space> out_rows("out_rows", num_rows_out);
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> out_idx_a("out_idx_a", num_rows_out);
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> out_idx_b("out_idx_b", num_rows_out);

  // Compact matching rows
  Kokkos::parallel_for(
      "intersection_row_compact",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (!flags(i)) {
          return;
        }
        const std::size_t pos = positions(i);
        out_rows(pos) = rows_a(i);
        out_idx_a(pos) = tmp_idx_a(i);
        out_idx_b(pos) = tmp_idx_b(i);
      });

  Kokkos::fence();

  // Allocate output mesh
  Mesh3DDevice out;
  if (num_rows_out > 0) {
    out.row_keys = typename Mesh3DDevice::RowKeyView("mesh_row_keys", num_rows_out);
    out.row_ptr = typename Mesh3DDevice::IndexView("mesh_row_ptr", num_rows_out + 1);
    out.intervals = typename Mesh3DDevice::IntervalView(
        "mesh_intervals", A.num_intervals + B.num_intervals);
  }

  // Copy row keys
  Kokkos::deep_copy(out.row_keys, out_rows);

  // Allocate row counts buffer
  Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space> row_counts("row_counts", num_rows_out);

  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  // Phase 2: Count intervals per row
  Kokkos::parallel_for(
      "intersection_count",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = out_idx_a(i);
        const int ib = out_idx_b(i);

        if (ib < 0) {
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

  // Phase 3: Scan to compute row_ptr offsets
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
    return Mesh3DDevice{};
  }

  // Phase 4: Fill intersected intervals
  Kokkos::parallel_for(
      "intersection_fill",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = out_idx_a(i);
        const int ib = out_idx_b(i);

        if (ib < 0) {
          return;
        }

        const auto r = detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

        if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
          return;
        }

        detail::row_intersection_impl<false>(
            intervals_a, r.begin_a, r.end_a,
            intervals_b, r.begin_b, r.end_b,
            out.intervals, out.row_ptr(i));
      });

  Kokkos::fence();

  // Phase 5: Compact - remove rows with no intervals
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> has_intervals("has_intervals", num_rows_out);
  Kokkos::parallel_for(
      "intersection_mark_rows",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        has_intervals(i) = (out.row_ptr(i) < out.row_ptr(i + 1)) ? 1 : 0;
      });

  Kokkos::fence();

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

  if (final_num_rows == num_rows_out) {
    return out;  // No compaction needed
  }

  if (final_num_rows == 0) {
    return Mesh3DDevice{};
  }

  // Allocate compacted output
  Mesh3DDevice compacted;
  compacted.row_keys = typename Mesh3DDevice::RowKeyView("compacted_row_keys", final_num_rows);
  compacted.row_ptr = typename Mesh3DDevice::IndexView("compacted_row_ptr", final_num_rows + 1);
  compacted.intervals = typename Mesh3DDevice::IntervalView("compacted_intervals", out.num_intervals);
  compacted.num_rows = final_num_rows;
  compacted.num_intervals = out.num_intervals;

  // Copy row keys and row_ptr for non-empty rows
  Kokkos::parallel_for(
      "intersection_compact_copy",
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
                compacted.row_keys(i - 1) = out.row_keys(j);
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

  return compacted;
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

  dst.row_keys = Kokkos::create_mirror_view_and_copy(ToSpace{}, src.row_keys);
  dst.row_ptr = Kokkos::create_mirror_view_and_copy(ToSpace{}, src.row_ptr);
  dst.intervals = Kokkos::create_mirror_view_and_copy(ToSpace{}, src.intervals);

  return dst;
}

} // namespace subsetix
