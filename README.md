# libkdtree++

Header-only **k-dimensional tree** for C++20: spatial ordering, nearest-neighbor search, and axis-aligned range queries. This tree is derived from **libkdtree++** (Martin F. Krafft, Sylvain Bougerel, Paul Harris) with a modernized API and internals.

## Requirements

- **C++20** or later
- CMake **3.20+** (for the bundled test target)

## Version

- CMake package version: **0.8.0**
- Macros in `kdtree.hpp`: `KDTREE_VERSION` (800), `KDTREE_LIB_VERSION` (`"0_8_0"`)

## Usage

Include the umbrella header and link nothing (header-only):

```cpp
#include "kdtree.hpp"  // adjust path to this directory

libkdtree::KDTree<2, MyPoint> tree;
tree.insert(point);
auto it = tree.findNearest(query);
```

### Main types

| Type | Role |
|------|------|
| `libkdtree::KDTree<K, Val, Acc, Dist, Cmp, Alloc>` | K-dimensional tree |
| `libkdtree::BracketAccessor<Val>` | Default accessor: `val[dim]` |
| `libkdtree::SquaredDifference<…>` | Default per-axis squared difference (Euclidean-style nearest) |
| `libkdtree::KdTreeRegion<…>` | Axis-aligned box for range queries |
| `libkdtree::TreeIterator<…>` | Bidirectional iterator over stored values |

### Typical operations

- **Construct / assign**: range (incl. single-pass iterators) and initializer-list constructors build a balanced tree; O(1) move construction / assignment and `swap`; `assignOptimized` (bulk rebuild from a vector, copying or moving)
- **Insert / clear**: `insert` (copy or move), `emplace`, `clear`
- **Lookup**: `find` (by coordinates), `findExact` (by `operator==` on value), `contains`
- **Nearest**: `findNearest`, optional max radius, `findNearestIf` with a predicate
- **Range**: `countWithinRange`, `visitWithinRange`, `findWithinRange` (L∞-style boxes in accessor space by default)
- **Erase**: `erase` (iterator, or coordinate-equivalent value), `eraseExact`; the value overloads return the number removed (0 or 1). Erasing never invalidates iterators to other elements.
- **Maintenance**: `optimize` (rebalance via median splits), `height`, `checkTreeInvariants` (returns `bool`; works in release builds)

All traversals are iterative or depth-capped, so even a degenerate tree (e.g. built by inserting sorted points one at a time) cannot overflow the call stack.

Accessors on the tree: `valueAccessor`, `valueComparator`, `valueDistance`, `getAllocator`, `maxSize`.

## CMake

**Standalone** (tests on by default; fetches GoogleTest):

```bash
cmake -S . -B build -DLIBKDTREE_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release
```

Disable tests:

```bash
cmake -S . -B build -DLIBKDTREE_BUILD_TESTS=OFF
```

**In another project** (`add_subdirectory`):

```cmake
add_subdirectory(path/to/libkdtree)
target_link_libraries(your_target PRIVATE libkdtree::libkdtree)
```

The interface target exports include directories for the libkdtree source directory.

## Tests

Tests live in `tests/test.cpp`: API unit tests, plus stress tests over uniform, clustered, lattice/duplicate-heavy, high-dimensional and degenerate data (tens of thousands of points), cross-checking every query against brute force, plus allocator, move and exception-safety tests (GoogleTest). With `LIBKDTREE_BUILD_TESTS=ON`, the executable `libkdtree_tests` is registered with CTest (label `libkdtree`).

## Optional compile-time defines

| Define | Effect |
|--------|--------|
| `KDTREE_DEFINE_OSTREAM_OPERATORS` | `operator<<` for nodes and trees (adds `<ostream>` / `<stack>` usage) |
| `KDTREE_CHECK_PERFORMANCE` | Global distance-call counter (`numDistCalcs`) when using instrumented functors |
| `KDTREE_CHECK_PERFORMANCE_COUNTERS` | Extra profiling hooks in the tree |

## Documentation

A **Doxygen** configuration is provided (`Doxyfile`). Run `doxygen` from this directory to generate API docs from the headers.

## License

Original libkdtree++ is distributed under the **Artistic License 2.0**. See the copyright blocks in the source files and any `COPYING` file shipped with the upstream distribution.

## Files

| File | Contents |
|------|----------|
| `kdtree.hpp` | `KDTree` public interface |
| `node.hpp` | `TreeNode`, `NodeBase`, nearest-neighbor helpers |
| `region.hpp` | `KdTreeRegion` |
| `iterator.hpp` | `TreeIterator` |
| `allocator.hpp` | `AllocBase`, node allocation via `std::allocator_traits` |
| `function.hpp` | `BracketAccessor`, `SquaredDifference`, helpers |
