/* COPYRIGHT --
 *
 * This file is part of libkdtree++, a C++ template KD-Tree sorting container.
 * libkdtree++ is (c) 2004-2007 Martin F. Krafft <libkdtree@pobox.madduck.net>
 * and Sylvain Bougerel <sylvain.bougerel.devel@gmail.com> distributed under the
 * terms of the Artistic License 2.0. See the ./COPYING file in the source tree
 * root for more information.
 * Parts of this file are (c) 2004-2007 Paul Harris <paulharris@computer.org>.
 *
 * THIS PACKAGE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES
 * OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */
/** \file
 * \brief Public interface for \ref KDTree, a k-dimensional search tree.
 *
 * \author Martin F. Krafft <libkdtree@pobox.madduck.net>
 * \author Paul Harris (tree invariants and design notes below)
 *
 * \details
 * This structure is similar to a binary search tree, but not identical.
 * Important differences:
 *
 *  * Each level is sorted by a different criteria (this is fundamental to the
 * design).
 *
 *  * It is possible to have children IDENTICAL to its parent in BOTH branches
 *    This is different to a binary tree, where identical children are always to
 * the right So, KDTree has the relationships:
 *    * The left branch is <= its parent (in binary tree, this relationship is a
 * plain < )
 *    * The right branch is >= its parent (same as binary tree)
 *
 *    This is done for mostly for performance.
 *    Its a LOT easier to maintain a consistent tree if we use the <=
 * relationship. Note that this relationship only makes a difference when
 * searching for an exact item with find() or findExact, other search, erase
 * and insert functions don't notice the difference.
 *
 *    In the case of binary trees, you can safely assume that the next identical
 * item will be the child leaf, but in the case of KDTree, the next identical
 * item might be a long way down a subtree, because of the various different
 * sort criteria.
 *
 *    So erase()ing a node from a KDTree could require serious and complicated
 *    tree rebalancing to maintain consistency... IF we required
 * binary-tree-like relationships.
 *
 *    This has no effect on insert()s, a < test is good enough to keep
 * consistency.
 *
 *    It has an effect on find() searches:
 *      * Instead of using Compare(child,node) for a < relationship and
 * following 1 branch, we must use !Compare(node,child) for a <= relationship,
 * and test BOTH branches, as we could potentially go down both branches.
 *
 *    It has no real effect on bounds-based searches (like findNearest,
 * findWithinRange) as it compares vs a boundary and would follow both
 * branches if required.
 *
 *    This has no real effect on erase()s, a < test is good enough to keep
 * consistency.
 */

#pragma once

//
//! \ingroup libkdtree
//! Integer library version: patch = `KDTREE_VERSION % 100`, minor = `(KDTREE_VERSION / 100) % 1000`, major = `KDTREE_VERSION / 100000`.
#define KDTREE_VERSION 800
//! String library version, same logical version as \ref KDTREE_VERSION (form `major_minor[_patch]`).
#define KDTREE_LIB_VERSION "0_8_0"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef KDTREE_DEFINE_OSTREAM_OPERATORS
#include <ostream>
#endif

#include "function.hpp"
#include "allocator.hpp"
#include "iterator.hpp"
#include "node.hpp"
#include "region.hpp"

/** \defgroup libkdtree libkdtree
 *  \brief Header-only k-d tree: spatial ordering, nearest neighbor, and range queries.
 *
 *  Include `kdtree.hpp` for \ref KDTree. Related types: \ref BracketAccessor,
 *  \ref SquaredDifference, \ref KdTreeRegion, \ref TreeIterator.
 */

namespace libkdtree
{
/** \addtogroup libkdtree
 *  @{
 */

#ifdef KDTREE_CHECK_PERFORMANCE
//! Incremented by instrumented distance functors when `KDTREE_CHECK_PERFORMANCE` is defined.
inline unsigned long long numDistCalcs = 0;
#endif

/**
 * \brief K-dimensional tree storing elements of type \c Val.
 *
 * Each level splits along one axis (dimension), cycling \f$0 \ldots K-1\f$.
 * Insertion uses a single-branch comparison; \ref find must sometimes explore
 * both subtrees because the implementation uses a \f$\le\f$ (not strict \f$<\f$)
 * discipline on the active dimension—see the notes at the top of this header.
 *
 * All traversals (insert, find, range queries, nearest neighbour, erase, clear)
 * are iterative and walk parent links, so they neither recurse nor allocate:
 * a degenerate (e.g. sorted-insert) tree cannot overflow the call stack.
 *
 * \tparam K Number of dimensions.
 * \tparam Val Element type stored in the tree (e.g. a point type).
 * \tparam Acc Coordinate accessor: `result_type operator()(Val const&, std::size_t dim) const`.
 *         Default \ref BracketAccessor requires `Val::value_type` and `operator[]`.
 * \tparam Dist Per-axis distance functor; its `distance_type` is accumulated across
 *         dimensions for nearest-neighbor search. Default \ref SquaredDifference yields
 *         Euclidean distance in \ref findNearest results (square root of the sum of per-axis squares).
 * \tparam Cmp Strict weak ordering on scalar coordinates; default `std::less`.
 * \tparam Alloc Allocator for `TreeNode<Val>` (an allocator for \c Val is rebound automatically).
 */
template <std::size_t K, typename Val,
          typename Acc = BracketAccessor<Val>,
          typename Dist = SquaredDifference<typename Acc::result_type,
                                            typename Acc::result_type>,
          typename Cmp = std::less<typename Acc::result_type>,
          typename Alloc = std::allocator<TreeNode<Val>>>
class KDTree : protected AllocBase<Val, Alloc>
{
    static_assert(K > 0, "KDTree needs at least one dimension");
    static_assert(std::invocable<Acc const&, Val const&, std::size_t>,
                  "Acc must be callable as acc(value, dim)");

protected:
    using Base = AllocBase<Val, Alloc>;
    using AllocTraits = typename Base::AllocTraits;

    using BasePtr = NodeBase*;
    using BaseConstPtr = NodeBase const*;
    using LinkType = TreeNode<Val>*;
    using LinkConstType = TreeNode<Val> const*;

    using NodeDimCompare = NodeDimensionComparator<Val, Acc, Cmp>;

public:
    using allocator_type = typename Base::allocator_type;
    //! Axis-aligned box for \ref countWithinRange, \ref findWithinRange, and \ref visitWithinRange.
    using Region = KdTreeRegion<K, Val, typename Acc::result_type, Acc, Cmp>;
    using value_type = Val;
    using pointer = value_type*;
    using const_pointer = value_type const*;
    using reference = value_type&;
    using const_reference = value_type const&;
    using subvalue_type = typename Acc::result_type;
    using distance_type = typename Dist::distance_type;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    //! In-order iterator over values (bidirectional). There is no mutable iterator.
    using const_iterator = TreeIterator<Val, const_reference, const_pointer>;
    using iterator = const_iterator;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using reverse_iterator = std::reverse_iterator<iterator>;

    // --- construction / assignment ------------------------------------------

    //! Empty tree with default-constructed functors and allocator.
    KDTree() : Base(allocator_type()) {}

    //! Empty tree with the given accessor, distance functor, comparator and allocator.
    explicit KDTree(Acc const& refAcc, Dist const& refDist = Dist(),
                    Cmp const& refCmp = Cmp(),
                    const allocator_type& refAlloc = allocator_type())
        : Base(refAlloc), mAcc(refAcc), mCmp(refCmp), mDist(refDist)
    {
    }

    //! Empty tree using \p refAlloc.
    explicit KDTree(const allocator_type& refAlloc) : Base(refAlloc) {}

    //! Copy constructor: builds a balanced tree holding copies of \p refRhs's elements.
    KDTree(const KDTree& refRhs)
        : Base(AllocTraits::select_on_container_copy_construction(refRhs.mNodeAlloc)),
          mAcc(refRhs.mAcc),
          mCmp(refRhs.mCmp),
          mDist(refRhs.mDist)
    {
        std::vector<value_type> vecTemp(refRhs.begin(), refRhs.end());
        buildBalanced<true>(vecTemp);
    }

    //! Move constructor: steals \p refRhs's nodes in O(1); \p refRhs is left empty.
    KDTree(KDTree&& refRhs) noexcept
        : Base(std::move(refRhs.mNodeAlloc)),
          mAcc(std::move(refRhs.mAcc)),
          mCmp(std::move(refRhs.mCmp)),
          mDist(std::move(refRhs.mDist))
    {
        adopt(refRhs);
    }

    /**
     * \brief Range constructor: bulk-builds a balanced tree from \p [itFirst, itLast).
     * \tparam Iter Input iterator (single-pass iterators are supported).
     */
    template <std::input_iterator Iter, std::sentinel_for<Iter> Sent>
    KDTree(Iter itFirst, Sent itLast,
           Acc const& refAcc = Acc(), Dist const& refDist = Dist(),
           Cmp const& refCmp = Cmp(),
           const allocator_type& refAlloc = allocator_type())
        : Base(refAlloc), mAcc(refAcc), mCmp(refCmp), mDist(refDist)
    {
        std::vector<value_type> vecTemp;
        if constexpr (std::sized_sentinel_for<Sent, Iter>)
            vecTemp.reserve(static_cast<size_type>(itLast - itFirst));
        for (; itFirst != itLast; ++itFirst) vecTemp.emplace_back(*itFirst);
        buildBalanced<true>(vecTemp);
    }

    //! Bulk-builds a balanced tree from an initializer list.
    KDTree(std::initializer_list<value_type> lstInit,
           Acc const& refAcc = Acc(), Dist const& refDist = Dist(),
           Cmp const& refCmp = Cmp(),
           const allocator_type& refAlloc = allocator_type())
        : KDTree(lstInit.begin(), lstInit.end(), refAcc, refDist, refCmp, refAlloc)
    {
    }

    KDTree& operator=(const KDTree& refRhs)
    {
        if (this != &refRhs) {
            std::vector<value_type> vecTemp(refRhs.begin(), refRhs.end());
            clear();
            if constexpr (AllocTraits::propagate_on_container_copy_assignment::value)
                this->mNodeAlloc = refRhs.mNodeAlloc;
            mAcc = refRhs.mAcc;
            mDist = refRhs.mDist;
            mCmp = refRhs.mCmp;
            buildBalanced<true>(vecTemp);
        }
        return *this;
    }

    KDTree& operator=(KDTree&& refRhs) noexcept(
        AllocTraits::propagate_on_container_move_assignment::value ||
        AllocTraits::is_always_equal::value)
    {
        if (this == &refRhs) return *this;
        clear();
        mAcc = std::move(refRhs.mAcc);
        mDist = std::move(refRhs.mDist);
        mCmp = std::move(refRhs.mCmp);
        if constexpr (AllocTraits::propagate_on_container_move_assignment::value) {
            this->mNodeAlloc = std::move(refRhs.mNodeAlloc);
            adopt(refRhs);
        } else if constexpr (AllocTraits::is_always_equal::value) {
            adopt(refRhs);
        } else if (this->mNodeAlloc == refRhs.mNodeAlloc) {
            adopt(refRhs);
        } else {
            // Unequal, non-propagating allocators: move element-wise.
            std::vector<value_type> vecTemp = refRhs.extractValues();
            refRhs.clear();
            buildBalanced<true>(vecTemp);
        }
        return *this;
    }

    KDTree& operator=(std::initializer_list<value_type> lstInit)
    {
        std::vector<value_type> vecTemp(lstInit);
        clear();
        buildBalanced<true>(vecTemp);
        return *this;
    }

    //! Destroys all nodes (\ref clear).
    ~KDTree() { clear(); }

    //! Swaps contents, functors and (if the allocator propagates on swap) allocators.
    void swap(KDTree& refOther) noexcept
    {
        using std::swap;
        if constexpr (AllocTraits::propagate_on_container_swap::value)
            swap(this->mNodeAlloc, refOther.mNodeAlloc);
        swap(mRoot, refOther.mRoot);
        swap(mHdr.mLeft, refOther.mHdr.mLeft);
        swap(mHdr.mRight, refOther.mHdr.mRight);
        swap(mNodeCount, refOther.mNodeCount);
        swap(mAcc, refOther.mAcc);
        swap(mCmp, refOther.mCmp);
        swap(mDist, refOther.mDist);
        fixHeader();
        refOther.fixHeader();
    }

    friend void swap(KDTree& refA, KDTree& refB) noexcept { refA.swap(refB); }

    /**
     * \brief Clears the tree and rebuilds it balanced from copies of \p vecWritable's elements.
     *
     * The vector is reordered by the median-split build (no extra copy of the vector).
     * \post `vecWritable` contents are a permutation of the previous elements (same multiset).
     */
    void assignOptimized(std::vector<value_type>& vecWritable)
    {
        clear();
        buildBalanced<false>(vecWritable);
    }

    //! As above, but moves the elements out of \p vecValues (left in a valid but unspecified state).
    void assignOptimized(std::vector<value_type>&& vecValues)
    {
        clear();
        buildBalanced<true>(vecValues);
    }

    // --- capacity / observers -----------------------------------------------

    //! Returns a copy of the node allocator.
    [[nodiscard]] allocator_type getAllocator() const noexcept { return Base::getAllocator(); }

    //! Number of elements in the tree.
    [[nodiscard]] size_type size() const noexcept { return mNodeCount; }

    //! Upper bound on container size.
    [[nodiscard]] size_type maxSize() const noexcept
    {
        return std::numeric_limits<size_type>::max();
    }

    //! \c true if `size() == 0`.
    [[nodiscard]] bool empty() const noexcept { return mNodeCount == 0; }

    //! Removes all elements. O(n), iterative, no allocation.
    void clear() noexcept
    {
        BasePtr pWalk = mRoot;
        while (pWalk) {
            if (pWalk->mLeft) {
                pWalk = pWalk->mLeft;
            } else if (pWalk->mRight) {
                pWalk = pWalk->mRight;
            } else {
                BasePtr const pParent = pWalk->mParent;
                if (pParent != &mHdr)
                    (pParent->mLeft == pWalk ? pParent->mLeft : pParent->mRight) = nullptr;
                this->deleteNode(static_cast<LinkType>(pWalk));
                if (pParent == &mHdr) break;
                pWalk = pParent;
            }
        }
        mRoot = nullptr;
        mNodeCount = 0;
        fixHeader();
    }

    /*! \brief Comparator for the values in the KDTree.

      The comparator shall not be modified, it could invalidate the tree.
      \return a copy of the comparator used by the KDTree.
     */
    [[nodiscard]] Cmp valueComparator() const { return mCmp; }

    /*! \brief Accessor to the value's elements.

      This accessor shall not be modified, it could invalidate the tree.
      \return a copy of the accessor used by the KDTree.
     */
    [[nodiscard]] Acc valueAccessor() const { return mAcc; }

    /*! \brief Distance functor used for nearest-neighbor and related queries.

      May be modified after construction; changes affect \ref findNearest and
      internal distance accumulation only (not tree layout).
      \return Reference to the distance functor.
     */
    [[nodiscard]] const Dist& valueDistance() const noexcept { return mDist; }

    //! \copydoc valueDistance() const
    [[nodiscard]] Dist& valueDistance() noexcept { return mDist; }

    // --- iteration ----------------------------------------------------------

    //! First element in tree (in-)order; equals \ref end when empty.
    [[nodiscard]] const_iterator begin() const noexcept { return const_iterator(leftmost()); }
    //! Past-the-end iterator; must not be dereferenced.
    [[nodiscard]] const_iterator end() const noexcept
    {
        return const_iterator(static_cast<LinkConstType>(&mHdr));
    }
    [[nodiscard]] const_iterator cbegin() const noexcept { return begin(); }
    [[nodiscard]] const_iterator cend() const noexcept { return end(); }
    [[nodiscard]] const_reverse_iterator rbegin() const noexcept
    {
        return const_reverse_iterator(end());
    }
    [[nodiscard]] const_reverse_iterator rend() const noexcept
    {
        return const_reverse_iterator(begin());
    }

    // --- insertion ----------------------------------------------------------

    /**
     * \brief Inserts \p refVal and returns an iterator to the new node.
     * \complexity \f$O(\log n)\f$ typical depth; worst-case \f$O(n)\f$ for a degenerate tree.
     */
    iterator insert(const_reference refVal) { return linkNode(this->createNode(refVal)); }

    //! Move-inserts \p refVal.
    iterator insert(value_type&& refVal) { return linkNode(this->createNode(std::move(refVal))); }

    //! Constructs a value in place from \p args and inserts it.
    template <typename... Args>
    iterator emplace(Args&&... args)
    {
        return linkNode(this->createNode(std::forward<Args>(args)...));
    }

    //! \copybrief insert(const_reference)
    //! \param ignored Position hint (ignored; same as \ref insert(const_reference)).
    iterator insert(iterator /* ignored */, const_reference refVal) { return insert(refVal); }

    //! Inserts each element in \p [itFirst, itLast) one at a time (use \ref optimize afterwards to rebalance).
    template <std::input_iterator Iter, std::sentinel_for<Iter> Sent>
    void insert(Iter itFirst, Sent itLast)
    {
        for (; itFirst != itLast; ++itFirst) emplace(*itFirst);
    }

    //! Inserts every element of \p lstInit.
    void insert(std::initializer_list<value_type> lstInit) { insert(lstInit.begin(), lstInit.end()); }

    //! Inserts \p cDup copies of \p refValDup; \p itPos is ignored.
    void insert(iterator /* itPos */, size_type cDup, const value_type& refValDup)
    {
        for (; cDup > 0; --cDup) insert(refValDup);
    }

    //! Inserts range \p [itFirst, itLast); \p itPos is ignored.
    template <std::input_iterator Iter>
    void insert(iterator /* itPos */, Iter itFirst, Iter itLast)
    {
        insert(itFirst, itLast);
    }

    // --- erase --------------------------------------------------------------

    /**
     * \brief Erases one element \ref find "equivalent" to \p refVal (same location in all dimensions).
     * \return Number of elements removed (0 or 1).
     * \see find, eraseExact
     */
    size_type erase(const_reference refVal)
    {
        const_iterator const it = find(refVal);
        if (it == end()) return 0;
        erase(it);
        return 1;
    }

    //! Erases one element `==` to \p refVal. \return Number of elements removed (0 or 1).
    size_type eraseExact(const_reference refVal)
    {
        const_iterator const it = findExact(refVal);
        if (it == end()) return 0;
        erase(it);
        return 1;
    }

    /**
     * \brief Erases the node referenced by \p itErase. Other iterators stay valid.
     * \pre `itErase != end()`.
     */
    void erase(const_iterator const& itErase) noexcept
    {
        assert(itErase != end());
        LinkType const pDead = asMutableNode(itErase.node());
        unlinkNode(pDead, depthOf(pDead));
        this->deleteNode(pDead);
        --mNodeCount;
    }

    // --- lookup -------------------------------------------------------------

    /**
     * \brief Finds any element with the same coordinates as \p refVal (all dimensions, via \c Cmp on accessor values).
     * \tparam SearchVal Type compatible with the accessor (need not be \c value_type).
     * \return Iterator to a matching element, or \ref end() if none.
     * \see findExact
     */
    template <class SearchVal>
    [[nodiscard]] const_iterator find(SearchVal const& refVal) const
    {
        return toIterator(walkSubtree(
            mRoot, 0, [&](LinkConstType n, std::size_t d) { return !lessOnAxis(n->mValue, refVal, d); },
            [&](LinkConstType n, std::size_t d) { return !lessOnAxis(refVal, n->mValue, d); },
            [&](LinkConstType n, size_type) { return matchesAllDims(n->mValue, refVal); }));
    }

    /**
     * \brief Finds an element equal to \p refVal using `value_type::operator==`.
     *
     * Use when several stored values may share the same coordinates but differ otherwise
     * (e.g. distinct IDs). \ref find only matches by coordinate equivalence.
     */
    template <class SearchVal>
    [[nodiscard]] const_iterator findExact(SearchVal const& refVal) const
    {
        return toIterator(walkSubtree(
            mRoot, 0, [&](LinkConstType n, std::size_t d) { return !lessOnAxis(n->mValue, refVal, d); },
            [&](LinkConstType n, std::size_t d) { return !lessOnAxis(refVal, n->mValue, d); },
            [&](LinkConstType n, size_type) { return n->mValue == refVal; }));
    }

    //! \c true if an element with the same coordinates as \p refVal exists.
    template <class SearchVal>
    [[nodiscard]] bool contains(SearchVal const& refVal) const
    {
        return find(refVal) != end();
    }

    // --- range queries ------------------------------------------------------

    //! Counts nodes inside the axis-aligned box centered at \p refVal with half-extent \p subR per axis (\ref findWithinRange "same metric").
    template <typename SearchVal>
    [[nodiscard]] size_type countWithinRange(SearchVal const& refVal, subvalue_type const subR) const
    {
        return countWithinRange(Region(refVal, subR, mAcc, mCmp));
    }

    //! Counts nodes inside an arbitrary axis-aligned \ref KdTreeRegion "region".
    [[nodiscard]] size_type countWithinRange(Region const& refRegion) const
    {
        size_type count = 0;
        forEachInRegion(refRegion, [&count](const_reference) { ++count; });
        return count;
    }

    /**
     * \brief Invokes \p visitor on each value inside the box centered at \p V with half-extent \p R per axis.
     * \return The visitor after traversal (for stateful functors).
     */
    template <typename SearchVal, class Visitor>
    Visitor visitWithinRange(SearchVal const& V, subvalue_type const R, Visitor visitor) const
    {
        return visitWithinRange(Region(V, R, mAcc, mCmp), std::move(visitor));
    }

    template <class Visitor>
    Visitor visitWithinRange(Region const& REGION, Visitor visitor) const
    {
        forEachInRegion(REGION, [&visitor](const_reference v) { visitor(v); });
        return visitor;
    }

    /**
     * \brief Outputs all values in the axis-aligned box: center \p val, per-axis half-width \p range.
     *
     * \warning This is an \f$L_\infty\f$-style box query (max-norm ball in accessor space), \e not Euclidean
     *          distance. It includes every point \f$(x,y,\ldots)\f$ with \f$|x-c_x|\le r\f$, etc.
     * \return Output iterator past the last written element.
     */
    template <typename SearchVal, typename OutputIterator>
    OutputIterator findWithinRange(SearchVal const& val, subvalue_type const range,
                                   OutputIterator out) const
    {
        return findWithinRange(Region(val, range, mAcc, mCmp), std::move(out));
    }

    template <typename OutputIterator>
    OutputIterator findWithinRange(Region const& region, OutputIterator out) const
    {
        forEachInRegion(region, [&out](const_reference v) { *out++ = v; });
        return out;
    }

    // --- nearest neighbour --------------------------------------------------

    /**
     * \brief Nearest neighbor to \p refSearchVal in the metric implied by \c Dist (default: Euclidean).
     * \return `{iterator, distance}` or `{end(), 0}` if the tree is empty.
     */
    template <class SearchVal>
    [[nodiscard]] std::pair<const_iterator, distance_type> findNearest(
        SearchVal const& refSearchVal) const
    {
        return nearestImpl(refSearchVal, std::nullopt, AlwaysTrue<value_type>());
    }

    /**
     * \brief Nearest neighbor within maximum distance \p distMaxBound.
     * \return Best pair inside the ball, or `{end(), distMaxBound}` if no element lies within it.
     */
    template <class SearchVal>
    [[nodiscard]] std::pair<const_iterator, distance_type> findNearest(
        SearchVal const& refSearchVal, distance_type distMaxBound) const
    {
        return nearestImpl(refSearchVal, distMaxBound, AlwaysTrue<value_type>());
    }

    //! Nearest neighbor within \p distMaxBound among elements for which \p fnPred returns \c true.
    template <class SearchVal, class Predicate>
    [[nodiscard]] std::pair<const_iterator, distance_type> findNearestIf(
        SearchVal const& refSearchVal, distance_type distMaxBound, Predicate fnPred) const
    {
        return nearestImpl(refSearchVal, distMaxBound, fnPred);
    }

    // --- maintenance --------------------------------------------------------

    /**
     * \brief Rebuilds the tree to a balanced shape (median split on every level).
     * \complexity \f$O(n \log n)\f$. Elements are moved, not copied, when that cannot throw.
     */
    void optimize()
    {
        std::vector<value_type> vecValues = extractValues();
        clear();
        buildBalanced<true>(vecValues);
    }

    /**
     * \brief Verifies every structural invariant (works in release builds).
     *
     * Checks the split ordering of every node against all of its ancestors, parent links,
     * \ref size, and the cached leftmost / rightmost nodes. O(n) time, O(depth) memory.
     * \return \c true if the tree is consistent.
     */
    bool checkTreeInvariants() const
    {
        if (!mRoot) return mNodeCount == 0 && leftmost() == &mHdr && rightmost() == &mHdr;
        if (mRoot->mParent != &mHdr) return false;
        if (leftmost() != NodeBase::Minimum(mRoot) || rightmost() != NodeBase::Maximum(mRoot))
            return false;

        // Each node must lie inside the box carved out by its ancestors' splits.
        struct Frame {
            LinkConstType node;
            size_type depth;
            std::array<LinkConstType, K> lo;  // nullptr = unbounded
            std::array<LinkConstType, K> hi;
        };
        std::vector<Frame> stack;
        stack.push_back(Frame{mRoot, 0, {}, {}});
        size_type visited = 0;
        while (!stack.empty()) {
            Frame f = stack.back();
            stack.pop_back();
            ++visited;
            for (std::size_t a = 0; a < K; ++a) {
                if (f.lo[a] && lessOnAxis(f.node->mValue, f.lo[a]->mValue, a)) return false;
                if (f.hi[a] && lessOnAxis(f.hi[a]->mValue, f.node->mValue, a)) return false;
            }
            std::size_t const axis = f.depth % K;
            if (LinkConstType l = leftOf(f.node)) {
                if (l->mParent != f.node) return false;
                Frame c{l, f.depth + 1, f.lo, f.hi};
                c.hi[axis] = f.node;
                stack.push_back(c);
            }
            if (LinkConstType r = rightOf(f.node)) {
                if (r->mParent != f.node) return false;
                Frame c{r, f.depth + 1, f.lo, f.hi};
                c.lo[axis] = f.node;
                stack.push_back(c);
            }
        }
        return visited == mNodeCount;
    }

    //! Height of the tree (0 when empty, 1 for a single node). O(n).
    [[nodiscard]] size_type height() const
    {
        size_type best = 0;
        walkSubtree(
            mRoot, 0, [](LinkConstType, std::size_t) { return true; },
            [](LinkConstType, std::size_t) { return true; },
            [&best](LinkConstType, size_type d) {
                best = std::max(best, d + 1);
                return false;
            });
        return best;
    }

protected:
    // --- small helpers ------------------------------------------------------

    static constexpr std::size_t nextAxis(std::size_t a) noexcept { return a + 1 == K ? 0 : a + 1; }
    static constexpr std::size_t prevAxis(std::size_t a) noexcept { return a == 0 ? K - 1 : a - 1; }

    template <typename A, typename B>
    [[nodiscard]] bool lessOnAxis(A const& a, B const& b, std::size_t axis) const
    {
        return mCmp(mAcc(a, axis), mAcc(b, axis));
    }

    template <typename A, typename B>
    [[nodiscard]] bool matchesAllDims(A const& a, B const& b) const
    {
        for (std::size_t d = 0; d < K; ++d)
            if (lessOnAxis(a, b, d) || lessOnAxis(b, a, d)) return false;
        return true;
    }

    [[nodiscard]] const_iterator toIterator(LinkConstType pNode) const noexcept
    {
        return pNode ? const_iterator(pNode) : end();
    }

    void fixHeader() noexcept
    {
        mHdr.mParent = nullptr;
        if (mRoot) {
            mRoot->mParent = &mHdr;
        } else {
            mHdr.mLeft = &mHdr;
            mHdr.mRight = &mHdr;
        }
    }

    void refreshExtremes() noexcept
    {
        if (mRoot) {
            mHdr.mLeft = NodeBase::Minimum(mRoot);
            mHdr.mRight = NodeBase::Maximum(mRoot);
        } else {
            fixHeader();
        }
    }

    //! Takes ownership of \p refRhs's nodes. \pre `*this` is empty.
    void adopt(KDTree& refRhs) noexcept
    {
        mRoot = refRhs.mRoot;
        mNodeCount = refRhs.mNodeCount;
        mHdr.mLeft = refRhs.mHdr.mLeft;
        mHdr.mRight = refRhs.mHdr.mRight;
        fixHeader();
        refRhs.mRoot = nullptr;
        refRhs.mNodeCount = 0;
        refRhs.fixHeader();
    }

    //! Moves (or copies, if moving could throw) every stored value into a vector.
    std::vector<value_type> extractValues()
    {
        std::vector<value_type> vecValues;
        vecValues.reserve(mNodeCount);
        for (auto it = begin(); it != end(); ++it)
            vecValues.push_back(std::move_if_noexcept(asMutableNode(it.node())->mValue));
        return vecValues;
    }

    /**
     * \internal
     * \brief Pre-order walk of the subtree at \p pTop using parent links (no recursion, no allocation).
     *
     * \param goLeft  `bool(node, axis)`: descend into the left child?
     * \param goRight `bool(node, axis)`: descend into the right child?
     * \param visit   `bool(node, depth)`: return \c true to stop; that node is returned.
     * \return The node at which \p visit stopped, or \c nullptr.
     */
    template <class GoLeft, class GoRight, class Visit>
    LinkConstType walkSubtree(LinkConstType pTop, size_type depth, GoLeft&& goLeft,
                              GoRight&& goRight, Visit&& visit) const
    {
        if (!pTop) return nullptr;
        LinkConstType n = pTop;
        std::size_t axis = depth % K;
        for (;;) {
            if (visit(n, depth)) return n;
            if (n->mLeft && goLeft(n, axis)) {
                n = leftOf(n);
            } else if (n->mRight && goRight(n, axis)) {
                n = rightOf(n);
            } else {
                // climb until an unexplored right sibling subtree is wanted
                for (;;) {
                    if (n == pTop) return nullptr;
                    LinkConstType const p = parentOf(n);
                    --depth;
                    axis = prevAxis(axis);
                    if (n == p->mLeft && p->mRight && goRight(p, axis)) {
                        n = rightOf(p);
                        break;
                    }
                    n = p;
                }
            }
            ++depth;
            axis = nextAxis(axis);
        }
    }

    //! Calls \p fn on every value inside \p region; prunes subtrees on the split axis.
    template <class Fn>
    void forEachInRegion(Region const& region, Fn&& fn) const
    {
        if (mRoot) regionRecurse(mRoot, 0, kMaxRecursion, region, fn);
    }

    //! Recursion depth after which range queries switch to the (slightly slower) iterative walk.
    static constexpr size_type kMaxRecursion = 128;

    // left subtree holds values <= node on its axis; right subtree holds values >= node
    [[nodiscard]] bool regionGoLeft(Region const& region, LinkConstType n, std::size_t axis) const
    {
        return !region.mComparator(mAcc(n->mValue, axis), region.mSubLow[axis]);
    }
    [[nodiscard]] bool regionGoRight(Region const& region, LinkConstType n, std::size_t axis) const
    {
        return !region.mComparator(region.mSubHigh[axis], mAcc(n->mValue, axis));
    }

    // Recursion is a little faster than climbing parent links; depth is capped so a
    // degenerate tree falls back to walkSubtree instead of overflowing the stack.
    template <class Fn>
    void regionRecurse(LinkConstType n, std::size_t axis, size_type budget, Region const& region,
                       Fn& fn) const
    {
        if (budget == 0) {
            walkSubtree(
                n, axis,
                [&](LinkConstType p, std::size_t a) { return regionGoLeft(region, p, a); },
                [&](LinkConstType p, std::size_t a) { return regionGoRight(region, p, a); },
                [&](LinkConstType p, size_type) {
                    if (region.encloses(p->mValue)) fn(p->mValue);
                    return false;
                });
            return;
        }
        if (region.encloses(n->mValue)) fn(n->mValue);
        std::size_t const next = nextAxis(axis);
        if (n->mLeft && regionGoLeft(region, n, axis))
            regionRecurse(leftOf(n), next, budget - 1, region, fn);
        if (n->mRight && regionGoRight(region, n, axis))
            regionRecurse(rightOf(n), next, budget - 1, region, fn);
    }

    template <class SearchVal, class Predicate>
    std::pair<const_iterator, distance_type> nearestImpl(
        SearchVal const& refSearchVal, std::optional<distance_type> optBound, Predicate fnPred) const
    {
        if (!mRoot) return {end(), optBound.value_or(distance_type(0))};

        distance_type const rootDistSq =
            accumulateNodeDistance(K, mDist, mAcc, mRoot->mValue, refSearchVal);
        distance_type distSqInit = rootDistSq;
        bool rootIsCandidate = true;
        if (optBound) {
            distance_type const boundSq = *optBound * *optBound;
            rootIsCandidate = fnPred(mRoot->mValue) && rootDistSq <= boundSq;
            if (!rootIsCandidate) distSqInit = boundSq;
        }
        auto const best = nodeNearest<K>(0, refSearchVal, mRoot, &mHdr, mRoot, distSqInit,
                                         mCmp, mAcc, mDist, fnPred);
        // make sure we didn't just get stuck with the (unqualified) root node
        if (rootIsCandidate || best.first != mRoot)
            return {const_iterator(best.first), best.second.second};
        return {end(), *optBound};
    }

    // --- structural modification --------------------------------------------

    //! Links a freshly created node into the tree (iterative descent).
    iterator linkNode(LinkType pNew) noexcept
    {
        ++mNodeCount;
        if (!mRoot) {
            mRoot = pNew;
            pNew->mParent = &mHdr;
            mHdr.mLeft = mHdr.mRight = pNew;
            return iterator(pNew);
        }
        LinkType p = mRoot;
        std::size_t axis = 0;
        for (;;) {
            if (lessOnAxis(pNew->mValue, p->mValue, axis)) {
                if (!p->mLeft) {
                    p->mLeft = pNew;
                    if (p == leftmost()) mHdr.mLeft = pNew;
                    break;
                }
                p = leftOf(p);
            } else {
                if (!p->mRight) {
                    p->mRight = pNew;
                    if (p == rightmost()) mHdr.mRight = pNew;
                    break;
                }
                p = rightOf(p);
            }
            axis = nextAxis(axis);
        }
        pNew->mParent = p;
        return iterator(pNew);
    }

    /**
     * \internal
     * \brief Builds a balanced tree from \p vecValues (median split per level). \pre empty tree.
     * \tparam MoveValues Move elements out of \p vecValues instead of copying them.
     * Strong-ish guarantee: on exception the tree is left empty.
     */
    template <bool MoveValues>
    void buildBalanced(std::vector<value_type>& vecValues)
    {
        assert(!mRoot);
        if (vecValues.empty()) return;
        try {
            buildRange<MoveValues>(vecValues.begin(), vecValues.end(), 0, &mHdr, false);
        } catch (...) {
            clear();
            throw;
        }
        refreshExtremes();
    }

    // Recursion depth is ceil(log2(n)) – the tree being built is balanced.
    // Each node is linked in as soon as it exists, so clear() can always clean up.
    template <bool MoveValues, std::random_access_iterator Iter>
    void buildRange(Iter itFirst, Iter itLast, std::size_t axis, BasePtr pParent, bool bLeft)
    {
        while (itFirst != itLast) {
            Iter const itMid = itFirst + (itLast - itFirst) / 2;
            std::nth_element(itFirst, itMid, itLast, NodeDimCompare(axis, mAcc, mCmp));
            LinkType pNode;
            if constexpr (MoveValues)
                pNode = this->createNode(std::move(*itMid));
            else
                pNode = this->createNode(std::as_const(*itMid));
            pNode->mParent = pParent;
            if (pParent == &mHdr)
                mRoot = pNode;
            else
                (bLeft ? pParent->mLeft : pParent->mRight) = pNode;
            ++mNodeCount;
            std::size_t const next = nextAxis(axis);
            buildRange<MoveValues>(itFirst, itMid, next, pNode, true);
            // tail-iterate on the right half
            itFirst = itMid + 1;
            axis = next;
            pParent = pNode;
            bLeft = false;
        }
    }

    [[nodiscard]] size_type depthOf(BaseConstPtr pNode) const noexcept
    {
        size_type level = 0;
        while ((pNode = pNode->mParent) != &mHdr) ++level;
        return level;
    }

    //! Node with the smallest coordinate on \p dim in the subtree at \p pTop (depth \p depth).
    std::pair<LinkType, size_type> findMinOnAxis(LinkConstType pTop, size_type depth,
                                                 std::size_t dim) const
    {
        std::pair<LinkConstType, size_type> best{pTop, depth};
        walkSubtree(
            pTop, depth, [](LinkConstType, std::size_t) { return true; },
            // right subtree of a node split on `dim` can't hold anything smaller
            [dim](LinkConstType, std::size_t a) { return a != dim; },
            [&](LinkConstType n, size_type d) {
                if (lessOnAxis(n->mValue, best.first->mValue, dim)) best = {n, d};
                return false;
            });
        return {asMutableNode(best.first), best.second};
    }

    //! Node with the largest coordinate on \p dim in the subtree at \p pTop (depth \p depth).
    std::pair<LinkType, size_type> findMaxOnAxis(LinkConstType pTop, size_type depth,
                                                 std::size_t dim) const
    {
        std::pair<LinkConstType, size_type> best{pTop, depth};
        walkSubtree(
            pTop, depth, [dim](LinkConstType, std::size_t a) { return a != dim; },
            [](LinkConstType, std::size_t) { return true; },
            [&](LinkConstType n, size_type d) {
                if (lessOnAxis(best.first->mValue, n->mValue, dim)) best = {n, d};
                return false;
            });
        return {asMutableNode(best.first), best.second};
    }

    //! Points whichever link refers to \p pOld (parent's child slot or the root) at \p pNew.
    void replaceChildLink(BasePtr pParent, BasePtr pOld, BasePtr pNew) noexcept
    {
        if (pParent == &mHdr)
            mRoot = static_cast<LinkType>(pNew);
        else if (pParent->mLeft == pOld)
            pParent->mLeft = pNew;
        else
            pParent->mRight = pNew;
    }

    //! Exchanges the tree positions of \p a and its descendant \p b (values stay in their nodes).
    void swapPositions(LinkType a, LinkType b) noexcept
    {
        BasePtr const ap = a->mParent, al = a->mLeft, ar = a->mRight;
        BasePtr const bp = b->mParent, bl = b->mLeft, br = b->mRight;
        replaceChildLink(ap, a, b);
        b->mParent = ap;
        if (bp == a) {
            if (al == b) {
                b->mLeft = a;
                b->mRight = ar;
                if (ar) ar->mParent = b;
            } else {
                b->mRight = a;
                b->mLeft = al;
                if (al) al->mParent = b;
            }
            a->mParent = b;
        } else {
            (bp->mLeft == b ? bp->mLeft : bp->mRight) = a;
            b->mLeft = al;
            b->mRight = ar;
            if (al) al->mParent = b;
            if (ar) ar->mParent = b;
            a->mParent = bp;
        }
        a->mLeft = bl;
        a->mRight = br;
        if (bl) bl->mParent = a;
        if (br) br->mParent = a;
    }

    /**
     * \internal
     * \brief Detaches \p pDead (at depth \p depth) from the tree, keeping all invariants.
     *
     * The node is repeatedly swapped with a replacement from its subtree – the minimum of the
     * right subtree or the maximum of the left subtree on its split axis – until it is a leaf,
     * then cut off. Iterative; no allocation.
     */
    void unlinkNode(LinkType pDead, size_type depth) noexcept
    {
        while (!isLeaf(pDead)) {
            std::size_t const dim = depth % K;
            LinkConstType const l = leftOf(pDead);
            LinkConstType const r = rightOf(pDead);
            std::pair<LinkType, size_type> repl;
            if (!l)
                repl = findMinOnAxis(r, depth + 1, dim);
            else if (!r)
                repl = findMaxOnAxis(l, depth + 1, dim);
            else if (lessOnAxis(r->mValue, l->mValue, dim))
                // children compare "backwards": effectively a coin flip that keeps the tree balanced
                repl = findMinOnAxis(r, depth + 1, dim);
            else
                repl = findMaxOnAxis(l, depth + 1, dim);
            swapPositions(pDead, repl.first);
            depth = repl.second;
        }
        replaceChildLink(pDead->mParent, pDead, nullptr);
        refreshExtremes();
    }

    // --- raw link accessors -------------------------------------------------

    [[nodiscard]] LinkConstType leftmost() const noexcept
    {
        return static_cast<LinkConstType>(mHdr.mLeft);
    }
    [[nodiscard]] LinkConstType rightmost() const noexcept
    {
        return static_cast<LinkConstType>(mHdr.mRight);
    }

    static LinkType parentOf(BasePtr N) noexcept { return static_cast<LinkType>(N->mParent); }
    static LinkConstType parentOf(BaseConstPtr N) noexcept
    {
        return static_cast<LinkConstType>(N->mParent);
    }
    static LinkType leftOf(BasePtr N) noexcept { return static_cast<LinkType>(N->mLeft); }
    static LinkConstType leftOf(BaseConstPtr N) noexcept
    {
        return static_cast<LinkConstType>(N->mLeft);
    }
    static LinkType rightOf(BasePtr N) noexcept { return static_cast<LinkType>(N->mRight); }
    static LinkConstType rightOf(BaseConstPtr N) noexcept
    {
        return static_cast<LinkConstType>(N->mRight);
    }
    static bool isLeaf(BaseConstPtr N) noexcept { return !N->mLeft && !N->mRight; }

    //! Single place for const-to-mutable node pointer when the tree owns the node (erase, delete).
    static LinkType asMutableNode(LinkConstType pNode) noexcept
    {
        return const_cast<LinkType>(pNode);
    }

    LinkType mRoot = nullptr;
    NodeBase mHdr{nullptr, &mHdr, &mHdr};  //!< Sentinel: mLeft = leftmost, mRight = rightmost.
    size_type mNodeCount = 0;
    KDTREE_NO_UNIQUE_ADDRESS Acc mAcc{};
    KDTREE_NO_UNIQUE_ADDRESS Cmp mCmp{};
    KDTREE_NO_UNIQUE_ADDRESS Dist mDist{};

#ifdef KDTREE_DEFINE_OSTREAM_OPERATORS
    friend std::ostream& operator<<(std::ostream& o, KDTree const& tree)
    {
        o << "meta node:   " << tree.mHdr << '\n';
        o << "root node:   " << tree.mRoot << '\n';

        if (tree.empty()) return o << "[empty " << K << "d-tree " << &tree << "]";

        o << "nodes total: " << tree.size() << '\n';
        o << "dimensions:  " << K << '\n';
        tree.walkSubtree(
            tree.mRoot, 0, [](LinkConstType, std::size_t) { return true; },
            [](LinkConstType, std::size_t) { return true; },
            [&o](LinkConstType n, size_type) {
                o << *n << '\n';
                return false;
            });
        return o;
    }
#endif
};

/** @} */

}  // namespace libkdtree
