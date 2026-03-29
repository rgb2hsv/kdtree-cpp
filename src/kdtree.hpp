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
 *    * The right branch is <= its parent (same as binary tree)
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
#define KDTREE_VERSION 701
//! String library version, same logical version as \ref KDTREE_VERSION (form `major_minor[_patch]`).
#define KDTREE_LIB_VERSION "0_7_1"

#include <vector>

#ifdef KDTREE_CHECK_PERFORMANCE_COUNTERS
#include <map>
#endif
#include <algorithm>
#include <concepts>
#include <functional>

#ifdef KDTREE_DEFINE_OSTREAM_OPERATORS
#include <ostream>
#include <stack>
#endif

#include <cmath>
#include <cstddef>
#include <cassert>
#include <iterator>
#include <limits>

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
unsigned long long numDistCalcs = 0;
#endif

/**
 * \brief K-dimensional tree storing elements of type \c Val.
 *
 * Each level splits along one axis (dimension), cycling \f$0 \ldots K-1\f$.
 * Insertion uses a single-branch comparison; \ref find must sometimes explore
 * both subtrees because the implementation uses a \f$\le\f$ (not strict \f$<\f$)
 * discipline on the active dimension—see the notes at the top of this header.
 *
 * \tparam K Number of dimensions.
 * \tparam Val Element type stored in the tree (e.g. a point type).
 * \tparam Acc Coordinate accessor: `result_type operator()(Val const&, std::size_t dim) const`.
 *         Default \ref BracketAccessor requires `Val::value_type` and `operator[]`.
 * \tparam Dist Per-axis distance functor; its `distance_type` is accumulated across
 *         dimensions for nearest-neighbor search. Default \ref SquaredDifference yields
 *         Euclidean distance in \ref findNearest results (square root of the sum of per-axis squares).
 * \tparam Cmp Strict weak ordering on scalar coordinates; default `std::less`.
 * \tparam Alloc Allocator for `TreeNode<Val>` nodes.
 */
template <std::size_t K, typename Val,
          typename Acc = BracketAccessor<Val>,
          typename Dist = SquaredDifference<typename Acc::result_type,
                                              typename Acc::result_type>,
          typename Cmp = std::less<typename Acc::result_type>,
          typename Alloc = std::allocator<TreeNode<Val> > >
class KDTree : protected AllocBase<Val, Alloc>
{
protected:
    using Base = AllocBase<Val, Alloc>;
    using allocator_type = typename Base::allocator_type;

    using BasePtr = NodeBase*;
    using BaseConstPtr = NodeBase const*;
    using LinkType = TreeNode<Val>*;
    using LinkConstType = TreeNode<Val> const*;

    using NodeDimCompare = NodeDimensionComparator<Val, Acc, Cmp>;

public:
    //! Axis-aligned box for \ref countWithinRange, \ref findWithinRange, and \ref visitWithinRange.
    using Region = KdTreeRegion<K, Val, typename Acc::result_type, Acc, Cmp>;
    using value_type = Val;
    using pointer = value_type*;
    using const_pointer = value_type const*;
    using reference = value_type&;
    using const_reference = value_type const&;
    using subvalue_type = typename Acc::result_type;
    using distance_type = typename Dist::distance_type;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    //! Default constructor: empty tree, optional accessor, distance functor, comparator, allocator.
    KDTree(Acc const& refAcc = Acc(), Dist const& refDist = Dist(),
           Cmp const& refCmp = Cmp(),
           const allocator_type& refAlloc = allocator_type())
        : Base(refAlloc),
          mHdr(),
          mNodeCount(0),
          mAcc(refAcc),
          mCmp(refCmp),
          mDist(refDist)
    {
        initializeEmpty();
    }

    //! Copy constructor: rebuilds an optimized tree from \p refRhs (bulk optimize, not per-insert).
    KDTree(const KDTree& refRhs)
        : Base(refRhs.getAllocator()),
          mHdr(),
          mNodeCount(0),
          mAcc(refRhs.mAcc),
          mCmp(refRhs.mCmp),
          mDist(refRhs.mDist)
    {
        initializeEmpty();
        // this is slow:
        // this->insert(begin(), __x.begin(), __x.end());
        // this->optimise();

        // this is much faster, as it skips a lot of useless work
        // do the optimisation before inserting
        // Needs to be stored in a vector first as optimize()
        // sorts the data in the passed iterators directly.
        std::vector<value_type> temp;
        temp.reserve(refRhs.size());
        std::copy(refRhs.begin(), refRhs.end(), std::back_inserter(temp));
        optimize(temp.begin(), temp.end(), 0);
    }

    /**
     * \brief Range constructor: copies \p [itFirst, itLast) into a temporary vector, then bulk-builds the tree.
     * \tparam Iter Input iterator with `value_type` convertible to \c value_type.
     * \note Iterators are read twice (for `distance` / `copy`); single-pass iterators are not supported.
     */
    template <std::input_iterator Iter>
    KDTree(Iter itFirst, Iter itLast,
           Acc const& refAcc = Acc(), Dist const& refDist = Dist(),
           Cmp const& refCmp = Cmp(),
           const allocator_type& refAlloc = allocator_type())
        : Base(refAlloc),
          mHdr(),
          mNodeCount(0),
          mAcc(refAcc),
          mCmp(refCmp),
          mDist(refDist)
    {
        initializeEmpty();
        // this is slow:
        // this->insert(begin(), __first, __last);
        // this->optimise();

        // this is much faster, as it skips a lot of useless work
        // do the optimisation before inserting
        // Needs to be stored in a vector first as optimize()
        // sorts the data in the passed iterators directly.
        std::vector<value_type> temp;
        temp.reserve(std::distance(itFirst, itLast));
        std::copy(itFirst, itLast, std::back_inserter(temp));
        optimize(temp.begin(), temp.end(), 0);

        // NOTE: this will BREAK users that are passing in
        // read-once data via the iterator...
        // We increment __first all the way to __last once within
        // the distance() call, and again within the copy() call.
        //
        // This should end up using some funky C++ concepts or
        // type traits to check that the iterators can be used in this way...
    }

    // this will CLEAR the tree and fill it with the contents
    // of 'writable_vector'.  it will use the passed vector directly,
    // and will basically resort the vector many times over while
    // optimising the tree.
    //
    /**
     * \brief Clears the tree and rebuilds it from \p vecWritable in place.
     *
     * The vector is reordered by the bulk \ref optimize routine (no extra copy).
     * \post `vecWritable` contents are a permutation of the previous elements (same multiset).
     */
    void assignOptimized(
        std::vector<value_type>& vecWritable)
    {
        this->clear();
        optimize(vecWritable.begin(), vecWritable.end(), 0);
    }

    KDTree& operator=(const KDTree& refRhs)
    {
        if (this != &refRhs) {
            mAcc = refRhs.mAcc;
            mDist = refRhs.mDist;
            mCmp = refRhs.mCmp;
            // this is slow:
            // this->insert(begin(), __x.begin(), __x.end());
            // this->optimise();

            // this is much faster, as it skips a lot of useless work
            // do the optimisation before inserting
            // Needs to be stored in a vector first as optimize()
            // sorts the data in the passed iterators directly.
            std::vector<value_type> temp;
            temp.reserve(refRhs.size());
            std::copy(refRhs.begin(), refRhs.end(), std::back_inserter(temp));
            assignOptimized(temp);
        }
        return *this;
    }

    //! Destroys all nodes (\ref clear).
    ~KDTree() { this->clear(); }

    //! Returns a copy of the node allocator.
    allocator_type getAllocator() const { return Base::getAllocator(); }

    //! Number of elements in the tree.
    size_type size() const { return mNodeCount; }

    //! Upper bound on container size (historical; effectively unbounded).
    [[nodiscard]] size_type maxSize() const
    {
        return std::numeric_limits<size_type>::max();
    }

    //! \c true if `size() == 0`.
    bool empty() const { return this->size() == 0; }

    //! Removes all elements and resets the tree to empty.
    void clear()
    {
        eraseSubtree(root());
        setLeftmost(&mHdr);
        setRightmost(&mHdr);
        setRoot(nullptr);
        mNodeCount = 0;
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
    const Dist& valueDistance() const { return mDist; }

    //! \copydoc valueDistance() const
    Dist& valueDistance() { return mDist; }

    //! In-order iterator over values (bidirectional). There is no mutable iterator.
    using const_iterator = TreeIterator<Val, const_reference, const_pointer>;
    using iterator = const_iterator;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using reverse_iterator = std::reverse_iterator<iterator>;

    //! Smallest element in tree order. Undefined if \ref empty (except matching \ref end).
    const_iterator begin() const { return const_iterator(leftmost()); }
    //! Past-the-end iterator; must not be dereferenced.
    const_iterator end() const
    {
        return const_iterator(static_cast<LinkConstType>(&mHdr));
    }
    const_reverse_iterator rbegin() const
    {
        return const_reverse_iterator(end());
    }
    const_reverse_iterator rend() const
    {
        return const_reverse_iterator(begin());
    }

    //! \copybrief insert(const_reference)
    //! \param ignored Position hint (ignored; same as \ref insert(const_reference)).
    iterator insert(iterator /* ignored */, const_reference refVal)
    {
        return this->insert(refVal);
    }

    /**
     * \brief Inserts \p refVal and returns an iterator to the new node.
     * \complexity \f$O(\log n)\f$ typical depth; worst-case \f$O(n)\f$ for a degenerate tree.
     */
    iterator insert(const_reference refVal)
    {
        if (!root()) {
            LinkType pWalk = createNode(refVal, &mHdr);
            ++mNodeCount;
            setRoot(pWalk);
            setLeftmost(pWalk);
            setRightmost(pWalk);
            return iterator(pWalk);
        }
        return insertNode(root(), refVal, 0);
    }

    //! Inserts each element in \p [itFirst, itLast) via \ref insert(const_reference).
    template <std::input_iterator Iter>
    void insert(Iter itFirst, Iter itLast)
    {
        for (; itFirst != itLast; ++itFirst) this->insert(*itFirst);
    }

    //! Inserts \p cDup copies of \p refValDup; \p itPos is only passed through to \ref insert(iterator,const_reference).
    void insert(iterator itPos, size_type cDup, const value_type& refValDup)
    {
        for (; cDup > 0; --cDup) this->insert(itPos, refValDup);
    }

    //! Inserts range \p [itFirst, itLast) at \p itPos (hint ignored per element).
    template <std::input_iterator Iter>
    void insert(iterator itPos, Iter itFirst, Iter itLast)
    {
        for (; itFirst != itLast; ++itFirst) this->insert(itPos, *itFirst);
    }

    /**
     * \brief Erases one element \ref find "equivalent" to \p refVal (same location in all dimensions).
     *
     * To remove a specific object (e.g. matching `operator==`), use \ref eraseExact.
     * \see find, eraseExact
     */
    void erase(const_reference refVal)
    {
        const_iterator b = this->find(refVal);
        this->erase(b);
    }

    void eraseExact(const_reference refVal)
    {
        this->erase(this->findExact(refVal));
    }

    /**
     * \brief Erases the node referenced by \p itErase.
     * \pre `itErase != end()`.
     */
    void erase(const_iterator const& itErase)
    {
        assert(itErase != this->end());
        LinkConstType target = itErase.node();
        LinkConstType n = target;
        size_type level = 0;
        while ((n = parentOf(n)) != &mHdr) ++level;
        LinkType const mutableTarget = asMutableNode(target);
        erase(mutableTarget, level);
        deleteNode(mutableTarget);
        --mNodeCount;
    }

    /* this does not work since erasure changes sort order
          void
          erase(const_iterator itRangeA, const_iterator const& itRangeB)
          {
            if (0 && itRangeA == this->begin() && itRangeB == this->end())
              {
                this->clear();
              }
            else
              {
                while (itRangeA != itRangeB)
                  this->erase(itRangeA++);
              }
          }
    */

    /**
     * \brief Finds any element with the same coordinates as \p refVal (all dimensions, via \c Cmp on accessor values).
     * \tparam SearchVal Type compatible with the accessor (need not be \c value_type).
     * \return Iterator to a matching element, or \ref end() if none.
     * \see findExact
     */
    template <class SearchVal>
    const_iterator find(SearchVal const& refVal) const
    {
        if (!root()) return this->end();
        return find(root(), refVal, 0);
    }

    /**
     * \brief Finds an element equal to \p refVal using `value_type::operator==`.
     *
     * Use when several stored values may share the same coordinates but differ otherwise
     * (e.g. distinct IDs). \ref find only matches by coordinate equivalence.
     * \tparam SearchVal Typically \c value_type or compatible for `==`.
     */
    template <class SearchVal>
    const_iterator findExact(SearchVal const& refVal) const
    {
        if (!root()) return this->end();
        return findExact(root(), refVal, 0);
    }

    //! Counts nodes inside the axis-aligned box centered at \p refVal with half-extent \p subR per axis (\ref findWithinRange "same metric").
    size_type countWithinRange(const_reference refVal,
                                 subvalue_type const subR) const
    {
        if (!root()) return 0;
        Region regRange(refVal, subR, mAcc, mCmp);
        return this->countWithinRange(regRange);
    }

    //! Counts nodes inside an arbitrary axis-aligned \ref KdTreeRegion "region".
    size_type countWithinRange(Region const& refRegion) const
    {
        if (!root()) return 0;

        Region regBounds(refRegion);
        return countWithinRangeRecursive(root(), refRegion, regBounds, 0);
    }

    /**
     * \brief Invokes \p visitor on each value inside the box centered at \p V with half-extent \p R per axis.
     * \return The visitor after traversal (for stateful functors).
     */
    template <typename SearchVal, class Visitor>
    Visitor visitWithinRange(SearchVal const& V, subvalue_type const R,
                               Visitor visitor) const
    {
        if (!root()) return visitor;
        Region region(V, R, mAcc, mCmp);
        return this->visitWithinRange(region, visitor);
    }

    template <class Visitor>
    Visitor visitWithinRange(Region const& REGION, Visitor visitor) const
    {
        if (root()) {
            Region bounds(REGION);
            return visitWithinRangeRec(visitor, root(), REGION, bounds, 0);
        }
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
    OutputIterator findWithinRange(SearchVal const& val,
                                      subvalue_type const range,
                                      OutputIterator out) const
    {
        if (!root()) return out;
        Region region(val, range, mAcc, mCmp);
        return this->findWithinRange(region, out);
    }

    template <typename OutputIterator>
    OutputIterator findWithinRange(Region const& region,
                                      OutputIterator out) const
    {
        if (root()) {
            Region bounds(region);
            out = findWithinRangeRec(out, root(), region, bounds, 0);
        }
        return out;
    }

    /**
     * \brief Nearest neighbor to \p refSearchVal in the Euclidean metric implied by \c Dist (default: sum of squared per-axis differences, square root returned).
     * \return `{iterator, distance}` or `{end(), 0}` if the tree is empty.
     */
    template <class SearchVal>
    std::pair<const_iterator, distance_type> findNearest(
        SearchVal const& refSearchVal) const
    {
        if (root()) {
            std::pair<const TreeNode<Val>*, std::pair<size_type, distance_type>> best =
                nodeNearest(
                    K, 0, refSearchVal, root(), &mHdr, root(),
                    accumulateNodeDistance(K, mDist, mAcc, root()->mValue,
                                           refSearchVal),
                    mCmp, mAcc, mDist, AlwaysTrue<value_type>());
            return std::pair<const_iterator, distance_type>(best.first,
                                                            best.second.second);
        }
        return std::pair<const_iterator, distance_type>(end(),
                                                        distance_type(0));
    }

    /**
     * \brief Nearest neighbor within maximum Euclidean distance \p distMaxBound.
     * \return Best pair inside the ball, or `{end(), distMaxBound}` if the root lies outside the ball
     *         and no acceptable point exists (see implementation: unqualified root outside radius).
     */
    template <class SearchVal>
    std::pair<const_iterator, distance_type> findNearest(
        SearchVal const& refSearchVal, distance_type distMaxBound) const
    {
        if (root()) {
            bool rootIsCandidate = false;
            const TreeNode<Val>* node = root();
            distance_type const rootDistSq =
                accumulateNodeDistance(K, mDist, mAcc, root()->mValue,
                                       refSearchVal);
            distance_type distSqInit = distMaxBound * distMaxBound;
            distance_type const bound_sq = distMaxBound * distMaxBound;
            if (rootDistSq <= bound_sq) {
                rootIsCandidate = true;
                distMaxBound = std::sqrt(rootDistSq);
                distSqInit = rootDistSq;
            }
            std::pair<const TreeNode<Val>*, std::pair<size_type, distance_type>> best =
                nodeNearest(K, 0, refSearchVal, root(), &mHdr,
                            node, distSqInit, mCmp, mAcc, mDist,
                            AlwaysTrue<value_type>());
            // make sure we didn't just get stuck with the root node...
            if (rootIsCandidate || best.first != root())
                return std::pair<const_iterator, distance_type>(
                    best.first, best.second.second);
        }
        return std::pair<const_iterator, distance_type>(end(), distMaxBound);
    }

    template <class SearchVal, class _Predicate>
    std::pair<const_iterator, distance_type> findNearestIf(
        SearchVal const& refSearchVal, distance_type distMaxBound, _Predicate fnPred) const
    {
        if (root()) {
            bool rootIsCandidate = false;
            const TreeNode<Val>* node = root();
            distance_type const rootDistSq =
                accumulateNodeDistance(K, mDist, mAcc, root()->mValue,
                                       refSearchVal);
            distance_type distSqInit = distMaxBound * distMaxBound;
            distance_type const bound_sq = distMaxBound * distMaxBound;
            if (fnPred(root()->mValue) && rootDistSq <= bound_sq) {
                rootIsCandidate = true;
                distMaxBound = std::sqrt(rootDistSq);
                distSqInit = rootDistSq;
            }
            std::pair<const TreeNode<Val>*, std::pair<size_type, distance_type>> best =
                nodeNearest(K, 0, refSearchVal, root(), &mHdr, node,
                            distSqInit, mCmp, mAcc, mDist, fnPred);
            // make sure we didn't just get stuck with the root node...
            if (rootIsCandidate || best.first != root())
                return std::pair<const_iterator, distance_type>(
                    best.first, best.second.second);
        }
        return std::pair<const_iterator, distance_type>(end(), distMaxBound);
    }

    /**
     * \brief Rebuilds the tree to a balanced shape by copying into a vector and bulk-inserting via median splits.
     * \complexity Approximately \f$O(n \log n)\f$.
     */
    void optimize()
    {
        std::vector<value_type> vecOptimise(this->begin(), this->end());
        this->clear();
        optimize(vecOptimise.begin(), vecOptimise.end(), 0);
    }

    //! Debug: recursively \c assert tree ordering invariants (no-op in \c NDEBUG). Expensive.
    void checkTreeInvariants() { checkNode(root(), 0); }

protected:
    void checkChildren(LinkConstType child, LinkConstType parent,
                           size_type const level, bool to_the_left)
    {
        assert(parent);
        if (child) {
            NodeDimCompare Compare(level % K, mAcc, mCmp);
            // REMEMBER! its a <= relationship for BOTH branches
            // for left-case (true), child<=node --> !(node<child)
            // for right-case (false), node<=child --> !(child<node)
            assert(
                !to_the_left ||
                !Compare(parent->mValue, child->mValue));  // check the left
            assert(to_the_left ||
                   !Compare(child->mValue,
                            parent->mValue));  // check the right
            // and recurse down the tree, checking everything
            checkChildren(leftOf(child), parent, level, to_the_left);
            checkChildren(rightOf(child), parent, level, to_the_left);
        }
    }

    void checkNode(LinkConstType node, size_type const level)
    {
        if (node) {
            // (comparing on this level)
            // everything to the left of this node must be smaller than this
            checkChildren(leftOf(node), node, level, true);
            // everything to the right of this node must be larger than this
            checkChildren(rightOf(node), node, level, false);

            checkNode(leftOf(node), level + 1);
            checkNode(rightOf(node), level + 1);
        }
    }

    void initializeEmpty()
    {
        setLeftmost(&mHdr);
        setRightmost(&mHdr);
        mHdr.mParent = nullptr;
        setRoot(nullptr);
    }

    iterator insertLeft(LinkType pNode, const_reference refVal)
    {
        setLeft(pNode, createNode(refVal));
        ++mNodeCount;
        setParent(leftOf(pNode), pNode);
        if (pNode == leftmost()) setLeftmost(leftOf(pNode));
        return iterator(leftOf(pNode));
    }

    iterator insertRight(LinkType pNode, const_reference refVal)
    {
        setRight(pNode, createNode(refVal));
        ++mNodeCount;
        setParent(rightOf(pNode), pNode);
        if (pNode == rightmost()) setRightmost(rightOf(pNode));
        return iterator(rightOf(pNode));
    }

    iterator insertNode(LinkType pNode, const_reference refVal, size_type const uLevel)
    {
        if (NodeDimCompare(uLevel % K, mAcc, mCmp)(refVal, pNode->mValue)) {
            if (!leftOf(pNode)) return insertLeft(pNode, refVal);
            return insertNode(leftOf(pNode), refVal, uLevel + 1);
        } else {
            if (!rightOf(pNode) || pNode == rightmost())
                return insertRight(pNode, refVal);
            return insertNode(rightOf(pNode), refVal, uLevel + 1);
        }
    }

    LinkType erase(LinkType dead_dad, size_type const level)
    {
        // find a new step_dad, he will become a drop-in replacement.
        LinkType step_dad = getEraseReplacement(dead_dad, level);

        // tell dead_dad's parent that his new child is step_dad
        if (dead_dad == root())
            setRoot(step_dad);
        else if (leftOf(parentOf(dead_dad)) == dead_dad)
            setLeft(parentOf(dead_dad), step_dad);
        else
            setRight(parentOf(dead_dad), step_dad);

        // deal with the left and right edges of the tree...
        // if the dead_dad was at the edge, then substitude...
        // but if there IS no new dead, then left_most is the dead_dad's parent
        if (dead_dad == leftmost())
            setLeftmost((step_dad ? step_dad : parentOf(dead_dad)));
        if (dead_dad == rightmost())
            setRightmost((step_dad ? step_dad : parentOf(dead_dad)));

        if (step_dad) {
            // step_dad gets dead_dad's parent
            setParent(step_dad, parentOf(dead_dad));

            // first tell the children that step_dad is their new dad
            if (leftOf(dead_dad)) setParent(leftOf(dead_dad), step_dad);
            if (rightOf(dead_dad)) setParent(rightOf(dead_dad), step_dad);

            // step_dad gets dead_dad's children
            setLeft(step_dad, leftOf(dead_dad));
            setRight(step_dad, rightOf(dead_dad));
        }

        return step_dad;
    }

    LinkType getEraseReplacement(LinkType node, size_type const level)
    {
        // if 'node' is null, then we can't do any better
        if (isLeaf(node)) return nullptr;

        std::pair<LinkType, size_type> candidate;
        // if there is nothing to the left, find a candidate on the right tree
        if (!leftOf(node))
            candidate = getJMin(
                std::pair<LinkType, size_type>(rightOf(node), level),
                level + 1);
        // ditto for the right
        else if ((!rightOf(node)))
            candidate = getJMax(
                std::pair<LinkType, size_type>(leftOf(node), level),
                level + 1);
        // we have both children ...
        else {
            // we need to do a little more work in order to find a good
            // candidate this is actually a technique used to choose a node from
            // either the left or right branch RANDOMLY, so that the tree has a
            // greater change of staying balanced. If this were a true binary
            // tree, we would always hunt down the right branch. See top for
            // notes.
            NodeDimCompare Compare(level % K, mAcc, mCmp);
            // Compare the children based on this level's criteria...
            // (this gives virtually random results)
            if (Compare(rightOf(node)->mValue, leftOf(node)->mValue))
                // the right is smaller, get our replacement from the SMALLEST
                // on the right
                candidate = getJMin(
                    std::pair<LinkType, size_type>(rightOf(node), level),
                    level + 1);
            else
                candidate = getJMax(
                    std::pair<LinkType, size_type>(leftOf(node), level),
                    level + 1);
        }

        // we have a candidate replacement by now.
        // remove it from the tree, but don't delete it.
        // it must be disconnected before it can be reconnected.
        LinkType parent = parentOf(candidate.first);
        if (leftOf(parent) == candidate.first)
            setLeft(parent, erase(candidate.first, candidate.second));
        else
            setRight(parent, erase(candidate.first, candidate.second));

        return candidate.first;
    }

    std::pair<LinkType, size_type> getJMin(
        std::pair<LinkType, size_type> const node, size_type const level)
    {
        using Result = std::pair<LinkType, size_type>;
        if (isLeaf(node.first)) return Result(node.first, level);

        NodeDimCompare Compare(node.second % K, mAcc, mCmp);
        Result candidate = node;
        if (leftOf(node.first)) {
            Result left = getJMin(Result(leftOf(node.first), node.second),
                                       level + 1);
            if (Compare(left.first->mValue, candidate.first->mValue))
                candidate = left;
        }
        if (rightOf(node.first)) {
            Result right = getJMin(
                Result(rightOf(node.first), node.second), level + 1);
            if (Compare(right.first->mValue, candidate.first->mValue))
                candidate = right;
        }
        if (candidate.first == node.first)
            return {candidate.first, level};

        return candidate;
    }

    std::pair<LinkType, size_type> getJMax(
        std::pair<LinkType, size_type> const node, size_type const level)
    {
        using Result = std::pair<LinkType, size_type>;

        if (isLeaf(node.first)) return Result(node.first, level);

        NodeDimCompare Compare(node.second % K, mAcc, mCmp);
        Result candidate = node;
        if (leftOf(node.first)) {
            Result left = getJMax(Result(leftOf(node.first), node.second),
                                       level + 1);
            if (Compare(candidate.first->mValue, left.first->mValue))
                candidate = left;
        }
        if (rightOf(node.first)) {
            Result right = getJMax(
                Result(rightOf(node.first), node.second), level + 1);
            if (Compare(candidate.first->mValue, right.first->mValue))
                candidate = right;
        }

        if (candidate.first == node.first)
            return {candidate.first, level};

        return candidate;
    }

    void eraseSubtree(LinkType pWalk)
    {
        while (pWalk) {
            eraseSubtree(rightOf(pWalk));
            LinkType pLeftHold = leftOf(pWalk);
            deleteNode(pWalk);
            pWalk = pLeftHold;
        }
    }

    const_iterator find(LinkConstType node, const_reference value,
                           size_type const level) const
    {
        // be aware! This is very different to normal binary searches, because
        // of the <= relationship used. See top for notes. Basically we have to
        // check ALL branches, as we may have an identical node in different
        // branches.
        const_iterator found = this->end();

        NodeDimCompare Compare(level % K, mAcc, mCmp);
        if (!Compare(node->mValue, value))  // note, this is a <= test
        {
            // this line is the only difference between findExact() and
            // find()
            if (matchesNode(node, value, level))
                return const_iterator(node);  // return right away
            if (leftOf(node)) found = find(leftOf(node), value, level + 1);
        }
        if (rightOf(node) && found == this->end() &&
            !Compare(value, node->mValue))  // note, this is a <= test
            found = find(rightOf(node), value, level + 1);
        return found;
    }

    const_iterator findExact(LinkConstType node, const_reference value,
                                 size_type const level) const
    {
        // be aware! This is very different to normal binary searches, because
        // of the <= relationship used. See top for notes. Basically we have to
        // check ALL branches, as we may have an identical node in different
        // branches.
        const_iterator found = this->end();

        NodeDimCompare Compare(level % K, mAcc, mCmp);
        if (!Compare(node->mValue, value))  // note, this is a <= test
        {
            // this line is the only difference between findExact() and
            // find()
            if (value == *const_iterator(node))
                return const_iterator(node);  // return right away
            if (leftOf(node))
                found = findExact(leftOf(node), value, level + 1);
        }

        // note: no else!  items that are identical can be down both branches
        if (rightOf(node) && found == this->end() &&
            !Compare(value, node->mValue))  // note, this is a <= test
            found = findExact(rightOf(node), value, level + 1);
        return found;
    }

    bool matchesNodeInDimension(LinkConstType pNode, const_reference refVal,
                              size_type const uLevel) const
    {
        NodeDimCompare Compare(uLevel % K, mAcc, mCmp);
        return !(Compare(pNode->mValue, refVal) || Compare(refVal, pNode->mValue));
    }

    bool matchesNodeInOtherDims(LinkConstType pNode, const_reference refVal,
                                     size_type const uLevel = 0) const
    {
        size_type uDimIdx = uLevel;
        while ((uDimIdx = (uDimIdx + 1) % K) != uLevel % K)
            if (!matchesNodeInDimension(pNode, refVal, uDimIdx)) return false;
        return true;
    }

    bool matchesNode(LinkConstType pNode, const_reference refVal,
                         size_type uLevel = 0) const
    {
        return matchesNodeInDimension(pNode, refVal, uLevel) &&
               matchesNodeInOtherDims(pNode, refVal, uLevel);
    }

    size_type countWithinRangeRecursive(LinkConstType pNode,
                                    Region const& refRegion,
                                    Region const& refBounds,
                                    size_type const uLevel) const
    {
        size_type count = 0;
        if (refRegion.encloses(nodeValue(pNode))) {
            ++count;
        }
        if (leftOf(pNode)) {
            Region regBounds(refBounds);
            regBounds.setHighBound(nodeValue(pNode), uLevel);
            if (refRegion.intersectsWith(regBounds))
                count += countWithinRangeRecursive(leftOf(pNode), refRegion, regBounds,
                                               uLevel + 1);
        }
        if (rightOf(pNode)) {
            Region regBounds(refBounds);
            regBounds.setLowBound(nodeValue(pNode), uLevel);
            if (refRegion.intersectsWith(regBounds))
                count += countWithinRangeRecursive(rightOf(pNode), refRegion,
                                               regBounds, uLevel + 1);
        }

        return count;
    }

    template <class Visitor>
    Visitor visitWithinRangeRec(Visitor visitor, LinkConstType N,
                                  Region const& REGION,
                                  Region const& BOUNDS,
                                  size_type const L) const
    {
        if (REGION.encloses(nodeValue(N))) {
            visitor(nodeValue(N));
        }
        if (leftOf(N)) {
            Region bounds(BOUNDS);
            bounds.setHighBound(nodeValue(N), L);
            if (REGION.intersectsWith(bounds))
                visitor = visitWithinRangeRec(visitor, leftOf(N), REGION,
                                                bounds, L + 1);
        }
        if (rightOf(N)) {
            Region bounds(BOUNDS);
            bounds.setLowBound(nodeValue(N), L);
            if (REGION.intersectsWith(bounds))
                visitor = visitWithinRangeRec(visitor, rightOf(N), REGION,
                                                bounds, L + 1);
        }

        return visitor;
    }

    template <typename OutputIterator>
    OutputIterator findWithinRangeRec(OutputIterator out,
                                         LinkConstType pNode,
                                         Region const& refRegion,
                                         Region const& refBounds,
                                         size_type const uLevel) const
    {
        if (refRegion.encloses(nodeValue(pNode))) {
            *out++ = nodeValue(pNode);
        }
        if (leftOf(pNode)) {
            Region regBounds(refBounds);
            regBounds.setHighBound(nodeValue(pNode), uLevel);
            if (refRegion.intersectsWith(regBounds))
                out = findWithinRangeRec(out, leftOf(pNode), refRegion,
                                           regBounds, uLevel + 1);
        }
        if (rightOf(pNode)) {
            Region regBounds(refBounds);
            regBounds.setLowBound(nodeValue(pNode), uLevel);
            if (refRegion.intersectsWith(regBounds))
                out = findWithinRangeRec(out, rightOf(pNode), refRegion,
                                           regBounds, uLevel + 1);
        }

        return out;
    }

    //! \internal Bulk-build: `nth_element` median split, recursive; reorders \p [itRangeA, itRangeB).
    template <std::random_access_iterator Iter>
    void optimize(Iter const& itRangeA, Iter const& itRangeB, size_type const uLevel)
    {
        if (itRangeA == itRangeB) return;
        NodeDimCompare Compare(uLevel % K, mAcc, mCmp);
        Iter itMid = itRangeA + (itRangeB - itRangeA) / 2;
        std::nth_element(itRangeA, itMid, itRangeB, Compare);
        this->insert(*itMid);
        if (itMid != itRangeA) optimize(itRangeA, itMid, uLevel + 1);
        if (++itMid != itRangeB) optimize(itMid, itRangeB, uLevel + 1);
    }

    LinkConstType root() const { return mRoot; }

    LinkType root() { return mRoot; }

    void setRoot(LinkType pNode) { mRoot = pNode; }

    LinkConstType leftmost() const
    {
        return static_cast<LinkConstType>(mHdr.mLeft);
    }

    void setLeftmost(NodeBase* a) { mHdr.mLeft = a; }

    LinkConstType rightmost() const
    {
        return static_cast<LinkConstType>(mHdr.mRight);
    }

    void setRightmost(NodeBase* a) { mHdr.mRight = a; }

    static LinkType parentOf(BasePtr N)
    {
        return static_cast<LinkType>(N->mParent);
    }

    static LinkConstType parentOf(BaseConstPtr N)
    {
        return static_cast<LinkConstType>(N->mParent);
    }

    static void setParent(BasePtr N, BasePtr p) { N->mParent = p; }

    static void setLeft(BasePtr N, BasePtr l) { N->mLeft = l; }

    static LinkType leftOf(BasePtr N)
    {
        return static_cast<LinkType>(N->mLeft);
    }

    static LinkConstType leftOf(BaseConstPtr N)
    {
        return static_cast<LinkConstType>(N->mLeft);
    }

    static void setRight(BasePtr N, BasePtr r) { N->mRight = r; }

    static LinkType rightOf(BasePtr N)
    {
        return static_cast<LinkType>(N->mRight);
    }

    static LinkConstType rightOf(BaseConstPtr N)
    {
        return static_cast<LinkConstType>(N->mRight);
    }

    static bool isLeaf(BaseConstPtr N)
    {
        return !leftOf(N) && !rightOf(N);
    }

    static const_reference nodeValue(LinkConstType N) { return N->mValue; }

    static const_reference nodeValue(BaseConstPtr N)
    {
        return static_cast<LinkConstType>(N)->mValue;
    }

    static LinkConstType Minimum(LinkConstType pNodeEnd)
    {
        return static_cast<LinkConstType>(NodeBase::Minimum(pNodeEnd));
    }

    static LinkConstType Maximum(LinkConstType pNodeEnd)
    {
        return static_cast<LinkConstType>(NodeBase::Maximum(pNodeEnd));
    }

    //! Single place for const-to-mutable node pointer when the tree owns the node (erase, delete).
    static LinkType asMutableNode(LinkConstType pNode) noexcept
    {
        return const_cast<LinkType>(pNode);
    }

    LinkType createNode(const_reference refVal,  //  = value_type(),
                           BasePtr const pParentArg = nullptr,
                           BasePtr const pLeftArg = nullptr,
                           BasePtr const pRightArg = nullptr)
    {
        typename Base::NoLeakAlloc allocGuard(this);
        LinkType pNewNode = allocGuard.get();
        Base::constructNode(pNewNode, refVal, pParentArg, pLeftArg, pRightArg);
        allocGuard.disconnect();
        return pNewNode;
    }

    /* WHAT was this for?
    LinkType
    cloneNode(LinkConstType pNodeEnd)
    {
      LinkType pAllocated = allocateNode(pNodeEnd->mValue);
      // TODO
      return pAllocated;
    }
    */

    void deleteNode(LinkType pNode)
    {
        Base::destroyNode(pNode);
        Base::deallocateNode(pNode);
    }

    LinkType mRoot;
    NodeBase mHdr;
    size_type mNodeCount;
    Acc mAcc;
    Cmp mCmp;
    Dist mDist;

#ifdef KDTREE_DEFINE_OSTREAM_OPERATORS
    friend std::ostream& operator<<(
        std::ostream& o,
        KDTree<K, Val, Acc, Dist, Cmp, Alloc> const& tree)
    {
        o << "meta node:   " << tree.mHdr << std::endl;
        o << "root node:   " << tree.mRoot << std::endl;

        if (tree.empty())
            return o << "[empty " << K << "d-tree " << &tree << "]";

        o << "nodes total: " << tree.size() << std::endl;
        o << "dimensions:  " << K << std::endl;

        using _Tree = KDTree<K, Val, Acc, Dist, Cmp, Alloc>;
        using LinkType = typename _Tree::LinkType;

        std::stack<LinkConstType> s;
        s.push(tree.root());

        while (!s.empty()) {
            LinkConstType n = s.top();
            s.pop();
            o << *n << std::endl;
            if (_Tree::leftOf(n)) s.push(_Tree::leftOf(n));
            if (_Tree::rightOf(n)) s.push(_Tree::rightOf(n));
        }

        return o;
    }
#endif
};

/** @} */

}  // namespace libkdtree





