/** \file
 * \brief Tree node types and helpers used by \ref KDTree.
 * \ingroup libkdtree
 * \author Martin F. Krafft <libkdtree@pobox.madduck.net>
 */

#pragma once
#ifdef KDTREE_DEFINE_OSTREAM_OPERATORS
#include <ostream>
#endif
#include <cstddef>
#include <cmath>
#include <utility>

#include "function.hpp"

namespace libkdtree
{
//! Intrusive binary tree links; \ref TreeNode "TreeNode" adds the stored value.
struct NodeBase {
    using BasePtr = NodeBase*;
    using BaseConstPtr = NodeBase const*;

    BasePtr mParent = nullptr;  //!< Parent node (the tree header for the root; \c nullptr for the header itself).
    BasePtr mLeft = nullptr;    //!< Left child (header: leftmost node).
    BasePtr mRight = nullptr;   //!< Right child (header: rightmost node).

    constexpr NodeBase() noexcept = default;
    constexpr NodeBase(BasePtr const parent_in, BasePtr const left_in = nullptr,
                       BasePtr const right_in = nullptr) noexcept
        : mParent(parent_in), mLeft(left_in), mRight(right_in)
    {
    }

    //! Leftmost descendant (minimum in threaded order used by the tree).
    [[nodiscard]] static BasePtr Minimum(BasePtr pNode) noexcept
    {
        while (pNode->mLeft) pNode = pNode->mLeft;
        return pNode;
    }

    //! Rightmost descendant.
    [[nodiscard]] static BasePtr Maximum(BasePtr pNode) noexcept
    {
        while (pNode->mRight) pNode = pNode->mRight;
        return pNode;
    }

#ifdef KDTREE_DEFINE_OSTREAM_OPERATORS
    template <typename Char, typename Traits>
    friend std::basic_ostream<Char, Traits>& operator<<(
        std::basic_ostream<Char, Traits>& out, NodeBase const& node)
    {
        out << &node;
        out << " parent: " << node.mParent;
        out << "; left: " << node.mLeft;
        out << "; right: " << node.mRight;
        return out;
    }
#endif
};

//! KD-tree node: \ref NodeBase links plus stored \ref mValue.
template <typename Val>
struct TreeNode : public NodeBase {
    using NodeBase::BasePtr;
    using LinkType = TreeNode*;

    Val mValue;  //!< User-visible element at this node.

    //! Constructs the stored value in place from \p args; links start null.
    template <typename... Args>
    explicit TreeNode(std::in_place_t, Args&&... args)
        : NodeBase(), mValue(std::forward<Args>(args)...)
    {
    }

#ifdef KDTREE_DEFINE_OSTREAM_OPERATORS
    template <typename Char, typename Traits>
    friend std::basic_ostream<Char, Traits>& operator<<(
        std::basic_ostream<Char, Traits>& out, TreeNode<Val> const& node)
    {
        out << &node;
        out << ' ' << node.mValue;
        out << "; parent: " << node.mParent;
        out << "; left: " << node.mLeft;
        out << "; right: " << node.mRight;
        return out;
    }
#endif
};

//! Compares two values on a single axis \p m_uDim using accessor + \c Cmp.
template <typename Val, typename Acc, typename Cmp>
class NodeDimensionComparator
{
public:
    NodeDimensionComparator(std::size_t const uDim, Acc const& refAcc, Cmp const& refCmp)
        : m_uDim(uDim), m_acc(refAcc), m_cmp(refCmp)
    {
    }

    [[nodiscard]] bool operator()(Val const& refA, Val const& refB) const
    {
        return m_cmp(m_acc(refA, m_uDim), m_acc(refB, m_uDim));
    }

private:
    std::size_t m_uDim;
    KDTREE_NO_UNIQUE_ADDRESS Acc m_acc;
    KDTREE_NO_UNIQUE_ADDRESS Cmp m_cmp;
};

//! \c true if `refAcc(refA,dim)` is ordered before `refAcc(refB,dim)` under \p refCmp.
template <typename ValA, typename ValB, typename Cmp, typename Acc>
[[nodiscard]] inline bool compareNodesOnDim(std::size_t const uDim, Cmp const& refCmp,
                                              Acc const& refAcc, ValA const& refA, ValB const& refB)
{
    return refCmp(refAcc(refA, uDim), refAcc(refB, uDim));
}

//! Per-axis contribution from \p refDist (e.g. squared difference) at dimension \p uDim.
template <typename ValA, typename ValB, typename Dist, typename Acc>
[[nodiscard]] inline typename Dist::distance_type nodeDistance(std::size_t const uDim,
                                                               Dist const& refDist,
                                                               Acc const& refAcc,
                                                               ValA const& refA,
                                                               ValB const& refB)
{
    return refDist(refAcc(refA, uDim), refAcc(refB, uDim));
}

//! Sums \ref nodeDistance for dimensions `0 .. uDim-1` (used as squared Euclidean distance when \p refDist is squared per axis).
template <typename ValA, typename ValB, typename Dist, typename Acc>
[[nodiscard]] inline typename Dist::distance_type accumulateNodeDistance(
    std::size_t const uDim, Dist const& refDist, Acc const& refAcc, ValA const& refA,
    ValB const& refB)
{
    typename Dist::distance_type distSum = 0;
    for (std::size_t uI = 0; uI < uDim; ++uI)
        distSum += nodeDistance(uI, refDist, refAcc, refA, refB);
    return distSum;
}

//! Child pointer followed for insertion / descent on axis \p uDim (left if search key is "less" on that axis).
template <typename Val, typename Cmp, typename Acc, typename NodeType>
[[nodiscard]] inline const NodeType* nodeDescend(std::size_t const uDim, Cmp const& refCmp,
                                                   Acc const& refAcc, Val const& val,
                                                   const NodeType* pNode)
{
    if (compareNodesOnDim(uDim, refCmp, refAcc, val, pNode->mValue))
        return static_cast<const NodeType*>(pNode->mLeft);
    return static_cast<const NodeType*>(pNode->mRight);
}

/**
 * \internal
 * \brief Single nearest-neighbor walk for a fixed tree layout.
 *
 * Iterative (uses parent links, no recursion or allocation).
 * \param distSqMax Squared distance threshold (best-so-far); comparisons use squared metric internally.
 * \return Best node and pair `{level, sqrt(distSqMax)}` for the best found point.
 */
template <std::size_t uK, class SearchVal, typename NodeType, typename Cmp, typename Acc,
          typename Dist, typename Predicate>
[[nodiscard]] inline std::pair<const NodeType*,
                               std::pair<std::size_t, typename Dist::distance_type>>
nodeNearest(std::size_t uDim, SearchVal const& val,
            const NodeType* pNode, const NodeBase* pEnd, const NodeType* pBest,
            typename Dist::distance_type distSqMax, Cmp const& refCmp, Acc const& refAcc,
            Dist const& refDist, Predicate fnPred)
{
    using NodePtr = const NodeType*;
    using DistSq = typename Dist::distance_type;
    NodePtr pCur = pNode;
    NodePtr pChild = nodeDescend(uDim % uK, refCmp, refAcc, val, pNode);
    std::size_t uCurDim = uDim + 1;
    while (pChild) {
        if (fnPred(pChild->mValue)) {
            DistSq const distSq =
                accumulateNodeDistance(uK, refDist, refAcc, val, pChild->mValue);
            if (distSq <= distSqMax) {
                pBest = pChild;
                distSqMax = distSq;
                uDim = uCurDim;
            }
        }
        pCur = pChild;
        pChild = nodeDescend(uCurDim % uK, refCmp, refAcc, val, pChild);
        ++uCurDim;
    }
    pChild = pCur;
    --uCurDim;
    pCur = nullptr;
    NodePtr pProbe = pChild;
    NodePtr pProbePrev = pProbe;
    NodePtr pNear;
    NodePtr pFar;
    std::size_t uProbeDim = uCurDim;
    if (compareNodesOnDim(uProbeDim % uK, refCmp, refAcc, val, pProbe->mValue))
        pNear = static_cast<NodePtr>(pProbe->mRight);
    else
        pNear = static_cast<NodePtr>(pProbe->mLeft);
    if (pNear && nodeDistance(uProbeDim % uK, refDist, refAcc, val, pProbe->mValue) <=
                     distSqMax) {
        pProbe = pNear;
        ++uProbeDim;
    }
    while (pChild != pEnd) {
        while (pProbe != pChild) {
            if (compareNodesOnDim(uProbeDim % uK, refCmp, refAcc, val, pProbe->mValue)) {
                pNear = static_cast<NodePtr>(pProbe->mLeft);
                pFar = static_cast<NodePtr>(pProbe->mRight);
            } else {
                pNear = static_cast<NodePtr>(pProbe->mRight);
                pFar = static_cast<NodePtr>(pProbe->mLeft);
            }
            if (pProbePrev == pProbe->mParent) {
                if (fnPred(pProbe->mValue)) {
                    DistSq const distSq =
                        accumulateNodeDistance(uK, refDist, refAcc, val, pProbe->mValue);
                    if (distSq <= distSqMax) {
                        pBest = pProbe;
                        distSqMax = distSq;
                        uDim = uProbeDim;
                    }
                }
                pProbePrev = pProbe;
                if (pNear) {
                    pProbe = pNear;
                    ++uProbeDim;
                } else if (pFar && nodeDistance(uProbeDim % uK, refDist, refAcc, val,
                                                 pProbe->mValue) <= distSqMax) {
                    pProbe = pFar;
                    ++uProbeDim;
                } else {
                    pProbe = static_cast<NodePtr>(pProbe->mParent);
                    --uProbeDim;
                }
            } else {
                if (pProbePrev == pNear && pFar &&
                    nodeDistance(uProbeDim % uK, refDist, refAcc, val, pProbe->mValue) <=
                        distSqMax) {
                    pProbePrev = pProbe;
                    pProbe = pFar;
                    ++uProbeDim;
                } else {
                    pProbePrev = pProbe;
                    pProbe = static_cast<NodePtr>(pProbe->mParent);
                    --uProbeDim;
                }
            }
        }
        pCur = pChild;
        pChild = static_cast<NodePtr>(pChild->mParent);
        --uCurDim;
        pProbePrev = pChild;
        pProbe = pChild;
        uProbeDim = uCurDim;
        if (pChild != pEnd) {
            if (pCur == pChild->mLeft)
                pNear = static_cast<NodePtr>(pChild->mRight);
            else
                pNear = static_cast<NodePtr>(pChild->mLeft);
            if (pNear && nodeDistance(uCurDim % uK, refDist, refAcc, val, pChild->mValue) <=
                              distSqMax) {
                pProbe = pNear;
                ++uProbeDim;
            }
        }
    }
    return {pBest, std::pair<std::size_t, typename Dist::distance_type>(
                       uDim, std::sqrt(distSqMax))};
}

}  // namespace libkdtree

/* COPYRIGHT --
 *
 * This file is part of libkdtree++, a C++ template KD-Tree sorting container.
 * libkdtree++ is (c) 2004-2007 Martin F. Krafft <libkdtree@pobox.madduck.net>
 * and Sylvain Bougerel <sylvain.bougerel.devel@gmail.com> distributed under the
 * terms of the Artistic License 2.0. See the ./COPYING file in the source tree
 * root for more information.
 *
 * THIS PACKAGE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES
 * OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */
