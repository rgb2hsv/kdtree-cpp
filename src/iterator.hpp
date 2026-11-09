/** \file
 * \brief Bidirectional iterators over \ref KDTree elements (in-order).
 * \ingroup libkdtree
 * \author Martin F. Krafft <libkdtree@pobox.madduck.net>
 */

#pragma once

#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>

#include "node.hpp"

namespace libkdtree
{

template <std::size_t K, typename Val, typename Acc, typename Dist, typename Cmp, typename Alloc>
class KDTree;

//! \internal In-order traversal using parent links; used by \ref TreeIterator.
class BaseIterator
{
protected:
    using BaseConstPtr = NodeBase::BaseConstPtr;
    BaseConstPtr mNode = nullptr;

    constexpr BaseIterator() noexcept = default;
    constexpr explicit BaseIterator(BaseConstPtr pNode) noexcept : mNode(pNode) {}

    void Increment() noexcept
    {
        if (mNode->mRight) {
            mNode = mNode->mRight;
            while (mNode->mLeft) mNode = mNode->mLeft;
        } else {
            BaseConstPtr pParent = mNode->mParent;
            while (pParent && mNode == pParent->mRight) {
                mNode = pParent;
                pParent = mNode->mParent;
            }
            if (pParent) mNode = pParent;
        }
    }

    void Decrement() noexcept
    {
        if (!mNode->mParent) {
            // header (end()): step to the rightmost node
            mNode = mNode->mRight;
        } else if (mNode->mLeft) {
            BaseConstPtr pWalk = mNode->mLeft;
            while (pWalk->mRight) pWalk = pWalk->mRight;
            mNode = pWalk;
        } else {
            BaseConstPtr pParent = mNode->mParent;
            while (pParent && mNode == pParent->mLeft) {
                mNode = pParent;
                pParent = mNode->mParent;
            }
            if (pParent) mNode = pParent;
        }
    }

    template <std::size_t K, typename Val, typename Acc, typename Dist, typename Cmp, typename Alloc>
    friend class KDTree;
};

/**
 * \brief Bidirectional iterator referencing a \ref TreeNode "TreeNode"'s \c value.
 *
 * 	param Val Stored element type.
 * 	param Ref Reference type returned by `operator*` (e.g. `Val const&` for const trees).
 * 	param Ptr Pointer type returned by `operator->`.
 */
template <typename Val, typename Ref, typename Ptr>
class TreeIterator : protected BaseIterator
{
public:
    using value_type = Val;
    using reference = Ref;
    using pointer = Ptr;
    using iterator = TreeIterator<Val, Val&, Val*>;
    using const_iterator = TreeIterator<Val, Val const&, Val const*>;
    using Self = TreeIterator<Val, Ref, Ptr>;
    using LinkConstType = TreeNode<Val> const*;
    using iterator_category = std::bidirectional_iterator_tag;
    using iterator_concept = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;

    constexpr TreeIterator() noexcept = default;
    constexpr explicit TreeIterator(LinkConstType pNode) noexcept : BaseIterator(pNode) {}

    //! Converts a mutable \ref iterator to a \c const_iterator.
    template <typename R2, typename P2>
        requires(std::is_same_v<TreeIterator<Val, R2, P2>, iterator> &&
                 !std::is_same_v<Self, iterator>)
    constexpr TreeIterator(TreeIterator<Val, R2, P2> const& that) noexcept
        : BaseIterator(that.node())
    {
    }

    //! Underlying tree node pointer (e.g. for \c KDTree::erase).
    [[nodiscard]] LinkConstType node() const noexcept
    {
        return static_cast<LinkConstType>(mNode);
    }

    [[nodiscard]] reference operator*() const noexcept { return node()->mValue; }

    [[nodiscard]] pointer operator->() const noexcept { return std::addressof(node()->mValue); }

    Self& operator++() noexcept
    {
        Increment();
        return *this;
    }

    Self operator++(int) noexcept
    {
        Self ret = *this;
        Increment();
        return ret;
    }

    Self& operator--() noexcept
    {
        Decrement();
        return *this;
    }

    Self operator--(int) noexcept
    {
        Self ret = *this;
        Decrement();
        return ret;
    }

    //! Equality (C++20 synthesizes `!=`; mixed const/mutable compares via the converting constructor).
    [[nodiscard]] friend bool operator==(Self const& refX, Self const& refY) noexcept
    {
        return refX.mNode == refY.mNode;
    }
};

static_assert(std::bidirectional_iterator<TreeIterator<int, int const&, int const*>>);

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
