/** \file
 * \brief Node allocation helpers for \ref KDTree (RAII guard for exception safety).
 * \ingroup libkdtree
 * \author Martin F. Krafft <libkdtree@pobox.madduck.net>
 */

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "node.hpp"

namespace libkdtree
{

/**
 * \brief Allocates / constructs / destroys \ref TreeNode "TreeNode<Tp>" via \c std::allocator_traits.
 *
 * \p Alloc may be an allocator for either \c Tp or \c TreeNode<Tp>; it is rebound to the node type.
 */
template <typename Tp, typename Alloc>
class AllocBase
{
public:
    using NodeType = TreeNode<Tp>;
    using BasePtr = typename NodeType::BasePtr;
    using allocator_type =
        typename std::allocator_traits<Alloc>::template rebind_alloc<NodeType>;
    using AllocTraits = std::allocator_traits<allocator_type>;

    explicit AllocBase(allocator_type const& refAlloc) noexcept : mNodeAlloc(refAlloc) {}
    explicit AllocBase(allocator_type&& refAlloc) noexcept : mNodeAlloc(std::move(refAlloc)) {}

    [[nodiscard]] allocator_type getAllocator() const noexcept { return mNodeAlloc; }

protected:
    KDTREE_NO_UNIQUE_ADDRESS allocator_type mNodeAlloc;

    //! Allocates and constructs a node holding `Tp(args...)`. Strong exception guarantee.
    template <typename... Args>
    [[nodiscard]] NodeType* createNode(Args&&... args)
    {
        NodeType* const pNode = std::to_address(AllocTraits::allocate(mNodeAlloc, 1));
        try {
            AllocTraits::construct(mNodeAlloc, pNode, std::in_place, std::forward<Args>(args)...);
        } catch (...) {
            AllocTraits::deallocate(mNodeAlloc, pNode, 1);
            throw;
        }
        return pNode;
    }

    //! Destroys and frees a node created by \ref createNode.
    void deleteNode(NodeType* const pNode) noexcept
    {
        AllocTraits::destroy(mNodeAlloc, pNode);
        AllocTraits::deallocate(mNodeAlloc, pNode, 1);
    }
};

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
