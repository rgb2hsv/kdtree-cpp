/** \file
 * \brief Axis-aligned \ref KdTreeRegion for box intersection tests in range queries.
 * \ingroup libkdtree
 * \author Martin F. Krafft <libkdtree@pobox.madduck.net>
 */

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "function.hpp"

namespace libkdtree
{

/**
 * \brief Closed axis-aligned box in \f$K\f$-dimensional accessor space.
 *
 * Used by \ref KDTree::countWithinRange, \ref KDTree::findWithinRange, and
 * \ref KDTree::visitWithinRange. Bounds are stored per axis in \ref mSubLow / \ref mSubHigh
 * and compared with \ref mComparator on scalar coordinates from \ref mAccessor.
 *
 * \tparam K Number of dimensions.
 * \tparam Val Type passed to the accessor (typically same as tree \c value_type).
 * \tparam SubVal Scalar coordinate type (interval endpoints).
 * \tparam Acc Accessor functor (same role as on \ref KDTree).
 * \tparam Cmp Strict weak ordering on \p SubVal.
 */
template <std::size_t K, typename Val, typename SubVal, typename Acc, typename Cmp>
struct KdTreeRegion {
    using value_type = Val;
    using subvalue_type = SubVal;

    //! Region plus radius (used by \ref intersectsWith(const CenterPair&) const).
    using CenterPair = std::pair<KdTreeRegion, SubVal>;

    //! Unbounded initialization: accessor and comparator only; bounds must be set before use.
    explicit KdTreeRegion(Acc const& acc = Acc(), Cmp const& cmp = Cmp()) : mAccessor(acc), mComparator(cmp) {}

    //! Degenerate box: point \p v0 (low and high equal on every axis).
    template <typename V>
    KdTreeRegion(V const& v0, Acc const& acc = Acc(), Cmp const& cmp = Cmp())
        : mAccessor(acc), mComparator(cmp)
    {
        for (std::size_t i = 0; i != K; ++i) {
            mSubLow[i] = mSubHigh[i] = mAccessor(v0, i);
        }
    }

    //! Box centered at \p v0 with half-width \p r along each axis (`[coord - r, coord + r]`).
    template <typename V>
    KdTreeRegion(V const& v0, subvalue_type const& r, Acc const& acc = Acc(), Cmp const& cmp = Cmp())
        : mAccessor(acc), mComparator(cmp)
    {
        for (std::size_t i = 0; i != K; ++i) {
            subvalue_type const c = mAccessor(v0, i);
            mSubLow[i] = c - r;
            mSubHigh[i] = c + r;
        }
    }

    //! Intersection test against \p that.first after growing this box by \p that.second on every side.
    [[nodiscard]] bool intersectsWith(CenterPair const& that) const
    {
        for (std::size_t i = 0; i != K; ++i) {
            if (mComparator(that.first.mSubHigh[i], mSubLow[i] - that.second) ||
                mComparator(mSubHigh[i] + that.second, that.first.mSubLow[i]))
                return false;
        }
        return true;
    }

    //! Non-empty intersection with another axis-aligned box.
    [[nodiscard]] bool intersectsWith(KdTreeRegion const& that) const
    {
        for (std::size_t i = 0; i != K; ++i) {
            if (mComparator(that.mSubHigh[i], mSubLow[i]) || mComparator(mSubHigh[i], that.mSubLow[i]))
                return false;
        }
        return true;
    }

    //! \c true if every coordinate of \p v lies inside this box (inclusive bounds).
    [[nodiscard]] bool encloses(value_type const& v) const
    {
        for (std::size_t i = 0; i != K; ++i) {
            auto const c = mAccessor(v, i);
            if (mComparator(c, mSubLow[i]) || mComparator(mSubHigh[i], c)) return false;
        }
        return true;
    }

    //! Sets high bound on axis `level % K` from `mAccessor(v, level % K)`.
    KdTreeRegion& setHighBound(value_type const& v, std::size_t level)
    {
        mSubHigh[level % K] = mAccessor(v, level % K);
        return *this;
    }

    //! Sets low bound on axis `level % K` from `mAccessor(v, level % K)`.
    KdTreeRegion& setLowBound(value_type const& v, std::size_t level)
    {
        mSubLow[level % K] = mAccessor(v, level % K);
        return *this;
    }

    std::array<subvalue_type, K> mSubLow{}, mSubHigh{};  //!< Inclusive per-axis interval `[mSubLow[i], mSubHigh[i]]`.
    KDTREE_NO_UNIQUE_ADDRESS Acc mAccessor;               //!< Coordinate accessor.
    KDTREE_NO_UNIQUE_ADDRESS Cmp mComparator;             //!< Scalar comparator.
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
