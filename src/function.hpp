/** \file
 * \brief Default accessor and distance functors for \ref KDTree.
 * \ingroup libkdtree
 *
 * \author Martin F. Krafft <libkdtree@pobox.madduck.net>
 * \author Sylvain Bougerel <sylvain.bougerel.devel@gmail.com>
 */

#pragma once

#include <cstddef>

//! Portable `[[no_unique_address]]` (MSVC only honours its vendor attribute).
#if defined(_MSC_VER) && !defined(__clang__)
#define KDTREE_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define KDTREE_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

namespace libkdtree
{

//! Default \c Acc: reads `refVal[uIdx]`; requires `Val::value_type` and index operator.
template <typename Val>
struct BracketAccessor {
    using result_type = typename Val::value_type;

    [[nodiscard]] constexpr result_type operator()(Val const& refVal, std::size_t uIdx) const
        noexcept(noexcept(refVal[uIdx]))
    {
        return refVal[uIdx];
    }
};

//! Predicate that accepts every \c Tp; used as default for unfiltered nearest-neighbor search.
template <typename Tp>
struct AlwaysTrue {
    [[nodiscard]] constexpr bool operator()(const Tp&) const noexcept { return true; }
};

/**
 * \brief Per-axis squared difference `(refA - refB)^2`; summed over dimensions for Euclidean-style distance.
 */
template <typename Tp, typename Dist>
struct SquaredDifference {
    using distance_type = Dist;

    [[nodiscard]] constexpr distance_type operator()(Tp const& refA, Tp const& refB) const
    {
        distance_type const distElem = refA - refB;
        return distElem * distElem;
    }
};

//! Like \ref SquaredDifference but increments a mutable call counter (profiling / tests).
template <typename Tp, typename Dist>
struct SquaredDifferenceCounted {
    using distance_type = Dist;

    void reset() noexcept { m_cCalls = 0; }

    //! Reference to the number of invocations of `operator()`.
    [[nodiscard]] long& count() const noexcept { return m_cCalls; }

    distance_type operator()(const Tp& refA, const Tp& refB) const
    {
        distance_type const distElem = refA - refB;
        ++m_cCalls;
        return distElem * distElem;
    }

private:
    mutable long m_cCalls = 0;
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
