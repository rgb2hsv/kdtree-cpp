/** \file
 * \brief \ref libkdtree::BoxTree "BoxTree": closest-point, signed-distance and ray queries over
 *        geometric primitives (segments, triangles, boxes, user types) in 2D or 3D.
 * \ingroup libkdtree
 *
 * The query API mirrors wedge's `ClosestPoint2D` / `ClosestPoint3D`
 * (`closestPoint`, `closestPointSqr`, `closestPointSignedDistance`, `closestPointOutside`,
 * `intersectRayDistance`, `intersectSegmentDistance`, `distance`, with the same optional
 * `lastPoint` / `lastPrimitive` out-parameters).
 *
 * Internally it is a bounding-volume hierarchy built with kd-tree style splits: each node
 * stores the tight box of its primitives, children are split at the centroid median
 * (\ref SplitMetric::Median, best for closest-point queries) or by a binned surface-area
 * heuristic (\ref SplitMetric::Weighted, best for rays). Nodes live in one flat array and all
 * queries are allocation-free depth-first traversals that visit the nearer child first.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "geometry.hpp"

namespace libkdtree
{
/** \addtogroup libkdtree
 *  @{
 */

// ---------------------------------------------------------------------------
// Built-in primitives
// ---------------------------------------------------------------------------

//! Line segment `a`–`b`. In 2D its normal points to the right of `a -> b`
//! (outward for counter-clockwise polygons).
template <std::size_t N>
struct Segment {
    Vec<N> a{}, b{};
};

//! Triangle `a`, `b`, `c`. In 2D it is a filled region (distance 0 inside);
//! in 3D its normal is `normalize(cross(b - a, c - a))` (outward for counter-clockwise faces).
template <std::size_t N>
struct Triangle {
    Vec<N> a{}, b{}, c{};
};


// ---------------------------------------------------------------------------
// Primitive traits (customisation point)
// ---------------------------------------------------------------------------

/**
 * \brief Customisation point describing a primitive to \ref BoxTree.
 *
 * Specialise for your own type. Required members:
 *  - `static constexpr std::size_t dimension;`
 *  - `static Box<dimension> bounds(Prim const&);`
 *  - `static Vec<dimension> closestPoint(Prim const&, Vec<dimension> const& query);`
 *
 * Optional members (enable extra queries):
 *  - `static Vec<dimension> normal(Prim const&, Vec<dimension> const& pointOnPrim);`
 *    enables `closestPointSignedDistance`.
 *  - `static std::optional<double> intersectRay(Prim const&, Vec<dimension> const& org, Vec<dimension> const& dir);`
 *    returns the smallest `t >= 0` with `org + t*dir` on the primitive; enables
 *    `intersectRayDistance` / `intersectSegmentDistance`.
 */
template <class Prim>
struct PrimitiveTraits;

template <class Traits, class Prim>
concept BoxPrimitiveTraits = requires(Prim const& p, Vec<Traits::dimension> const& q) {
    { Traits::dimension } -> std::convertible_to<std::size_t>;
    { Traits::bounds(p) } -> std::convertible_to<Box<Traits::dimension>>;
    { Traits::closestPoint(p, q) } -> std::convertible_to<Vec<Traits::dimension>>;
};

template <class Traits, class Prim>
concept HasPrimitiveNormal = requires(Prim const& p, Vec<Traits::dimension> const& q) {
    { Traits::normal(p, q) } -> std::convertible_to<Vec<Traits::dimension>>;
};

template <class Traits, class Prim>
concept HasRayIntersection = requires(Prim const& p, Vec<Traits::dimension> const& o) {
    { Traits::intersectRay(p, o, o) } -> std::convertible_to<std::optional<double>>;
};

template <std::size_t N>
struct PrimitiveTraits<Segment<N>> {
    static constexpr std::size_t dimension = N;
    static Box<N> bounds(Segment<N> const& s) noexcept { return Box<N>::fromPoints(s.a, s.b); }
    static Vec<N> closestPoint(Segment<N> const& s, Vec<N> const& q) noexcept
    {
        return geom::closestPointOnSegment(s.a, s.b, q);
    }
    static Vec<N> normal(Segment<N> const& s, Vec<N> const&) noexcept
        requires(N == 2)
    {
        Vec<N> const e = geom::sub(s.b, s.a);
        return geom::normalize(Vec<N>{e[1], -e[0]});
    }
    static std::optional<double> intersectRay(Segment<N> const& s, Vec<N> const& o,
                                              Vec<N> const& d) noexcept
        requires(N == 2)
    {
        return geom::intersectRaySegment(s.a, s.b, o, d);
    }
};

template <std::size_t N>
struct PrimitiveTraits<Triangle<N>> {
    static constexpr std::size_t dimension = N;
    static Box<N> bounds(Triangle<N> const& t) noexcept
    {
        Box<N> b = Box<N>::fromPoints(t.a, t.b);
        return b.expand(t.c);
    }
    static Vec<N> closestPoint(Triangle<N> const& t, Vec<N> const& q) noexcept
    {
        if constexpr (N == 2) {
            if (geom::triangleContains(t.a, t.b, t.c, q)) return q;  // filled: exact 0 inside
        }
        return geom::closestPointOnTriangle(t.a, t.b, t.c, q);
    }
    static Vec<N> normal(Triangle<N> const& t, Vec<N> const&) noexcept
        requires(N == 3)
    {
        return geom::normalize(geom::cross(geom::sub(t.b, t.a), geom::sub(t.c, t.a)));
    }
    static std::optional<double> intersectRay(Triangle<N> const& t, Vec<N> const& o,
                                              Vec<N> const& d) noexcept
        requires(N == 2 || N == 3)
    {
        if constexpr (N == 3) {
            return geom::intersectRayTriangle(t.a, t.b, t.c, o, d);
        } else {
            // filled 2D triangle: 0 when starting inside, else the first edge crossing
            if (geom::triangleContains(t.a, t.b, t.c, o)) return 0.0;
            std::optional<double> best;
            for (auto const& [p, q] : {std::pair{t.a, t.b}, std::pair{t.b, t.c}, std::pair{t.c, t.a}})
                if (auto h = geom::intersectRaySegment(p, q, o, d); h && (!best || *h < *best))
                    best = h;
            return best;
        }
    }
};

//! A box as a primitive (solid: distance 0 inside).
template <std::size_t N>
struct PrimitiveTraits<Box<N>> {
    static constexpr std::size_t dimension = N;
    static Box<N> bounds(Box<N> const& b) noexcept { return b; }
    static Vec<N> closestPoint(Box<N> const& b, Vec<N> const& q) noexcept { return b.clamp(q); }
    static std::optional<double> intersectRay(Box<N> const& b, Vec<N> const& o, Vec<N> const& d) noexcept
    {
        auto const h = b.intersectRay(o, d, 0.0, std::numeric_limits<double>::infinity());
        return h ? std::optional<double>(h->first) : std::nullopt;
    }
};

//! A `std::variant` of primitives of one dimension (e.g. mixed segments and triangles).
//! Normal / ray support is available when every alternative provides it.
template <class P0, class... Ps>
struct PrimitiveTraits<std::variant<P0, Ps...>> {
    using Variant = std::variant<P0, Ps...>;
    static constexpr std::size_t dimension = PrimitiveTraits<P0>::dimension;
    static_assert(((PrimitiveTraits<Ps>::dimension == dimension) && ...),
                  "all variant alternatives must have the same dimension");
    using V = Vec<dimension>;

    static Box<dimension> bounds(Variant const& v)
    {
        return std::visit([](auto const& p) -> Box<dimension> { return traitsOf(p).bounds(p); }, v);
    }
    static V closestPoint(Variant const& v, V const& q)
    {
        return std::visit([&q](auto const& p) -> V { return traitsOf(p).closestPoint(p, q); }, v);
    }
    static V normal(Variant const& v, V const& q)
        requires(HasPrimitiveNormal<PrimitiveTraits<P0>, P0> &&
                 (HasPrimitiveNormal<PrimitiveTraits<Ps>, Ps> && ...))
    {
        return std::visit([&q](auto const& p) -> V { return traitsOf(p).normal(p, q); }, v);
    }
    static std::optional<double> intersectRay(Variant const& v, V const& o, V const& d)
        requires(HasRayIntersection<PrimitiveTraits<P0>, P0> &&
                 (HasRayIntersection<PrimitiveTraits<Ps>, Ps> && ...))
    {
        return std::visit(
            [&](auto const& p) -> std::optional<double> { return traitsOf(p).intersectRay(p, o, d); }, v);
    }

private:
    template <class P>
    static PrimitiveTraits<P> traitsOf(P const&)
    {
        return {};
    }
};

// ---------------------------------------------------------------------------
// BoxTree
// ---------------------------------------------------------------------------

//! How \ref BoxTree chooses split positions.
enum class SplitMetric {
    Median,   //!< centroid median on the longest axis; balanced, best for closest-point queries
    Weighted  //!< binned surface-area heuristic (SAH); best for ray / segment queries
};

/**
 * \brief Spatial index over primitives with bounding boxes, for closest-point, signed-distance
 *        and ray queries in 2D or 3D.
 *
 * Primitives are copied into the tree and reordered for locality; pointers returned by queries
 * point into the tree and stay valid until the next \ref build / \ref finish / \ref reset (or a
 * copy / move of the tree). \ref index maps such a pointer back to its insertion order.
 *
 * Const queries keep no mutable state, so a built tree can be queried from many threads.
 *
 * \tparam Prim   Primitive type, e.g. \ref Segment, \ref Triangle, \ref Box, a `std::variant`
 *                of those, or any type with a \ref PrimitiveTraits specialisation.
 * \tparam Traits Geometry of \p Prim; defaults to `PrimitiveTraits<Prim>`.
 */
template <class Prim, class Traits = PrimitiveTraits<Prim>>
class BoxTree
{
    static_assert(BoxPrimitiveTraits<Traits, Prim>,
                  "Traits must provide dimension, bounds(prim) and closestPoint(prim, point)");

public:
    static constexpr std::size_t N = Traits::dimension;
    static constexpr bool kHasNormal = HasPrimitiveNormal<Traits, Prim>;
    static constexpr bool kHasRay = HasRayIntersection<Traits, Prim>;
    //! Hard cap on tree depth (keeps the traversal stack fixed-size).
    static constexpr std::size_t kMaxDepthLimit = 64;
    //! Leaves hold at most this many primitives unless \ref maxDepth forces a larger leaf.
    static constexpr std::size_t kLeafSize = 4;

    using primitive_type = Prim;
    using point_type = Vec<N>;
    using box_type = Box<N>;
    using size_type = std::size_t;
    using SplitMetric = libkdtree::SplitMetric;

    //! Result of a closest-point query.
    struct Hit {
        double distanceSqr;     //!< squared distance from the query point
        point_type point;       //!< closest point on \ref primitive
        Prim const* primitive;  //!< points into the tree
    };

    //! Result of a ray / segment query.
    struct RayHit {
        double t;               //!< ray parameter: hit point = `org + t * dir`
        point_type point;       //!< hit point
        Prim const* primitive;  //!< points into the tree
    };

    // --- construction -------------------------------------------------------

    BoxTree() = default;

    explicit BoxTree(SplitMetric metric, size_type maxDepth = kMaxDepthLimit)
        : mSplitMetric(metric), mMaxDepth(std::min(maxDepth, kMaxDepthLimit))
    {
    }

    //! Builds the tree from \p prims.
    explicit BoxTree(std::vector<Prim> prims, size_type maxDepth = kMaxDepthLimit,
                     SplitMetric metric = SplitMetric::Median)
        : mSplitMetric(metric), mMaxDepth(std::min(maxDepth, kMaxDepthLimit))
    {
        build(std::move(prims));
    }

    //! Replaces the contents with \p prims and builds the tree.
    void build(std::vector<Prim> prims)
    {
        mPrims = std::move(prims);
        mPending.clear();
        rebuild();
    }

    void build(std::vector<Prim> prims, SplitMetric metric, size_type maxDepth = kMaxDepthLimit)
    {
        mSplitMetric = metric;
        setMaxDepth(maxDepth);
        build(std::move(prims));
    }

    //! Queues a primitive; call \ref finish to (re)build. Queries throw until then.
    void append(Prim prim) { mPending.push_back(std::move(prim)); }

    template <class... Args>
    void emplace(Args&&... args)
    {
        mPending.emplace_back(std::forward<Args>(args)...);
    }

    //! Builds the tree from all primitives, including those added by \ref append.
    void finish()
    {
        if (mPending.empty() && !mNodes.empty()) return;
        std::vector<Prim> all = takeInOriginalOrder();
        all.reserve(all.size() + mPending.size());
        for (Prim& p : mPending) all.push_back(std::move(p));
        mPending.clear();
        mPrims = std::move(all);
        rebuild();
    }

    //! Removes everything.
    void reset() noexcept
    {
        mPrims.clear();
        mPending.clear();
        mNodes.clear();
        mOriginalIndex.clear();
        mPosition.clear();
    }

    // --- observers ----------------------------------------------------------

    [[nodiscard]] size_type size() const noexcept { return mPrims.size(); }
    [[nodiscard]] bool empty() const noexcept { return mPrims.empty(); }
    [[nodiscard]] SplitMetric getSplitMetric() const noexcept { return mSplitMetric; }
    //! Takes effect on the next build.
    void setSplitMetric(SplitMetric metric) noexcept { mSplitMetric = metric; }
    [[nodiscard]] size_type maxDepth() const noexcept { return mMaxDepth; }
    //! Takes effect on the next build; clamped to \ref kMaxDepthLimit.
    void setMaxDepth(size_type depth) noexcept { mMaxDepth = std::min(depth, kMaxDepthLimit); }

    //! Bounding box of all primitives (empty box when the tree is empty).
    [[nodiscard]] box_type bounds() const noexcept { return mNodes.empty() ? box_type{} : mNodes[0].bounds; }
    [[nodiscard]] size_type nodeCount() const noexcept { return mNodes.size(); }
    //! Number of node levels (0 when empty).
    [[nodiscard]] size_type depth() const noexcept { return mDepth; }

    //! Primitives in tree (storage) order.
    [[nodiscard]] std::span<Prim const> primitives() const noexcept { return mPrims; }
    //! Insertion index of a primitive returned by a query.
    [[nodiscard]] size_type index(Prim const* prim) const noexcept
    {
        return mOriginalIndex[static_cast<size_type>(prim - mPrims.data())];
    }
    //! Primitive by insertion index.
    [[nodiscard]] Prim const& primitive(size_type insertionIndex) const noexcept
    {
        return mPrims[mPosition[insertionIndex]];
    }

    // --- modern query API ---------------------------------------------------

    //! Closest primitive to \p pt within \p maxDistance (inclusive).
    [[nodiscard]] std::optional<Hit> nearest(
        point_type const& pt, double maxDistance = std::numeric_limits<double>::infinity()) const
    {
        return nearestImpl(pt, sqr(maxDistance), [](Prim const&, point_type const&) { return true; });
    }

    //! Every primitive within \p maxDistance (inclusive) of \p pt, in no particular order.
    [[nodiscard]] std::vector<Hit> within(point_type const& pt, double maxDistance) const
    {
        std::vector<Hit> out;
        visitWithin(pt, maxDistance, [&out](Hit const& h) { out.push_back(h); });
        return out;
    }

    //! Calls \p fn(Hit const&) for every primitive within \p maxDistance (inclusive) of \p pt.
    template <class Fn>
    void visitWithin(point_type const& pt, double maxDistance, Fn&& fn) const
    {
        checkBuilt();
        if (mNodes.empty()) return;
        double const maxSq = sqr(maxDistance);
        NodeEntry stack[kStackSize];
        size_type top = 0;
        stack[top++] = {0, mNodes[0].bounds.distanceSqr(pt)};
        while (top) {
            NodeEntry const e = stack[--top];
            if (e.key > maxSq) continue;
            Node const& n = mNodes[e.node];
            if (n.count) {
                for (std::uint32_t i = n.first; i < n.first + n.count; ++i) {
                    point_type const cp = Traits::closestPoint(mPrims[i], pt);
                    double const d2 = geom::distanceSqr(cp, pt);
                    if (d2 <= maxSq) fn(Hit{d2, cp, &mPrims[i]});
                }
            } else {
                for (std::uint32_t c = n.first; c < n.first + 2; ++c) {
                    double const d = mNodes[c].bounds.distanceSqr(pt);
                    if (d <= maxSq) stack[top++] = {c, d};
                }
            }
        }
    }

    //! First hit of the ray `org + t * dir`, `t` in `[0, tMax]`.
    [[nodiscard]] std::optional<RayHit> raycast(
        point_type const& org, point_type const& dir,
        double tMax = std::numeric_limits<double>::infinity()) const
        requires kHasRay
    {
        checkBuilt();
        if (mNodes.empty()) return std::nullopt;
        auto const rootHit = mNodes[0].bounds.intersectRay(org, dir, 0.0, tMax);
        if (!rootHit) return std::nullopt;

        std::optional<RayHit> best;
        double bestT = tMax;
        NodeEntry stack[kStackSize];
        size_type top = 0;
        stack[top++] = {0, rootHit->first};
        while (top) {
            NodeEntry const e = stack[--top];
            if (e.key > bestT) continue;
            Node const& n = mNodes[e.node];
            if (n.count) {
                for (std::uint32_t i = n.first; i < n.first + n.count; ++i) {
                    std::optional<double> const t = Traits::intersectRay(mPrims[i], org, dir);
                    if (t && *t >= 0 && (*t < bestT || (!best && *t <= bestT))) {
                        bestT = *t;
                        best = RayHit{*t, geom::madd(org, dir, *t), &mPrims[i]};
                    }
                }
            } else {
                auto const hl = mNodes[n.first].bounds.intersectRay(org, dir, 0.0, bestT);
                auto const hr = mNodes[n.first + 1].bounds.intersectRay(org, dir, 0.0, bestT);
                pushOrdered(stack, top, n.first, hl ? std::optional(hl->first) : std::nullopt,
                            n.first + 1, hr ? std::optional(hr->first) : std::nullopt);
            }
        }
        return best;
    }

    // --- ClosestPoint2D / ClosestPoint3D compatible API ----------------------
    // Return the (squared) distance, or nullopt when nothing qualifies (or the tree is empty).
    // On a miss `lastPrimitive` is set to nullptr and `lastPoint` is left untouched.

    std::optional<double> closestPointSqr(point_type const& pt, point_type& lastPoint,
                                          Prim const*& lastPrimitive) const
    {
        return report(nearest(pt), lastPoint, lastPrimitive, false);
    }
    std::optional<double> closestPointSqr(point_type const& pt, double maxDistance,
                                          point_type& lastPoint, Prim const*& lastPrimitive) const
    {
        return report(nearest(pt, maxDistance), lastPoint, lastPrimitive, false);
    }
    //! All primitives within \p maxDistance go to \p lastPrimitives; \p lastPoint is the closest point.
    std::optional<double> closestPointSqr(point_type const& pt, double maxDistance,
                                          point_type& lastPoint,
                                          std::vector<Prim const*>& lastPrimitives) const
    {
        lastPrimitives.clear();
        std::optional<Hit> best;
        visitWithin(pt, maxDistance, [&](Hit const& h) {
            lastPrimitives.push_back(h.primitive);
            if (!best || h.distanceSqr < best->distanceSqr) best = h;
        });
        if (!best) return std::nullopt;
        lastPoint = best->point;
        return best->distanceSqr;
    }
    std::optional<double> closestPointSqr(point_type const& pt, point_type& lastPoint) const
    {
        Prim const* prim;
        return closestPointSqr(pt, lastPoint, prim);
    }
    std::optional<double> closestPointSqr(point_type const& pt, double maxDistance,
                                          point_type& lastPoint) const
    {
        Prim const* prim;
        return closestPointSqr(pt, maxDistance, lastPoint, prim);
    }
    std::optional<double> closestPointSqr(point_type const& pt) const
    {
        auto const h = nearest(pt);
        return h ? std::optional(h->distanceSqr) : std::nullopt;
    }
    std::optional<double> closestPointSqr(point_type const& pt, double maxDistance) const
    {
        auto const h = nearest(pt, maxDistance);
        return h ? std::optional(h->distanceSqr) : std::nullopt;
    }

    std::optional<double> closestPoint(point_type const& pt, point_type& lastPoint,
                                       Prim const*& lastPrimitive) const
    {
        return report(nearest(pt), lastPoint, lastPrimitive, true);
    }
    std::optional<double> closestPoint(point_type const& pt, double maxDistance,
                                       point_type& lastPoint, Prim const*& lastPrimitive) const
    {
        return report(nearest(pt, maxDistance), lastPoint, lastPrimitive, true);
    }
    std::optional<double> closestPoint(point_type const& pt, double maxDistance,
                                       point_type& lastPoint,
                                       std::vector<Prim const*>& lastPrimitives) const
    {
        auto const d2 = closestPointSqr(pt, maxDistance, lastPoint, lastPrimitives);
        return d2 ? std::optional(std::sqrt(*d2)) : std::nullopt;
    }
    std::optional<double> closestPoint(point_type const& pt, point_type& lastPoint) const
    {
        Prim const* prim;
        return closestPoint(pt, lastPoint, prim);
    }
    std::optional<double> closestPoint(point_type const& pt, double maxDistance,
                                       point_type& lastPoint) const
    {
        Prim const* prim;
        return closestPoint(pt, maxDistance, lastPoint, prim);
    }
    std::optional<double> closestPoint(point_type const& pt, Prim const*& lastPrimitive) const
    {
        point_type p;
        return closestPoint(pt, p, lastPrimitive);
    }
    std::optional<double> closestPoint(point_type const& pt) const
    {
        auto const d2 = closestPointSqr(pt);
        return d2 ? std::optional(std::sqrt(*d2)) : std::nullopt;
    }
    std::optional<double> closestPoint(point_type const& pt, double maxDistance) const
    {
        auto const d2 = closestPointSqr(pt, maxDistance);
        return d2 ? std::optional(std::sqrt(*d2)) : std::nullopt;
    }

    //! Distance to the closest primitive; +infinity when the tree is empty.
    [[nodiscard]] double distance(point_type const& pt) const
    {
        return closestPoint(pt).value_or(std::numeric_limits<double>::infinity());
    }
    //! Squared distance to the closest primitive; +infinity when the tree is empty.
    [[nodiscard]] double distanceSqr(point_type const& pt) const
    {
        return closestPointSqr(pt).value_or(std::numeric_limits<double>::infinity());
    }
    //! Distance, clamped to \p threshold (returned when nothing is closer).
    [[nodiscard]] double distance(point_type const& pt, double threshold) const
    {
        return closestPoint(pt, threshold).value_or(threshold);
    }
    //! Squared distance, clamped to `threshold^2` (returned when nothing is closer).
    [[nodiscard]] double distanceSqr(point_type const& pt, double threshold) const
    {
        return closestPointSqr(pt, threshold).value_or(threshold * threshold);
    }

    /**
     * \brief Distance to the closest primitive, signed by the primitive's normal at the closest
     *        point: positive on the side the normal points to, negative behind, 0 on the surface.
     */
    std::optional<double> closestPointSignedDistance(point_type const& r, point_type& lastPoint,
                                                     Prim const*& lastPrimitive) const
        requires kHasNormal
    {
        auto const d = closestPoint(r, lastPoint, lastPrimitive);  // must run before reading the out-params
        return signedFrom(r, d, lastPoint, lastPrimitive);
    }
    std::optional<double> closestPointSignedDistance(point_type const& r, double maxDistance,
                                                     point_type& lastPoint,
                                                     Prim const*& lastPrimitive) const
        requires kHasNormal
    {
        auto const d = closestPoint(r, maxDistance, lastPoint, lastPrimitive);
        return signedFrom(r, d, lastPoint, lastPrimitive);
    }
    std::optional<double> closestPointSignedDistance(point_type const& r, point_type& lastPoint) const
        requires kHasNormal
    {
        Prim const* prim;
        return closestPointSignedDistance(r, lastPoint, prim);
    }
    std::optional<double> closestPointSignedDistance(point_type const& r,
                                                     Prim const*& lastPrimitive) const
        requires kHasNormal
    {
        point_type p;
        return closestPointSignedDistance(r, p, lastPrimitive);
    }
    std::optional<double> closestPointSignedDistance(point_type const& r) const
        requires kHasNormal
    {
        point_type p;
        Prim const* prim;
        return closestPointSignedDistance(r, p, prim);
    }
    std::optional<double> closestPointSignedDistance(point_type const& r, double maxDistance) const
        requires kHasNormal
    {
        point_type p;
        Prim const* prim;
        return closestPointSignedDistance(r, maxDistance, p, prim);
    }

    /**
     * \brief Closest primitive whose closest point lies in the half-space `dot(p - r, v) >= 0`
     *        (in front of the plane through \p r with normal \p v). Returns the distance.
     */
    std::optional<double> closestPointOutside(point_type const& r, point_type const& v,
                                              point_type& lastPoint, Prim const*& lastPrimitive) const
    {
        auto const h = nearestImpl(r, std::numeric_limits<double>::infinity(),
                                   [&](Prim const&, point_type const& cp) {
                                       return geom::dot(geom::sub(cp, r), v) >= 0;
                                   });
        return report(h, lastPoint, lastPrimitive, true);
    }
    std::optional<double> closestPointOutside(point_type const& r, point_type const& v,
                                              point_type& lastPoint) const
    {
        Prim const* prim;
        return closestPointOutside(r, v, lastPoint, prim);
    }
    std::optional<double> closestPointOutside(point_type const& r, point_type const& v,
                                              Prim const*& lastPrimitive) const
    {
        point_type p;
        return closestPointOutside(r, v, p, lastPrimitive);
    }
    std::optional<double> closestPointOutside(point_type const& r, point_type const& v) const
    {
        point_type p;
        Prim const* prim;
        return closestPointOutside(r, v, p, prim);
    }

    //! Distance from \p p1 along direction \p v to the first primitive hit (nullopt on a miss).
    std::optional<double> intersectRayDistance(point_type const& p1, point_type const& v,
                                               point_type& lastPoint, Prim const*& lastPrimitive) const
        requires kHasRay
    {
        return reportRay(p1, v, std::numeric_limits<double>::infinity(), lastPoint, lastPrimitive);
    }
    std::optional<double> intersectRayDistance(point_type const& p1, point_type const& v,
                                               point_type& lastPoint) const
        requires kHasRay
    {
        Prim const* prim;
        return intersectRayDistance(p1, v, lastPoint, prim);
    }
    std::optional<double> intersectRayDistance(point_type const& p1, point_type const& v,
                                               Prim const*& lastPrimitive) const
        requires kHasRay
    {
        point_type p;
        return intersectRayDistance(p1, v, p, lastPrimitive);
    }
    std::optional<double> intersectRayDistance(point_type const& p1, point_type const& v) const
        requires kHasRay
    {
        point_type p;
        Prim const* prim;
        return intersectRayDistance(p1, v, p, prim);
    }

    //! Distance from \p p1 to the first primitive hit on the segment `p1`–`p2` (nullopt on a miss).
    std::optional<double> intersectSegmentDistance(point_type const& p1, point_type const& p2,
                                                   point_type& lastPoint,
                                                   Prim const*& lastPrimitive) const
        requires kHasRay
    {
        return reportRay(p1, geom::sub(p2, p1), 1.0, lastPoint, lastPrimitive);
    }
    std::optional<double> intersectSegmentDistance(point_type const& p1, point_type const& p2,
                                                   point_type& lastPoint) const
        requires kHasRay
    {
        Prim const* prim;
        return intersectSegmentDistance(p1, p2, lastPoint, prim);
    }
    std::optional<double> intersectSegmentDistance(point_type const& p1, point_type const& p2,
                                                   Prim const*& lastPrimitive) const
        requires kHasRay
    {
        point_type p;
        return intersectSegmentDistance(p1, p2, p, lastPrimitive);
    }
    std::optional<double> intersectSegmentDistance(point_type const& p1, point_type const& p2) const
        requires kHasRay
    {
        point_type p;
        Prim const* prim;
        return intersectSegmentDistance(p1, p2, p, prim);
    }

private:
    struct Node {
        box_type bounds;
        std::uint32_t first = 0;  //!< leaf: first primitive; internal: left child (right = first + 1)
        std::uint32_t count = 0;  //!< leaf: number of primitives; 0 for internal nodes
    };

    struct NodeEntry {
        std::uint32_t node;
        double key;  //!< squared box distance (closest point) or entry t (rays)
    };

    // Depth-first traversal leaves at most one pending sibling per level.
    static constexpr size_type kStackSize = 2 * kMaxDepthLimit + 4;

    struct BuildRef {
        box_type box;
        point_type centroid;
        std::uint32_t prim;
    };

    static constexpr double sqr(double v) noexcept { return v * v; }

    void checkBuilt() const
    {
        if (!mPending.empty())
            throw std::logic_error("BoxTree: call finish() after append() before querying");
    }

    //! Pushes the farther child first so the nearer one is popped next.
    static void pushOrdered(NodeEntry* stack, size_type& top, std::uint32_t a, std::optional<double> ka,
                            std::uint32_t b, std::optional<double> kb) noexcept
    {
        if (ka && kb) {
            if (*ka < *kb) {
                stack[top++] = {b, *kb};
                stack[top++] = {a, *ka};
            } else {
                stack[top++] = {a, *ka};
                stack[top++] = {b, *kb};
            }
        } else if (ka) {
            stack[top++] = {a, *ka};
        } else if (kb) {
            stack[top++] = {b, *kb};
        }
    }

    template <class Accept>
    std::optional<Hit> nearestImpl(point_type const& pt, double boundSq, Accept&& accept) const
    {
        checkBuilt();
        if (mNodes.empty()) return std::nullopt;
        double bestSq = boundSq;
        std::optional<Hit> best;
        NodeEntry stack[kStackSize];
        size_type top = 0;
        stack[top++] = {0, mNodes[0].bounds.distanceSqr(pt)};
        while (top) {
            NodeEntry const e = stack[--top];
            if (e.key > bestSq) continue;
            Node const& n = mNodes[e.node];
            if (n.count) {
                for (std::uint32_t i = n.first; i < n.first + n.count; ++i) {
                    point_type const cp = Traits::closestPoint(mPrims[i], pt);
                    double const d2 = geom::distanceSqr(cp, pt);
                    // first hit may sit exactly on the bound; later hits must improve
                    if ((d2 < bestSq || (!best && d2 <= bestSq)) && accept(mPrims[i], cp)) {
                        bestSq = d2;
                        best = Hit{d2, cp, &mPrims[i]};
                    }
                }
            } else {
                double const dl = mNodes[n.first].bounds.distanceSqr(pt);
                double const dr = mNodes[n.first + 1].bounds.distanceSqr(pt);
                pushOrdered(stack, top, n.first, dl <= bestSq ? std::optional(dl) : std::nullopt,
                            n.first + 1, dr <= bestSq ? std::optional(dr) : std::nullopt);
            }
        }
        return best;
    }

    static std::optional<double> report(std::optional<Hit> const& h, point_type& lastPoint,
                                        Prim const*& lastPrimitive, bool takeSqrt)
    {
        if (!h) {
            lastPrimitive = nullptr;
            return std::nullopt;
        }
        lastPoint = h->point;
        lastPrimitive = h->primitive;
        return takeSqrt ? std::sqrt(h->distanceSqr) : h->distanceSqr;
    }

    std::optional<double> signedFrom(point_type const& r, std::optional<double> d,
                                     point_type const& lastPoint, Prim const* lastPrimitive) const
    {
        if (!d) return std::nullopt;
        double const s = geom::dot(geom::sub(r, lastPoint), Traits::normal(*lastPrimitive, lastPoint));
        return (s > 0) ? *d : (s < 0) ? -*d : 0.0;
    }

    std::optional<double> reportRay(point_type const& org, point_type const& dir, double tMax,
                                    point_type& lastPoint, Prim const*& lastPrimitive) const
        requires kHasRay
    {
        double const len = geom::length(dir);
        auto const h = len > 0 ? raycast(org, dir, tMax) : std::nullopt;
        if (!h) {
            lastPrimitive = nullptr;
            return std::nullopt;
        }
        lastPoint = h->point;
        lastPrimitive = h->primitive;
        return h->t * len;
    }

    //! Moves the primitives out in insertion order (inverse of the build permutation).
    std::vector<Prim> takeInOriginalOrder()
    {
        std::vector<Prim> out;
        out.reserve(mPrims.size());
        for (size_type i = 0; i < mPrims.size(); ++i) out.push_back(std::move(mPrims[mPosition[i]]));
        mPrims.clear();
        return out;
    }

    void rebuild()
    {
        mNodes.clear();
        mDepth = 0;
        size_type const n = mPrims.size();
        mOriginalIndex.resize(n);
        mPosition.resize(n);
        if (n == 0) return;
        if (n > std::numeric_limits<std::uint32_t>::max() / 2)
            throw std::length_error("BoxTree: too many primitives");

        std::vector<BuildRef> refs(n);
        for (size_type i = 0; i < n; ++i) {
            refs[i].box = Traits::bounds(mPrims[i]);
            refs[i].centroid = refs[i].box.center();
            refs[i].prim = static_cast<std::uint32_t>(i);
        }
        mNodes.reserve(2 * n);
        mNodes.emplace_back();
        buildNode(0, refs, 0, n, 1);

        // store primitives in leaf order for locality
        std::vector<Prim> ordered;
        ordered.reserve(n);
        for (size_type i = 0; i < n; ++i) {
            ordered.push_back(std::move(mPrims[refs[i].prim]));
            mOriginalIndex[i] = refs[i].prim;
            mPosition[refs[i].prim] = i;
        }
        mPrims = std::move(ordered);
    }

    // Recursion depth is bounded by mMaxDepth (<= kMaxDepthLimit).
    void buildNode(std::uint32_t nodeIdx, std::vector<BuildRef>& refs, size_type first, size_type last,
                   size_type level)
    {
        mDepth = std::max(mDepth, level);
        box_type bounds;
        box_type centroidBounds;
        for (size_type i = first; i < last; ++i) {
            bounds.expand(refs[i].box);
            centroidBounds.expand(refs[i].centroid);
        }
        mNodes[nodeIdx].bounds = bounds;

        size_type const count = last - first;
        std::size_t const axis = centroidBounds.longestAxis();
        bool const flatCentroids = !(centroidBounds.hi[axis] > centroidBounds.lo[axis]);
        if (count <= kLeafSize || level >= mMaxDepth || flatCentroids) {
            makeLeaf(nodeIdx, first, count);
            return;
        }

        size_type mid = first;
        if (mSplitMetric == SplitMetric::Weighted) mid = sahSplit(refs, first, last, bounds, centroidBounds);
        if (mid == first || mid == last) {  // median (or SAH fallback)
            mid = first + count / 2;
            std::nth_element(refs.begin() + static_cast<std::ptrdiff_t>(first),
                             refs.begin() + static_cast<std::ptrdiff_t>(mid),
                             refs.begin() + static_cast<std::ptrdiff_t>(last),
                             [axis](BuildRef const& a, BuildRef const& b) {
                                 return a.centroid[axis] < b.centroid[axis];
                             });
        }

        auto const left = static_cast<std::uint32_t>(mNodes.size());
        mNodes.emplace_back();
        mNodes.emplace_back();
        mNodes[nodeIdx].first = left;
        mNodes[nodeIdx].count = 0;
        buildNode(left, refs, first, mid, level + 1);
        buildNode(left + 1, refs, mid, last, level + 1);
    }

    void makeLeaf(std::uint32_t nodeIdx, size_type first, size_type count) noexcept
    {
        mNodes[nodeIdx].first = static_cast<std::uint32_t>(first);
        mNodes[nodeIdx].count = static_cast<std::uint32_t>(count);
    }

    //! Binned SAH over every axis. Returns the partition point, or `first` if no useful split.
    size_type sahSplit(std::vector<BuildRef>& refs, size_type first, size_type last,
                       box_type const& bounds, box_type const& centroidBounds) const
    {
        constexpr std::size_t kBins = 16;
        double const parentArea = bounds.surfaceMeasure();
        if (!(parentArea > 0)) return first;

        double bestCost = std::numeric_limits<double>::infinity();
        std::size_t bestAxis = 0;
        std::size_t bestBin = 0;
        for (std::size_t axis = 0; axis < N; ++axis) {
            double const lo = centroidBounds.lo[axis];
            double const ext = centroidBounds.hi[axis] - lo;
            if (!(ext > 0)) continue;
            double const scale = kBins / ext;

            std::array<box_type, kBins> binBox{};
            std::array<size_type, kBins> binCount{};
            for (size_type i = first; i < last; ++i) {
                auto const b = std::min(kBins - 1, static_cast<std::size_t>((refs[i].centroid[axis] - lo) * scale));
                binBox[b].expand(refs[i].box);
                ++binCount[b];
            }
            // sweep: cost of splitting after bin k = A_L * N_L + A_R * N_R
            std::array<double, kBins> rightCost{};
            box_type acc;
            size_type cnt = 0;
            for (std::size_t k = kBins - 1; k > 0; --k) {
                acc.expand(binBox[k]);
                cnt += binCount[k];
                rightCost[k - 1] = acc.surfaceMeasure() * static_cast<double>(cnt);
            }
            acc = box_type{};
            cnt = 0;
            for (std::size_t k = 0; k + 1 < kBins; ++k) {
                acc.expand(binBox[k]);
                cnt += binCount[k];
                if (cnt == 0 || cnt == last - first) continue;
                double const cost = acc.surfaceMeasure() * static_cast<double>(cnt) + rightCost[k];
                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestBin = k;
                }
            }
        }
        if (!(bestCost < std::numeric_limits<double>::infinity())) return first;

        double const lo = centroidBounds.lo[bestAxis];
        double const scale = kBins / (centroidBounds.hi[bestAxis] - lo);
        auto const it = std::partition(
            refs.begin() + static_cast<std::ptrdiff_t>(first), refs.begin() + static_cast<std::ptrdiff_t>(last),
            [&](BuildRef const& r) {
                return std::min(kBins - 1, static_cast<std::size_t>((r.centroid[bestAxis] - lo) * scale)) <= bestBin;
            });
        return static_cast<size_type>(it - refs.begin());
    }

    std::vector<Prim> mPrims;                 //!< tree (leaf) order
    std::vector<Prim> mPending;               //!< appended, not yet built
    std::vector<Node> mNodes;                 //!< mNodes[0] is the root
    std::vector<size_type> mOriginalIndex;    //!< tree position -> insertion index
    std::vector<size_type> mPosition;         //!< insertion index -> tree position
    SplitMetric mSplitMetric = SplitMetric::Median;
    size_type mMaxDepth = kMaxDepthLimit;
    size_type mDepth = 0;
};

//! 2D tree; defaults to line segments (mirrors wedge's `ClosestPoint2D`).
template <class Prim = Segment<2>>
using ClosestPoint2D = BoxTree<Prim>;

//! 3D tree; defaults to triangles (mirrors wedge's `ClosestPoint3D`).
template <class Prim = Triangle<3>>
using ClosestPoint3D = BoxTree<Prim>;

/** @} */

}  // namespace libkdtree
