/** \file
 * \brief Geometry utilities for \ref libkdtree::BoxTree "BoxTree": fixed-size vectors,
 *        axis-aligned boxes, rays, closest-point and ray-intersection routines.
 * \ingroup libkdtree
 *
 * Per-axis loops are unrolled at compile time with fold expressions (see
 * \ref libkdtree::geom::detail "detail"). Sums keep the left-to-right order of a plain loop,
 * so results are bit-identical to the straightforward implementation. The one exception is
 * the ray / box slab test (\ref libkdtree::Box::intersect "Box::intersect"), which stays a
 * fixed-trip-count loop because it measured faster that way.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace libkdtree
{
/** \addtogroup libkdtree
 *  @{
 */

//! Point / vector in \p N dimensions.
template <std::size_t N>
using Vec = std::array<double, N>;
using Vec2 = Vec<2>;
using Vec3 = Vec<3>;

//! Small vector helpers (named functions: operators on std::array would not be found by ADL).
namespace geom
{
//! Compile-time unrolling helpers: `f` receives `std::integral_constant<std::size_t, I>`.
namespace detail
{
template <std::size_t I>
using Axis = std::integral_constant<std::size_t, I>;

//! Calls `f(Axis<0>{}) ... f(Axis<N-1>{})` in order.
template <std::size_t N, class F>
constexpr void forEachAxis(F&& f)
{
    [&]<std::size_t... I>(std::index_sequence<I...>) { (f(Axis<I>{}), ...); }(std::make_index_sequence<N>{});
}

//! `Vec<N>{f(Axis<0>{}), ..., f(Axis<N-1>{})}`.
template <std::size_t N, class F>
[[nodiscard]] constexpr Vec<N> makeVec(F&& f)
{
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return Vec<N>{f(Axis<I>{})...};
    }(std::make_index_sequence<N>{});
}

//! `((f(0) + f(1)) + ...) + f(N-1)` – same summation order as a loop.
template <std::size_t N, class F>
[[nodiscard]] constexpr double sumAxes(F&& f)
{
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return (0.0 + ... + f(Axis<I>{}));
    }(std::make_index_sequence<N>{});
}

//! `f(0) && f(1) && ...` (short-circuits).
template <std::size_t N, class F>
[[nodiscard]] constexpr bool allAxes(F&& f)
{
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return (f(Axis<I>{}) && ...);
    }(std::make_index_sequence<N>{});
}

//! `f(0) || f(1) || ...` (short-circuits).
template <std::size_t N, class F>
[[nodiscard]] constexpr bool anyAxis(F&& f)
{
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return (f(Axis<I>{}) || ...);
    }(std::make_index_sequence<N>{});
}
}  // namespace detail

template <std::size_t N>
[[nodiscard]] constexpr Vec<N> add(Vec<N> const& a, Vec<N> const& b) noexcept
{
    return detail::makeVec<N>([&](auto i) { return a[i] + b[i]; });
}
template <std::size_t N>
[[nodiscard]] constexpr Vec<N> sub(Vec<N> const& a, Vec<N> const& b) noexcept
{
    return detail::makeVec<N>([&](auto i) { return a[i] - b[i]; });
}
template <std::size_t N>
[[nodiscard]] constexpr Vec<N> mul(Vec<N> const& a, double s) noexcept
{
    return detail::makeVec<N>([&](auto i) { return a[i] * s; });
}
//! `a + s * b`
template <std::size_t N>
[[nodiscard]] constexpr Vec<N> madd(Vec<N> const& a, Vec<N> const& b, double s) noexcept
{
    return detail::makeVec<N>([&](auto i) { return a[i] + b[i] * s; });
}
template <std::size_t N>
[[nodiscard]] constexpr double dot(Vec<N> const& a, Vec<N> const& b) noexcept
{
    return detail::sumAxes<N>([&](auto i) { return a[i] * b[i]; });
}
template <std::size_t N>
[[nodiscard]] constexpr double lengthSqr(Vec<N> const& a) noexcept
{
    return dot(a, a);
}
template <std::size_t N>
[[nodiscard]] inline double length(Vec<N> const& a) noexcept
{
    return std::sqrt(lengthSqr(a));
}
template <std::size_t N>
[[nodiscard]] constexpr double distanceSqr(Vec<N> const& a, Vec<N> const& b) noexcept
{
    return detail::sumAxes<N>([&](auto i) {
        double const d = a[i] - b[i];
        return d * d;
    });
}
//! Unit vector, or the zero vector when \p a has zero length.
template <std::size_t N>
[[nodiscard]] inline Vec<N> normalize(Vec<N> const& a) noexcept
{
    double const l = length(a);
    return l > 0 ? mul(a, 1.0 / l) : Vec<N>{};
}
[[nodiscard]] constexpr Vec3 cross(Vec3 const& a, Vec3 const& b) noexcept
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
//! z-component of the 3D cross product of two 2D vectors.
[[nodiscard]] constexpr double cross(Vec2 const& a, Vec2 const& b) noexcept
{
    return a[0] * b[1] - a[1] * b[0];
}
}  // namespace geom

// ---------------------------------------------------------------------------
// Ray with precomputed reciprocal direction
// ---------------------------------------------------------------------------

/**
 * \brief Ray `org + t * dir` with per-axis data precomputed for repeated box tests.
 *
 * Build once per query with \ref make and pass to \ref Box::intersect: each box test then
 * needs no divisions and no per-axis swap.
 */
template <std::size_t N>
struct Ray {
    Vec<N> org{};
    Vec<N> dir{};
    Vec<N> inv{};                  //!< `1 / dir[i]` (unused where `dir[i] == 0`)
    std::array<bool, N> zero{};    //!< `dir[i] == 0`: the ray never crosses this axis' slabs
    std::array<bool, N> negative{};  //!< `dir[i] < 0`: enter through `hi`, leave through `lo`

    [[nodiscard]] static constexpr Ray make(Vec<N> const& org, Vec<N> const& dir) noexcept
    {
        Ray r;
        r.org = org;
        r.dir = dir;
        geom::detail::forEachAxis<N>([&](auto i) {
            r.zero[i] = dir[i] == 0;
            r.negative[i] = dir[i] < 0;
            r.inv[i] = r.zero[i] ? 0.0 : 1.0 / dir[i];
        });
        return r;
    }

    [[nodiscard]] constexpr Vec<N> at(double t) const noexcept { return geom::madd(org, dir, t); }
};

// ---------------------------------------------------------------------------
// Axis-aligned box
// ---------------------------------------------------------------------------

//! Closed axis-aligned box. Default-constructed boxes are empty (lo = +inf, hi = -inf).
template <std::size_t N>
struct Box {
    Vec<N> lo = filled(std::numeric_limits<double>::infinity());
    Vec<N> hi = filled(-std::numeric_limits<double>::infinity());

    [[nodiscard]] static constexpr Vec<N> filled(double v) noexcept
    {
        return geom::detail::makeVec<N>([v](auto) { return v; });
    }

    [[nodiscard]] static constexpr Box fromPoints(Vec<N> const& a, Vec<N> const& b) noexcept
    {
        Box r;
        geom::detail::forEachAxis<N>([&](auto i) {
            r.lo[i] = std::min(a[i], b[i]);
            r.hi[i] = std::max(a[i], b[i]);
        });
        return r;
    }

    [[nodiscard]] constexpr bool isEmpty() const noexcept
    {
        return geom::detail::anyAxis<N>([this](auto i) { return hi[i] < lo[i]; });
    }

    constexpr Box& expand(Vec<N> const& p) noexcept
    {
        geom::detail::forEachAxis<N>([&](auto i) {
            lo[i] = std::min(lo[i], p[i]);
            hi[i] = std::max(hi[i], p[i]);
        });
        return *this;
    }

    constexpr Box& expand(Box const& b) noexcept
    {
        geom::detail::forEachAxis<N>([&](auto i) {
            lo[i] = std::min(lo[i], b.lo[i]);
            hi[i] = std::max(hi[i], b.hi[i]);
        });
        return *this;
    }

    [[nodiscard]] constexpr Vec<N> center() const noexcept
    {
        return geom::detail::makeVec<N>([this](auto i) { return 0.5 * (lo[i] + hi[i]); });
    }

    [[nodiscard]] constexpr Vec<N> extent() const noexcept { return geom::sub(hi, lo); }

    [[nodiscard]] constexpr std::size_t longestAxis() const noexcept
    {
        std::size_t best = 0;
        double bestExtent = hi[0] - lo[0];
        geom::detail::forEachAxis<N>([&](auto i) {
            if constexpr (i > 0) {
                double const e = hi[i] - lo[i];
                if (e > bestExtent) {
                    best = i;
                    bestExtent = e;
                }
            }
        });
        return best;
    }

    //! Half the surface area (3D), half the perimeter (2D); the SAH cost measure.
    [[nodiscard]] constexpr double surfaceMeasure() const noexcept
    {
        if (isEmpty()) return 0;
        Vec<N> const e = extent();
        if constexpr (N == 1) {
            return e[0];
        } else if constexpr (N == 2) {
            return e[0] + e[1];
        } else if constexpr (N == 3) {
            return e[0] * e[1] + e[0] * e[2] + e[1] * e[2];
        } else {
            double s = 0;
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = i + 1; j < N; ++j) s += e[i] * e[j];
            return s;
        }
    }

    [[nodiscard]] constexpr bool contains(Vec<N> const& p) const noexcept
    {
        return geom::detail::allAxes<N>([&](auto i) { return lo[i] <= p[i] && p[i] <= hi[i]; });
    }

    [[nodiscard]] constexpr bool intersects(Box const& b) const noexcept
    {
        return geom::detail::allAxes<N>([&](auto i) { return lo[i] <= b.hi[i] && b.lo[i] <= hi[i]; });
    }

    //! Squared distance from \p p to the box; 0 when \p p is inside. Branch-free.
    [[nodiscard]] constexpr double distanceSqr(Vec<N> const& p) const noexcept
    {
        return geom::detail::sumAxes<N>([&](auto i) {
            // at most one of the two differences is positive for a non-empty box
            double const d = std::max(std::max(lo[i] - p[i], p[i] - hi[i]), 0.0);
            return d * d;
        });
    }

    [[nodiscard]] double distance(Vec<N> const& p) const noexcept { return std::sqrt(distanceSqr(p)); }

    //! Point of the box closest to \p p (\p p itself when inside).
    [[nodiscard]] constexpr Vec<N> clamp(Vec<N> const& p) const noexcept
    {
        return geom::detail::makeVec<N>([&](auto i) { return std::clamp(p[i], lo[i], hi[i]); });
    }

    /**
     * \brief Slab test against a precomputed \ref Ray, `t` in `[tMin, tMax]`.
     * \return `{tEnter, tExit}` clipped to `[tMin, tMax]`, or nullopt on a miss.
     */
    [[nodiscard]] constexpr std::optional<std::pair<double, double>> intersect(
        Ray<N> const& ray, double tMin, double tMax) const noexcept
    {
        // Deliberately a plain loop: the early return on a miss matters, and a fold-expression
        // version measured ~18% slower for ray queries under MSVC (tMin / tMax captured by
        // reference were not kept in registers). The trip count is a constant, so compilers
        // unroll it anyway.
        for (std::size_t i = 0; i < N; ++i) {
            if (ray.zero[i]) {
                if (ray.org[i] < lo[i] || hi[i] < ray.org[i]) return std::nullopt;
                continue;
            }
            double const tNear = ((ray.negative[i] ? hi[i] : lo[i]) - ray.org[i]) * ray.inv[i];
            double const tFar = ((ray.negative[i] ? lo[i] : hi[i]) - ray.org[i]) * ray.inv[i];
            tMin = std::max(tMin, tNear);
            tMax = std::min(tMax, tFar);
            if (tMin > tMax) return std::nullopt;
        }
        return std::pair{tMin, tMax};
    }

    /**
     * \brief Slab test for the ray `org + t * dir`, `t` in `[tMin, tMax]`.
     * \return `{tEnter, tExit}` clipped to `[tMin, tMax]`, or nullopt on a miss.
     * Zero direction components are handled without producing NaNs. For many boxes per
     * ray, build a \ref Ray once and call \ref intersect.
     */
    [[nodiscard]] constexpr std::optional<std::pair<double, double>> intersectRay(
        Vec<N> const& org, Vec<N> const& dir, double tMin, double tMax) const noexcept
    {
        return intersect(Ray<N>::make(org, dir), tMin, tMax);
    }
};

namespace geom
{
template <std::size_t N>
[[nodiscard]] constexpr Vec<N> closestPointOnSegment(Vec<N> const& a, Vec<N> const& b,
                                                     Vec<N> const& p) noexcept
{
    Vec<N> const ab = sub(b, a);
    double const denom = dot(ab, ab);
    if (!(denom > 0)) return a;
    double const t = std::clamp(dot(sub(p, a), ab) / denom, 0.0, 1.0);
    return madd(a, ab, t);
}

/**
 * \brief Closest point on triangle `abc` to `p` (Ericson, *Real-Time Collision Detection* §5.1.5).
 *
 * Works in any dimension (uses dot products only). Degenerate (collinear) triangles fall
 * back to the closest of the three edges.
 */
template <std::size_t N>
[[nodiscard]] constexpr Vec<N> closestPointOnTriangle(Vec<N> const& a, Vec<N> const& b,
                                                      Vec<N> const& c, Vec<N> const& p) noexcept
{
    Vec<N> const ab = sub(b, a);
    Vec<N> const ac = sub(c, a);
    Vec<N> const ap = sub(p, a);
    double const d1 = dot(ab, ap);
    double const d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return a;

    Vec<N> const bp = sub(p, b);
    double const d3 = dot(ab, bp);
    double const d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return b;

    double const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0 && d1 - d3 > 0) return madd(a, ab, d1 / (d1 - d3));

    Vec<N> const cp = sub(p, c);
    double const d5 = dot(ab, cp);
    double const d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return c;

    double const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0 && d2 - d6 > 0) return madd(a, ac, d2 / (d2 - d6));

    double const va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0 && (d4 - d3) + (d5 - d6) > 0)
        return madd(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)));

    double const sum = va + vb + vc;
    if (!(sum > 0)) {
        // degenerate triangle: best of the three edges
        Vec<N> best = closestPointOnSegment(a, b, p);
        double bestSq = distanceSqr(best, p);
        Vec<N> const q1 = closestPointOnSegment(b, c, p);
        if (double const d = distanceSqr(q1, p); d < bestSq) {
            best = q1;
            bestSq = d;
        }
        Vec<N> const q2 = closestPointOnSegment(c, a, p);
        if (distanceSqr(q2, p) < bestSq) best = q2;
        return best;
    }
    double const inv = 1.0 / sum;
    double const v = vb * inv;
    double const w = vc * inv;
    return detail::makeVec<N>([&](auto i) { return a[i] + ab[i] * v + ac[i] * w; });
}

//! \c true if 2D point \p p lies inside or on triangle `abc` (either winding).
[[nodiscard]] constexpr bool triangleContains(Vec2 const& a, Vec2 const& b, Vec2 const& c,
                                              Vec2 const& p) noexcept
{
    double const d1 = cross(sub(b, a), sub(p, a));
    double const d2 = cross(sub(c, b), sub(p, b));
    double const d3 = cross(sub(a, c), sub(p, c));
    bool const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
    bool const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
    return !(hasNeg && hasPos);
}

//! Ray `org + t*dir` against 2D segment `ab`; returns `t >= 0` of the crossing (parallel rays miss).
[[nodiscard]] constexpr std::optional<double> intersectRaySegment(Vec2 const& a, Vec2 const& b,
                                                                  Vec2 const& org,
                                                                  Vec2 const& dir) noexcept
{
    Vec2 const e = sub(b, a);
    double const denom = cross(dir, e);
    if (denom == 0) return std::nullopt;
    Vec2 const ao = sub(a, org);
    double const t = cross(ao, e) / denom;
    double const s = cross(ao, dir) / denom;
    if (t < 0 || s < 0 || s > 1) return std::nullopt;
    return t;
}

//! Möller–Trumbore ray / triangle intersection; returns `t >= 0` (two-sided, parallel rays miss).
[[nodiscard]] constexpr std::optional<double> intersectRayTriangle(Vec3 const& a, Vec3 const& b,
                                                                   Vec3 const& c, Vec3 const& org,
                                                                   Vec3 const& dir) noexcept
{
    Vec3 const e1 = sub(b, a);
    Vec3 const e2 = sub(c, a);
    Vec3 const pvec = cross(dir, e2);
    double const det = dot(e1, pvec);
    if (det == 0) return std::nullopt;
    double const inv = 1.0 / det;
    Vec3 const tvec = sub(org, a);
    double const u = dot(tvec, pvec) * inv;
    if (u < 0 || u > 1) return std::nullopt;
    Vec3 const qvec = cross(tvec, e1);
    double const v = dot(dir, qvec) * inv;
    if (v < 0 || u + v > 1) return std::nullopt;
    double const t = dot(e2, qvec) * inv;
    if (t < 0) return std::nullopt;
    return t;
}
}  // namespace geom

/** @} */

}  // namespace libkdtree
