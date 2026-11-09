#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "boxtree.hpp"
#include "kdtree.hpp"

namespace
{
using Point2 = std::array<float, 2>;
using Tree2 = libkdtree::KDTree<2, Point2>;

/** Point with id so \c find (location) vs \c findExact (==) differ. */
struct PointWithId {
    using value_type = float;
    float data[2]{};
    int id{};

    float operator[](std::size_t i) const { return data[i]; }

    bool operator==(const PointWithId& o) const
    {
        return data[0] == o.data[0] && data[1] == o.data[1] && id == o.id;
    }
};

using TreeTagged = libkdtree::KDTree<2, PointWithId>;

void expectSamePointMultiset(const Tree2& t, std::vector<Point2> expected)
{
    std::vector<Point2> got;
    got.reserve(t.size());
    for (auto it = t.begin(); it != t.end(); ++it) {
        got.push_back(*it);
    }
    std::sort(expected.begin(), expected.end());
    std::sort(got.begin(), got.end());
    EXPECT_EQ(got, expected);
}

struct VisitCounter {
    int count = 0;
    void operator()(Point2 const&) { ++count; }
};
}  // namespace

// --- Lifecycle / capacity ---------------------------------------------------

TEST(KDTree, DefaultEmpty)
{
    Tree2 t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.begin(), t.end());
}

TEST(KDTree, MaxSize)
{
    Tree2 t;
    EXPECT_EQ(t.maxSize(), std::numeric_limits<Tree2::size_type>::max());
}

TEST(KDTree, GetAllocator)
{
    Tree2 t;
    [[maybe_unused]] auto a = t.getAllocator();
}

TEST(KDTree, InsertAndSize)
{
    Tree2 t;
    t.insert(Point2{1.0f, 2.0f});
    t.insert(Point2{3.0f, 4.0f});
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.size(), 2u);
}

TEST(KDTree, Clear)
{
    Tree2 t;
    t.insert(Point2{0.0f, 0.0f});
    t.clear();
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
}

TEST(KDTree, RangeConstructorPreservesCount)
{
    std::vector<Point2> pts(50);
    for (int i = 0; i < 50; ++i) {
        pts[static_cast<std::size_t>(i)] = Point2{static_cast<float>(i), static_cast<float>(-i)};
    }
    Tree2 t(pts.begin(), pts.end());
    EXPECT_EQ(t.size(), 50u);
}

TEST(KDTree, CopyConstruct)
{
    Tree2 a;
    a.insert(Point2{1.0f, 1.0f});
    a.insert(Point2{2.0f, 2.0f});
    Tree2 b(a);
    EXPECT_EQ(b.size(), a.size());
    expectSamePointMultiset(b, {{1.f, 1.f}, {2.f, 2.f}});
}

TEST(KDTree, CopyAssign)
{
    Tree2 a;
    a.insert(Point2{1.f, 2.f});
    Tree2 b;
    b.insert(Point2{9.f, 9.f});
    b.insert(Point2{8.f, 8.f});
    a = b;
    EXPECT_EQ(a.size(), 2u);
    expectSamePointMultiset(a, {{8.f, 8.f}, {9.f, 9.f}});
}

TEST(KDTree, CopyAssignSelf)
{
    Tree2 t;
    t.insert(Point2{1.f, 1.f});
    t = t;
    EXPECT_EQ(t.size(), 1u);
}

// --- Accessors / functors ---------------------------------------------------

TEST(KDTree, ValueCompValueAcc)
{
    Tree2 t;
    [[maybe_unused]] auto cmp = t.valueComparator();
    [[maybe_unused]] auto acc = t.valueAccessor();
    const Tree2& ct = t;
    EXPECT_EQ(&ct.valueDistance(), &t.valueDistance());
}

TEST(KDTree, ValueDistanceNonConst)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    auto& d = t.valueDistance();
    (void)d;
    const Tree2& ct = t;
    EXPECT_EQ(&ct.valueDistance(), &t.valueDistance());
}

// --- Iterators --------------------------------------------------------------

TEST(KDTree, ForwardIteration)
{
    Tree2 t;
    t.insert(Point2{1.f, 0.f});
    t.insert(Point2{0.f, 1.f});
    int n = 0;
    for (auto it = t.begin(); it != t.end(); ++it) {
        EXPECT_GE((*it)[0], 0.f);
        ++n;
    }
    EXPECT_EQ(n, 2);
}

TEST(KDTree, ReverseIteration)
{
    Tree2 t;
    t.insert(Point2{1.f, 2.f});
    t.insert(Point2{3.f, 4.f});
    std::vector<Point2> forward;
    for (auto it = t.begin(); it != t.end(); ++it) {
        forward.push_back(*it);
    }
    std::vector<Point2> backward;
    for (auto it = t.rbegin(); it != t.rend(); ++it) {
        backward.push_back(*it);
    }
    std::reverse(backward.begin(), backward.end());
    std::sort(forward.begin(), forward.end());
    std::sort(backward.begin(), backward.end());
    EXPECT_EQ(forward, backward);
}

TEST(KDTree, IteratorArrow)
{
    Tree2 t;
    t.insert(Point2{5.f, 6.f});
    auto it = t.begin();
    ASSERT_NE(it, t.end());
    EXPECT_FLOAT_EQ(it->operator[](0), 5.f);
}

// --- insert -----------------------------------------------------------------

TEST(KDTree, InsertReturnsIterator)
{
    Tree2 t;
    auto it = t.insert(Point2{7.f, 8.f});
    ASSERT_NE(it, t.end());
    EXPECT_FLOAT_EQ((*it)[0], 7.f);
}

TEST(KDTree, InsertWithIgnoredPosition)
{
    Tree2 t;
    t.insert(t.end(), Point2{1.f, 1.f});
    EXPECT_EQ(t.size(), 1u);
}

TEST(KDTree, InsertRange)
{
    Tree2 t;
    std::vector<Point2> v = {{0.f, 0.f}, {1.f, 1.f}, {2.f, 2.f}};
    t.insert(v.begin(), v.end());
    EXPECT_EQ(t.size(), 3u);
}

TEST(KDTree, InsertNCopiesAtIterator)
{
    Tree2 t;
    t.insert(t.begin(), 4u, Point2{3.f, 3.f});
    EXPECT_EQ(t.size(), 4u);
}

TEST(KDTree, InsertIteratorRangeAtPosition)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    std::vector<Point2> more = {{10.f, 10.f}, {11.f, 11.f}};
    t.insert(t.begin(), more.begin(), more.end());
    EXPECT_EQ(t.size(), 3u);
}

// --- erase / find -----------------------------------------------------------

TEST(KDTree, FindMiss)
{
    Tree2 t;
    t.insert(Point2{1.f, 2.f});
    EXPECT_EQ(t.find(Point2{99.f, 99.f}), t.end());
}

TEST(KDTree, FindHit)
{
    Tree2 t;
    t.insert(Point2{4.f, 5.f});
    auto it = t.find(Point2{4.f, 5.f});
    ASSERT_NE(it, t.end());
    EXPECT_FLOAT_EQ((*it)[0], 4.f);
    EXPECT_FLOAT_EQ((*it)[1], 5.f);
}

TEST(KDTree, FindExactDistinctIds)
{
    TreeTagged t;
    PointWithId a{{1.f, 1.f}, 10};
    PointWithId b{{1.f, 1.f}, 20};
    t.insert(a);
    t.insert(b);
    auto ia = t.findExact(a);
    auto ib = t.findExact(b);
    ASSERT_NE(ia, t.end());
    ASSERT_NE(ib, t.end());
    EXPECT_EQ(ia->id, 10);
    EXPECT_EQ(ib->id, 20);
}

TEST(KDTree, EraseByIterator)
{
    Tree2 t;
    t.insert(Point2{1.f, 1.f});
    t.insert(Point2{2.f, 2.f});
    auto it = t.find(Point2{1.f, 1.f});
    ASSERT_NE(it, t.end());
    t.erase(it);
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(t.find(Point2{1.f, 1.f}), t.end());
}

TEST(KDTree, EraseByValueUsesFind)
{
    Tree2 t;
    t.insert(Point2{5.f, 5.f});
    t.erase(Point2{5.f, 5.f});
    EXPECT_TRUE(t.empty());
}

TEST(KDTree, EraseExact)
{
    TreeTagged t;
    PointWithId p{{2.f, 3.f}, 7};
    t.insert(p);
    t.eraseExact(p);
    EXPECT_TRUE(t.empty());
}

// --- Nearest ----------------------------------------------------------------

TEST(KDTree, FindNearest)
{
    std::vector<Point2> pts = {
        {0.0f, 0.0f},
        {10.0f, 0.0f},
        {0.0f, 10.0f},
    };
    Tree2 t(pts.begin(), pts.end());

    Point2 query{0.1f, 0.2f};
    auto [it, dist] = t.findNearest(query);
    ASSERT_NE(it, t.end());
    EXPECT_NEAR((*it)[0], 0.0f, 1e-5f);
    EXPECT_NEAR((*it)[1], 0.0f, 1e-5f);
    EXPECT_LT(dist, 1.0f);
}

TEST(KDTree, FindNearestEmptyReturnsEnd)
{
    Tree2 t;
    auto [it, dist] = t.findNearest(Point2{0.f, 0.f});
    EXPECT_EQ(it, t.end());
    EXPECT_FLOAT_EQ(dist, 0.f);
}

TEST(KDTree, FindNearestWithMaxRadius)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{100.f, 100.f});
    {
        auto [it, d] = t.findNearest(Point2{0.1f, 0.1f}, 0.5f);
        ASSERT_NE(it, t.end());
        EXPECT_LT(d, 0.5f);
    }
    {
        auto [it, d] = t.findNearest(Point2{50.f, 50.f}, 0.001f);
        EXPECT_EQ(it, t.end());
        EXPECT_FLOAT_EQ(d, 0.001f);
    }
}

TEST(KDTree, FindNearestIf)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{10.f, 0.f});
    auto [it, d] = t.findNearestIf(
        Point2{1.f, 0.f}, 100.f,
        [](Point2 const& p) { return p[0] > 5.f; });
    ASSERT_NE(it, t.end());
    EXPECT_GT((*it)[0], 5.f);
    EXPECT_LT(d, 100.f);
}

// --- Range queries (axis-aligned box / L_inf) -------------------------------

TEST(KDTree, CountWithinRangeEmptyTree)
{
    Tree2 t;
    EXPECT_EQ(t.countWithinRange(Point2{0.f, 0.f}, 1.f), 0u);
}

TEST(KDTree, CountWithinRangeCenterRadius)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{1.f, 0.f});
    t.insert(Point2{10.f, 10.f});
    EXPECT_EQ(t.countWithinRange(Point2{0.f, 0.f}, 2.f), 2u);
    EXPECT_EQ(t.countWithinRange(Point2{0.f, 0.f}, 0.1f), 1u);
}

TEST(KDTree, CountWithinRangeRegion)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{2.f, 2.f});
    Tree2::Region reg(Point2{1.f, 1.f}, 1.5f, t.valueAccessor(), t.valueComparator());
    EXPECT_EQ(t.countWithinRange(reg), 2u);
}

TEST(KDTree, VisitWithinRangeCenterRadius)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{5.f, 5.f});
    VisitCounter c;
    VisitCounter out = t.visitWithinRange(Point2{0.f, 0.f}, 1.f, c);
    EXPECT_EQ(out.count, 1);
}

TEST(KDTree, VisitWithinRangeRegion)
{
    Tree2 t;
    t.insert(Point2{1.f, 1.f});
    Tree2::Region reg(Point2{0.f, 0.f}, 2.f, t.valueAccessor(), t.valueComparator());
    VisitCounter c;
    VisitCounter out = t.visitWithinRange(reg, c);
    EXPECT_EQ(out.count, 1);
}

TEST(KDTree, FindWithinRangeEmptyTree)
{
    Tree2 t;
    std::vector<Point2> out;
    t.findWithinRange(Point2{0.f, 0.f}, 1.f, std::back_inserter(out));
    EXPECT_TRUE(out.empty());
}

TEST(KDTree, VisitWithinRangeEmptyTree)
{
    Tree2 t;
    VisitCounter c;
    VisitCounter out = t.visitWithinRange(Point2{0.f, 0.f}, 1.f, c);
    EXPECT_EQ(out.count, 0);
}

TEST(KDTree, FindWithinRangeCenterRadius)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{3.f, 3.f});
    std::vector<Point2> out;
    t.findWithinRange(Point2{0.f, 0.f}, 1.f, std::back_inserter(out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FLOAT_EQ(out[0][0], 0.f);
}

TEST(KDTree, FindWithinRangeRegion)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{10.f, 10.f});
    Tree2::Region reg(Point2{0.f, 0.f}, 0.5f, t.valueAccessor(), t.valueComparator());
    std::vector<Point2> out;
    t.findWithinRange(reg, std::back_inserter(out));
    ASSERT_EQ(out.size(), 1u);
}

// --- optimise / efficient_replace -------------------------------------------

TEST(KDTree, OptimisePreservesElements)
{
    Tree2 t;
    std::vector<Point2> pts = {{3.f, 3.f}, {1.f, 1.f}, {2.f, 2.f}};
    for (auto& p : pts) {
        t.insert(p);
    }
    t.optimize();
    expectSamePointMultiset(t, pts);
    t.checkTreeInvariants();
}

TEST(KDTree, OptimizeAlias)
{
    Tree2 t;
    t.insert(Point2{1.f, 2.f});
    t.optimize();
    EXPECT_EQ(t.size(), 1u);
    t.checkTreeInvariants();
}

TEST(KDTree, EfficientReplaceAndOptimise)
{
    Tree2 t;
    t.insert(Point2{99.f, 99.f});
    std::vector<Point2> v = {{0.f, 0.f}, {1.f, 1.f}};
    t.assignOptimized(v);
    EXPECT_EQ(t.size(), 2u);
    expectSamePointMultiset(t, {{0.f, 0.f}, {1.f, 1.f}});
    t.checkTreeInvariants();
}

// ===========================================================================
// Large / adversarial data: every query is cross-checked against brute force.
// ===========================================================================

namespace
{
// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <std::size_t K>
using PointD = std::array<double, K>;

template <std::size_t K>
using TreeD = libkdtree::KDTree<K, PointD<K>>;

template <std::size_t K>
double sqDist(PointD<K> const& a, PointD<K> const& b)
{
    double s = 0;
    for (std::size_t i = 0; i < K; ++i) {
        double const d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

template <std::size_t K>
std::vector<PointD<K>> uniformPoints(std::size_t n, double lo, double hi, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(lo, hi);
    std::vector<PointD<K>> pts(n);
    for (auto& p : pts)
        for (auto& c : p) c = u(rng);
    return pts;
}

//! Points drawn from several tight Gaussian clusters (very uneven density).
template <std::size_t K>
std::vector<PointD<K>> clusteredPoints(std::size_t n, std::size_t clusters, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> centre(-1000.0, 1000.0);
    std::uniform_real_distribution<double> spread(0.01, 5.0);
    std::vector<PointD<K>> centres(clusters);
    std::vector<double> sigmas(clusters);
    for (std::size_t c = 0; c < clusters; ++c) {
        for (auto& x : centres[c]) x = centre(rng);
        sigmas[c] = spread(rng);
    }
    std::uniform_int_distribution<std::size_t> pick(0, clusters - 1);
    std::vector<PointD<K>> pts(n);
    for (auto& p : pts) {
        std::size_t const c = pick(rng);
        std::normal_distribution<double> g(0.0, sigmas[c]);
        for (std::size_t i = 0; i < K; ++i) p[i] = centres[c][i] + g(rng);
    }
    return pts;
}

//! Integer-valued coordinates in [0, range): many exact duplicates and ties.
template <std::size_t K>
std::vector<PointD<K>> latticePoints(std::size_t n, int range, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> u(0, range - 1);
    std::vector<PointD<K>> pts(n);
    for (auto& p : pts)
        for (auto& c : p) c = static_cast<double>(u(rng));
    return pts;
}

template <class Tree, class Pt>
std::vector<Pt> contents(Tree const& t)
{
    std::vector<Pt> v(t.begin(), t.end());
    std::sort(v.begin(), v.end());
    return v;
}

template <std::size_t K>
double bruteNearestSq(std::vector<PointD<K>> const& pts, PointD<K> const& q)
{
    double best = std::numeric_limits<double>::infinity();
    for (auto const& p : pts) best = std::min(best, sqDist<K>(p, q));
    return best;
}

template <std::size_t K>
bool inBox(PointD<K> const& p, PointD<K> const& lo, PointD<K> const& hi)
{
    for (std::size_t i = 0; i < K; ++i)
        if (p[i] < lo[i] || hi[i] < p[i]) return false;
    return true;
}

template <std::size_t K>
std::size_t ceilLog2(std::size_t n)
{
    std::size_t h = 0;
    while ((std::size_t{1} << h) < n + 1) ++h;
    return h;
}

//! Checks nearest / bounded nearest / box queries for \p queries random probes.
template <std::size_t K>
void crossCheckQueries(TreeD<K> const& t, std::vector<PointD<K>> const& pts, std::size_t queries,
                       double lo, double hi, unsigned seed)
{
    std::mt19937 rng(seed);
    // probe slightly outside the data range too
    std::uniform_real_distribution<double> u(lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.1);
    std::uniform_real_distribution<double> ext(0.0, (hi - lo) * 0.2);

    for (std::size_t qi = 0; qi < queries; ++qi) {
        PointD<K> q;
        for (auto& c : q) c = u(rng);

        // nearest
        double const bestSq = bruteNearestSq<K>(pts, q);
        auto [it, d] = t.findNearest(q);
        ASSERT_NE(it, t.end());
        ASSERT_EQ(sqDist<K>(*it, q), bestSq) << "query " << qi;
        ASSERT_DOUBLE_EQ(d, std::sqrt(bestSq));

        // bounded nearest, bound both just inside and just outside the true distance
        double const r = std::sqrt(bestSq);
        {
            auto [it2, d2] = t.findNearest(q, r * 1.0001 + 1e-12);
            ASSERT_NE(it2, t.end());
            ASSERT_EQ(sqDist<K>(*it2, q), bestSq);
        }
        if (bestSq > 0) {
            double const tight = r * 0.999;
            auto [it3, d3] = t.findNearest(q, tight);
            ASSERT_EQ(it3, t.end());
            ASSERT_EQ(d3, tight);
        }

        // box query with random extents
        PointD<K> blo, bhi;
        for (std::size_t i = 0; i < K; ++i) {
            blo[i] = q[i] - ext(rng);
            bhi[i] = q[i] + ext(rng);
        }
        typename TreeD<K>::Region reg(t.valueAccessor(), t.valueComparator());
        for (std::size_t i = 0; i < K; ++i) {
            reg.mSubLow[i] = blo[i];
            reg.mSubHigh[i] = bhi[i];
        }
        std::vector<PointD<K>> expected;
        for (auto const& p : pts)
            if (inBox<K>(p, blo, bhi)) expected.push_back(p);
        std::vector<PointD<K>> got;
        t.findWithinRange(reg, std::back_inserter(got));
        std::sort(expected.begin(), expected.end());
        std::sort(got.begin(), got.end());
        ASSERT_EQ(got, expected) << "box query " << qi;
        ASSERT_EQ(t.countWithinRange(reg), expected.size());
    }
}

// ---------------------------------------------------------------------------
// Instrumentation: counting allocator and a value type that can throw on copy
// ---------------------------------------------------------------------------

struct AllocStats {
    long live = 0;
    long allocations = 0;
};

template <class T>
struct CountingAlloc {
    using value_type = T;
    AllocStats* stats;

    explicit CountingAlloc(AllocStats* s) noexcept : stats(s) {}
    template <class U>
    CountingAlloc(CountingAlloc<U> const& o) noexcept : stats(o.stats)
    {
    }

    T* allocate(std::size_t n)
    {
        stats->live += static_cast<long>(n);
        ++stats->allocations;
        return std::allocator<T>().allocate(n);
    }
    void deallocate(T* p, std::size_t n) noexcept
    {
        stats->live -= static_cast<long>(n);
        std::allocator<T>().deallocate(p, n);
    }
    template <class U>
    bool operator==(CountingAlloc<U> const& o) const noexcept
    {
        return stats == o.stats;
    }
};

using PointD2 = std::array<double, 2>;
using CountedTree = libkdtree::KDTree<2, PointD2, libkdtree::BracketAccessor<PointD2>,
                                      libkdtree::SquaredDifference<double, double>,
                                      std::less<double>, CountingAlloc<PointD2>>;

CountedTree makeCounted(AllocStats& s)
{
    return CountedTree(libkdtree::BracketAccessor<PointD2>(),
                       libkdtree::SquaredDifference<double, double>(), std::less<double>(),
                       CountedTree::allocator_type(&s));
}

struct Throwy {
    using value_type = int;
    static inline int copiesBeforeThrow = -1;  // -1: never throw

    std::array<int, 2> v{};

    Throwy(int a, int b) : v{a, b} {}
    Throwy(Throwy const& o) : v(o.v)
    {
        if (copiesBeforeThrow == 0) throw std::runtime_error("copy failed");
        if (copiesBeforeThrow > 0) --copiesBeforeThrow;
    }
    Throwy(Throwy&&) noexcept = default;
    Throwy& operator=(Throwy const&) = default;
    Throwy& operator=(Throwy&&) noexcept = default;

    int operator[](std::size_t i) const { return v[i]; }
};

// ---------------------------------------------------------------------------
// Custom accessor over a struct with extra payload
// ---------------------------------------------------------------------------

struct Particle {
    double pos[3];
    int id;
    bool operator==(Particle const& o) const
    {
        return pos[0] == o.pos[0] && pos[1] == o.pos[1] && pos[2] == o.pos[2] && id == o.id;
    }
};

struct ParticleAcc {
    using result_type = double;
    double operator()(Particle const& p, std::size_t d) const { return p.pos[d]; }
};

using ParticleTree = libkdtree::KDTree<3, Particle, ParticleAcc>;

//! Scalar constructible from int, for the single-pass iterator test.
struct Scalar {
    using value_type = int;
    int v;
    Scalar(int x) : v(x) {}  // NOLINT: implicit on purpose
    int operator[](std::size_t) const { return v; }
};

}  // namespace

// ===========================================================================
// Queries against brute force
// ===========================================================================

TEST(KDTreeLarge, Uniform3DQueriesMatchBruteForce)
{
    auto const pts = uniformPoints<3>(20000, -500.0, 500.0, 1);
    TreeD<3> t(pts.begin(), pts.end());
    ASSERT_EQ(t.size(), pts.size());
    ASSERT_TRUE(t.checkTreeInvariants());
    EXPECT_EQ(t.height(), ceilLog2<3>(pts.size()));
    EXPECT_EQ((contents<TreeD<3>, PointD<3>>(t)), [&] {
        auto v = pts;
        std::sort(v.begin(), v.end());
        return v;
    }());
    crossCheckQueries<3>(t, pts, 300, -500.0, 500.0, 2);
}

TEST(KDTreeLarge, ClusteredDataQueriesMatchBruteForce)
{
    auto const pts = clusteredPoints<2>(30000, 12, 3);
    TreeD<2> t(pts.begin(), pts.end());
    ASSERT_TRUE(t.checkTreeInvariants());
    crossCheckQueries<2>(t, pts, 300, -1000.0, 1000.0, 4);
}

TEST(KDTreeLarge, HighDimensionalQueriesMatchBruteForce)
{
    auto const pts = uniformPoints<8>(8000, 0.0, 1.0, 5);
    TreeD<8> t(pts.begin(), pts.end());
    ASSERT_TRUE(t.checkTreeInvariants());
    crossCheckQueries<8>(t, pts, 150, 0.0, 1.0, 6);
}

TEST(KDTreeLarge, IncrementallyInsertedTreeMatchesBruteForce)
{
    // unbalanced (one-by-one) insertion, then the same queries after optimize()
    auto const pts = uniformPoints<3>(10000, 0.0, 100.0, 7);
    TreeD<3> t;
    for (auto const& p : pts) t.insert(p);
    ASSERT_TRUE(t.checkTreeInvariants());
    crossCheckQueries<3>(t, pts, 150, 0.0, 100.0, 8);

    t.optimize();
    ASSERT_TRUE(t.checkTreeInvariants());
    EXPECT_EQ(t.size(), pts.size());
    EXPECT_EQ(t.height(), ceilLog2<3>(pts.size()));
    crossCheckQueries<3>(t, pts, 150, 0.0, 100.0, 9);
}

TEST(KDTreeLarge, LatticeWithHeavyDuplicates)
{
    // 20k points on a 12x12x12 lattice => ~11 copies of every location
    auto const pts = latticePoints<3>(20000, 12, 10);
    TreeD<3> t(pts.begin(), pts.end());
    ASSERT_TRUE(t.checkTreeInvariants());

    for (int x = -1; x <= 12; ++x)
        for (int y = -1; y <= 12; ++y)
            for (int z = -1; z <= 12; ++z) {
                PointD<3> const q{double(x), double(y), double(z)};
                auto const n = static_cast<std::size_t>(std::count(pts.begin(), pts.end(), q));
                EXPECT_EQ(t.contains(q), n > 0);
                // zero-radius box == exact-location count
                ASSERT_EQ(t.countWithinRange(q, 0.0), n);
            }

    crossCheckQueries<3>(t, pts, 200, 0.0, 11.0, 11);
}

TEST(KDTreeLarge, FindNearestIfMatchesFilteredBruteForce)
{
    auto const pts = uniformPoints<2>(15000, -100.0, 100.0, 12);
    TreeD<2> t(pts.begin(), pts.end());
    auto const pred = [](PointD<2> const& p) { return p[0] > p[1]; };  // half the plane

    std::mt19937 rng(13);
    std::uniform_real_distribution<double> u(-120.0, 120.0);
    for (int qi = 0; qi < 300; ++qi) {
        PointD<2> const q{u(rng), u(rng)};
        double bestSq = std::numeric_limits<double>::infinity();
        for (auto const& p : pts)
            if (pred(p)) bestSq = std::min(bestSq, sqDist<2>(p, q));
        auto [it, d] = t.findNearestIf(q, 1e9, pred);
        ASSERT_NE(it, t.end());
        ASSERT_TRUE(pred(*it));
        ASSERT_EQ(sqDist<2>(*it, q), bestSq) << "query " << qi;
    }
}

TEST(KDTreeLarge, CustomAccessorStructWithPayload)
{
    std::mt19937 rng(14);
    std::uniform_real_distribution<double> u(-10.0, 10.0);
    std::vector<Particle> ps;
    for (int i = 0; i < 12000; ++i) {
        // every location appears 3 times with different ids
        Particle p{{std::round(u(rng)), std::round(u(rng)), std::round(u(rng))}, i};
        ps.push_back(p);
    }
    ParticleTree t(ps.begin(), ps.end());
    ASSERT_TRUE(t.checkTreeInvariants());

    // findExact must locate each specific particle among same-location duplicates
    for (std::size_t i = 0; i < ps.size(); i += 7) {
        auto it = t.findExact(ps[i]);
        ASSERT_NE(it, t.end());
        EXPECT_EQ(it->id, ps[i].id);
    }
    // erase by id, every particle with an even id
    for (auto const& p : ps)
        if (p.id % 2 == 0) ASSERT_EQ(t.eraseExact(p), 1u);
    ASSERT_TRUE(t.checkTreeInvariants());
    EXPECT_EQ(t.size(), ps.size() / 2);
    for (auto const& v : t) EXPECT_EQ(v.id % 2, 1);
    EXPECT_EQ(t.eraseExact(ps[0]), 0u);  // already gone
}

// ===========================================================================
// Mutation: randomized operations against a reference model
// ===========================================================================

TEST(KDTreeLarge, RandomizedOperationsAgainstModel)
{
    std::mt19937 rng(15);
    std::uniform_int_distribution<int> coord(0, 30);
    std::uniform_int_distribution<int> op(0, 99);

    TreeD<3> t;
    std::vector<PointD<3>> model;

    auto randomPoint = [&] {
        return PointD<3>{double(coord(rng)), double(coord(rng)), double(coord(rng))};
    };

    for (int step = 0; step < 12000; ++step) {
        int const o = op(rng);
        if (o < 55 || model.empty()) {
            auto const p = randomPoint();
            t.insert(p);
            model.push_back(p);
        } else if (o < 75) {
            // erase by (possibly absent) value
            auto const p = randomPoint();
            auto mit = std::find(model.begin(), model.end(), p);
            ASSERT_EQ(t.erase(p), mit != model.end() ? 1u : 0u);
            if (mit != model.end()) model.erase(mit);
        } else if (o < 90) {
            // erase via iterator at a random position
            std::uniform_int_distribution<std::size_t> pos(0, model.size() - 1);
            auto it = std::next(t.begin(), static_cast<std::ptrdiff_t>(pos(rng)));
            auto const v = *it;
            t.erase(it);
            model.erase(std::find(model.begin(), model.end(), v));
        } else if (o < 99) {
            auto const q = randomPoint();
            auto [it, d] = t.findNearest(q);
            ASSERT_NE(it, t.end());
            ASSERT_EQ(sqDist<3>(*it, q), bruteNearestSq<3>(model, q));
        } else {
            t.optimize();
        }

        ASSERT_EQ(t.size(), model.size());
        if (step % 200 == 0) {
            ASSERT_TRUE(t.checkTreeInvariants()) << "step " << step;
            auto m = model;
            std::sort(m.begin(), m.end());
            ASSERT_EQ((contents<TreeD<3>, PointD<3>>(t)), m) << "step " << step;
        }
    }
    ASSERT_TRUE(t.checkTreeInvariants());
}

TEST(KDTreeLarge, EraseEverythingInRandomOrder)
{
    auto pts = latticePoints<2>(3000, 40, 16);
    TreeD<2> t(pts.begin(), pts.end());
    std::shuffle(pts.begin(), pts.end(), std::mt19937(17));

    for (std::size_t i = 0; i < pts.size(); ++i) {
        ASSERT_EQ(t.erase(pts[i]), 1u) << i;
        ASSERT_TRUE(t.checkTreeInvariants()) << "after erasing " << i;
        ASSERT_EQ(t.size(), pts.size() - i - 1);
    }
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.begin(), t.end());
}

TEST(KDTreeLarge, IteratorsToOtherElementsSurviveErase)
{
    auto const pts = uniformPoints<2>(4000, 0.0, 1.0, 18);
    TreeD<2> t(pts.begin(), pts.end());

    std::vector<TreeD<2>::const_iterator> keep;
    std::vector<TreeD<2>::const_iterator> drop;
    std::size_t i = 0;
    for (auto it = t.begin(); it != t.end(); ++it, ++i) (i % 3 == 0 ? keep : drop).push_back(it);
    std::vector<PointD<2>> keptValues;
    for (auto it : keep) keptValues.push_back(*it);

    for (auto it : drop) t.erase(it);
    ASSERT_TRUE(t.checkTreeInvariants());
    ASSERT_EQ(t.size(), keep.size());
    for (std::size_t k = 0; k < keep.size(); ++k) EXPECT_EQ(*keep[k], keptValues[k]);
}

// ===========================================================================
// Degenerate shapes (would overflow the stack with recursive traversal)
// ===========================================================================

TEST(KDTreeLarge, SortedInsertDegenerateChain)
{
    constexpr std::size_t N = 25000;
    TreeD<2> t;
    for (std::size_t i = 0; i < N; ++i) t.insert(PointD<2>{double(i), double(i)});
    EXPECT_EQ(t.height(), N);  // a single right-leaning chain
    ASSERT_TRUE(t.checkTreeInvariants());

    // iterate both ways
    EXPECT_EQ(static_cast<std::size_t>(std::distance(t.begin(), t.end())), N);
    EXPECT_EQ(static_cast<std::size_t>(std::distance(t.rbegin(), t.rend())), N);

    // deep finds, range and nearest queries
    for (std::size_t i = 0; i < N; i += 997) EXPECT_TRUE(t.contains(PointD<2>{double(i), double(i)}));
    EXPECT_FALSE(t.contains(PointD<2>{0.5, 0.5}));
    EXPECT_EQ(t.countWithinRange(PointD<2>{double(N - 10), double(N - 10)}, 5.0), 11u);
    {
        auto [it, d] = t.findNearest(PointD<2>{double(N) + 3.0, double(N) + 3.0});
        ASSERT_NE(it, t.end());
        EXPECT_EQ((*it)[0], double(N - 1));
    }

    // erase near the top (forces long replacement chains), then rebalance
    for (std::size_t i = 0; i < 50; ++i) ASSERT_EQ(t.erase(PointD<2>{double(i), double(i)}), 1u);
    ASSERT_TRUE(t.checkTreeInvariants());
    t.optimize();
    EXPECT_EQ(t.height(), ceilLog2<2>(N - 50));
    ASSERT_TRUE(t.checkTreeInvariants());

    // destroy a fresh degenerate chain (clear must be iterative too)
    TreeD<2> chain;
    for (std::size_t i = 0; i < N; ++i) chain.insert(PointD<2>{double(i), 0.0});
    chain.clear();
    EXPECT_TRUE(chain.empty());
}

TEST(KDTreeLarge, AllIdenticalPoints)
{
    TreeD<3> t;
    PointD<3> const p{1.0, 2.0, 3.0};
    for (int i = 0; i < 5000; ++i) t.insert(p);
    ASSERT_TRUE(t.checkTreeInvariants());
    EXPECT_EQ(t.countWithinRange(p, 0.0), 5000u);
    auto [it, d] = t.findNearest(PointD<3>{0.0, 0.0, 0.0});
    ASSERT_NE(it, t.end());
    EXPECT_DOUBLE_EQ(d, std::sqrt(14.0));
    for (int i = 0; i < 5000; ++i) ASSERT_EQ(t.erase(p), 1u);
    EXPECT_TRUE(t.empty());
    EXPECT_TRUE(t.checkTreeInvariants());
}

// ===========================================================================
// Value semantics, allocators and exception safety
// ===========================================================================

TEST(KDTreeLarge, MoveAndSwapStealNodes)
{
    auto const pts = uniformPoints<2>(5000, 0.0, 1.0, 19);
    TreeD<2> a(pts.begin(), pts.end());
    auto const firstAddr = &*a.begin();

    TreeD<2> b(std::move(a));
    EXPECT_TRUE(a.empty());  // NOLINT(bugprone-use-after-move): moved-from is specified empty
    EXPECT_EQ(a.begin(), a.end());
    EXPECT_TRUE(a.checkTreeInvariants());
    EXPECT_EQ(b.size(), pts.size());
    EXPECT_EQ(&*b.begin(), firstAddr);  // same nodes, no copy
    EXPECT_TRUE(b.checkTreeInvariants());

    TreeD<2> c;
    c.insert(PointD<2>{5.0, 5.0});
    c = std::move(b);
    EXPECT_EQ(c.size(), pts.size());
    EXPECT_TRUE(c.checkTreeInvariants());

    TreeD<2> d{PointD<2>{1.0, 1.0}, PointD<2>{2.0, 2.0}};
    swap(c, d);
    EXPECT_EQ(c.size(), 2u);
    EXPECT_EQ(d.size(), pts.size());
    EXPECT_TRUE(c.checkTreeInvariants());
    EXPECT_TRUE(d.checkTreeInvariants());
    crossCheckQueries<2>(d, pts, 50, 0.0, 1.0, 20);

    // moved-from trees stay usable
    a.insert(PointD<2>{3.0, 3.0});
    EXPECT_EQ(a.size(), 1u);
    EXPECT_TRUE(a.checkTreeInvariants());
}

TEST(KDTreeLarge, CopyIsDeepAndBalanced)
{
    TreeD<2> src;
    for (int i = 0; i < 4000; ++i) src.insert(PointD<2>{double(i), double(i % 17)});
    TreeD<2> copy(src);
    EXPECT_EQ(copy.size(), src.size());
    EXPECT_EQ(copy.height(), ceilLog2<2>(src.size()));
    EXPECT_EQ((contents<TreeD<2>, PointD<2>>(copy)), (contents<TreeD<2>, PointD<2>>(src)));
    src.clear();
    EXPECT_EQ(copy.size(), 4000u);
    EXPECT_TRUE(copy.checkTreeInvariants());
}

TEST(KDTreeLarge, CountingAllocatorNoLeaks)
{
    AllocStats s1, s2;
    {
        CountedTree t = makeCounted(s1);
        auto const pts = uniformPoints<2>(3000, 0.0, 1.0, 21);
        for (auto const& p : pts) t.insert(p);
        EXPECT_EQ(s1.live, 3000);
        for (std::size_t i = 0; i < pts.size(); i += 2) t.erase(pts[i]);
        EXPECT_EQ(s1.live, 1500);
        t.optimize();
        EXPECT_EQ(s1.live, 1500);

        CountedTree copy(t);  // select_on_container_copy_construction -> same stats
        EXPECT_EQ(s1.live, 3000);

        // move-assign between unequal, non-propagating allocators => element-wise move
        CountedTree other = makeCounted(s2);
        other = std::move(copy);
        EXPECT_EQ(other.size(), 1500u);
        EXPECT_EQ(s2.live, 1500);
        EXPECT_EQ(s1.live, 1500);
        EXPECT_TRUE(other.checkTreeInvariants());
    }
    EXPECT_EQ(s1.live, 0);
    EXPECT_EQ(s2.live, 0);
}

TEST(KDTreeLarge, ExceptionDuringBulkBuildLeavesEmptyTreeAndNoLeaks)
{
    std::vector<Throwy> vals;
    for (int i = 0; i < 2000; ++i) vals.emplace_back(i % 97, i % 89);

    libkdtree::KDTree<2, Throwy> t;
    t.insert(Throwy(1, 1));
    Throwy::copiesBeforeThrow = 1000;  // fail halfway through building
    EXPECT_THROW(t.assignOptimized(vals), std::runtime_error);
    Throwy::copiesBeforeThrow = -1;
    EXPECT_TRUE(t.empty());
    EXPECT_TRUE(t.checkTreeInvariants());

    t.assignOptimized(vals);
    EXPECT_EQ(t.size(), vals.size());
    EXPECT_TRUE(t.checkTreeInvariants());
}

TEST(KDTreeLarge, SinglePassInputIterators)
{
    std::ostringstream os;
    for (int i = 0; i < 5000; ++i) os << (i * 7919) % 10007 << ' ';
    std::istringstream is(os.str());
    libkdtree::KDTree<1, Scalar> t{std::istream_iterator<int>(is), std::istream_iterator<int>()};
    EXPECT_EQ(t.size(), 5000u);
    EXPECT_TRUE(t.checkTreeInvariants());
    EXPECT_TRUE(t.contains(Scalar((123 * 7919) % 10007)));
}

TEST(KDTreeLarge, EmplaceAndMoveInsert)
{
    libkdtree::KDTree<2, Throwy> t;
    for (int i = 0; i < 1000; ++i) t.emplace(i, -i);
    Throwy::copiesBeforeThrow = 0;  // any copy would now throw
    for (int i = 0; i < 1000; ++i) t.insert(Throwy(i, i));
    t.optimize();  // must move, not copy
    Throwy::copiesBeforeThrow = -1;
    EXPECT_EQ(t.size(), 2000u);
    EXPECT_TRUE(t.checkTreeInvariants());
}

// ===========================================================================
// BoxTree: closest-point / signed-distance / ray queries over primitives.
// Every query is cross-checked against a brute-force scan.
// ===========================================================================

namespace
{
using libkdtree::Box;
using libkdtree::BoxTree;
using libkdtree::Segment;
using libkdtree::SplitMetric;
using libkdtree::Triangle;
using libkdtree::Vec2;
using libkdtree::Vec3;
namespace geom = libkdtree::geom;

//! Filled disk: a user-defined primitive (traits specialised below).
struct Disk {
    Vec2 c;
    double r;
};
}  // namespace

template <>
struct libkdtree::PrimitiveTraits<Disk> {
    static constexpr std::size_t dimension = 2;
    static Box<2> bounds(Disk const& d) noexcept
    {
        return Box<2>{{d.c[0] - d.r, d.c[1] - d.r}, {d.c[0] + d.r, d.c[1] + d.r}};
    }
    static Vec2 closestPoint(Disk const& d, Vec2 const& q) noexcept
    {
        Vec2 const v = geom::sub(q, d.c);
        double const len = geom::length(v);
        return len <= d.r ? q : geom::madd(d.c, v, d.r / len);
    }
};

namespace
{
template <class Prim>
using TraitsOf = libkdtree::PrimitiveTraits<Prim>;

//! Brute-force nearest: {distanceSqr, insertion index} (index -1 when none qualifies).
template <class Prim, class Accept>
std::pair<double, long> bruteNearest(std::vector<Prim> const& prims,
                                     libkdtree::Vec<TraitsOf<Prim>::dimension> const& q,
                                     double maxDistance, Accept accept)
{
    double best = maxDistance * maxDistance;
    long idx = -1;
    for (std::size_t i = 0; i < prims.size(); ++i) {
        auto const cp = TraitsOf<Prim>::closestPoint(prims[i], q);
        double const d2 = geom::distanceSqr(cp, q);
        if ((d2 < best || (idx < 0 && d2 <= best)) && accept(cp)) {
            best = d2;
            idx = static_cast<long>(i);
        }
    }
    return {best, idx};
}

template <class Prim>
std::pair<double, long> bruteNearest(std::vector<Prim> const& prims,
                                     libkdtree::Vec<TraitsOf<Prim>::dimension> const& q,
                                     double maxDistance = std::numeric_limits<double>::infinity())
{
    return bruteNearest(prims, q, maxDistance, [](auto const&) { return true; });
}

//! Brute-force first ray hit: smallest t in [0, tMax], or nullopt.
template <class Prim>
std::optional<double> bruteRay(std::vector<Prim> const& prims,
                               libkdtree::Vec<TraitsOf<Prim>::dimension> const& o,
                               libkdtree::Vec<TraitsOf<Prim>::dimension> const& d, double tMax)
{
    std::optional<double> best;
    for (auto const& p : prims) {
        auto const t = TraitsOf<Prim>::intersectRay(p, o, d);
        if (t && *t >= 0 && *t <= tMax && (!best || *t < *best)) best = t;
    }
    return best;
}

std::vector<Triangle<3>> randomTriangles(std::size_t n, double range, double maxSize, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> c(-range, range);
    std::uniform_real_distribution<double> s(0.05, maxSize);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<Triangle<3>> tris(n);
    for (auto& t : tris) {
        Vec3 const ctr{c(rng), c(rng), c(rng)};
        double const sz = s(rng);
        auto vtx = [&] { return Vec3{ctr[0] + sz * u(rng), ctr[1] + sz * u(rng), ctr[2] + sz * u(rng)}; };
        t = {vtx(), vtx(), vtx()};
    }
    return tris;
}

//! Closed lat-long sphere with outward-facing (counter-clockwise) triangles.
std::vector<Triangle<3>> sphereMesh(double radius, int nLat, int nLon)
{
    auto vtx = [&](int i, int j) {
        double const th = std::numbers::pi * i / nLat;
        double const ph = 2 * std::numbers::pi * j / nLon;
        return Vec3{radius * std::sin(th) * std::cos(ph), radius * std::sin(th) * std::sin(ph),
                    radius * std::cos(th)};
    };
    std::vector<Triangle<3>> tris;
    auto add = [&](Vec3 a, Vec3 b, Vec3 c) {
        Vec3 const n = geom::cross(geom::sub(b, a), geom::sub(c, a));
        if (geom::lengthSqr(n) == 0) return;  // pole slivers
        Vec3 const ctr = geom::mul(geom::add(geom::add(a, b), c), 1.0 / 3);
        if (geom::dot(n, ctr) < 0) std::swap(b, c);
        tris.push_back({a, b, c});
    };
    for (int i = 0; i < nLat; ++i)
        for (int j = 0; j < nLon; ++j) {
            add(vtx(i, j), vtx(i + 1, j), vtx(i + 1, j + 1));
            add(vtx(i, j), vtx(i + 1, j + 1), vtx(i, j + 1));
        }
    return tris;
}

Vec3 randomDirection(std::mt19937& rng)
{
    std::normal_distribution<double> g;
    Vec3 v;
    do v = {g(rng), g(rng), g(rng)};
    while (geom::lengthSqr(v) < 1e-12);
    return geom::normalize(v);
}

//! Cross-checks every closest-point flavour plus rays for a 3D triangle tree.
void crossCheckTriangles(BoxTree<Triangle<3>> const& tree, std::vector<Triangle<3>> const& tris,
                         int queries, double range, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-range * 1.2, range * 1.2);
    for (int qi = 0; qi < queries; ++qi) {
        Vec3 const q{u(rng), u(rng), u(rng)};

        // unbounded
        auto const [bestSq, bestIdx] = bruteNearest(tris, q);
        Vec3 lastPoint{};
        Triangle<3> const* lastTri = nullptr;
        auto const d2 = tree.closestPointSqr(q, lastPoint, lastTri);
        ASSERT_TRUE(d2.has_value());
        ASSERT_EQ(*d2, bestSq) << "query " << qi;
        ASSERT_NE(lastTri, nullptr);
        ASSERT_EQ(geom::distanceSqr(lastPoint, q), bestSq);
        // the reported primitive really is at that distance
        ASSERT_EQ(geom::distanceSqr(TraitsOf<Triangle<3>>::closestPoint(*lastTri, q), q), bestSq);
        ASSERT_EQ(&tree.primitive(tree.index(lastTri)), lastTri);
        ASSERT_DOUBLE_EQ(*tree.closestPoint(q), std::sqrt(bestSq));
        ASSERT_EQ(tree.distanceSqr(q), bestSq);

        // bounded: just above and just below the true distance
        double const r = std::sqrt(bestSq);
        ASSERT_EQ(tree.closestPointSqr(q, r * 1.001 + 1e-12).value(), bestSq);
        if (bestSq > 0) {
            Triangle<3> const* miss = lastTri;
            Vec3 untouched{7, 7, 7};
            ASSERT_FALSE(tree.closestPoint(q, r * 0.999, untouched, miss).has_value());
            ASSERT_EQ(miss, nullptr);
            ASSERT_EQ(untouched, (Vec3{7, 7, 7}));
            ASSERT_EQ(tree.distance(q, r * 0.999), r * 0.999);
        }

        // all within a radius
        double const radius = r + 3.0;
        std::vector<Triangle<3> const*> found;
        Vec3 nearestPt{};
        ASSERT_EQ(tree.closestPointSqr(q, radius, nearestPt, found).value(), bestSq);
        std::vector<std::size_t> got, expected;
        for (auto const* t : found) got.push_back(tree.index(t));
        for (std::size_t i = 0; i < tris.size(); ++i)
            if (geom::distanceSqr(TraitsOf<Triangle<3>>::closestPoint(tris[i], q), q) <= radius * radius)
                expected.push_back(i);
        std::sort(got.begin(), got.end());
        ASSERT_EQ(got, expected) << "within query " << qi;

        // half-space restricted
        Vec3 const v = randomDirection(rng);
        auto const [outSq, outIdx] = bruteNearest(tris, q, std::numeric_limits<double>::infinity(),
                                                  [&](Vec3 const& cp) { return geom::dot(geom::sub(cp, q), v) >= 0; });
        auto const out = tree.closestPointOutside(q, v);
        ASSERT_EQ(out.has_value(), outIdx >= 0);
        if (out) ASSERT_DOUBLE_EQ(*out, std::sqrt(outSq));

        // ray and segment
        auto const rayT = bruteRay(tris, q, v, std::numeric_limits<double>::infinity());
        Triangle<3> const* hitTri = nullptr;
        Vec3 hitPt{};
        auto const rayD = tree.intersectRayDistance(q, v, hitPt, hitTri);
        ASSERT_EQ(rayD.has_value(), rayT.has_value()) << "ray " << qi;
        if (rayD) {
            ASSERT_EQ(*rayD, *rayT * geom::length(v));
            ASSERT_EQ(TraitsOf<Triangle<3>>::intersectRay(*hitTri, q, v), rayT);
        }
        Vec3 const p2 = geom::madd(q, v, range * 0.5);
        auto const segT = bruteRay(tris, q, geom::sub(p2, q), 1.0);
        auto const segD = tree.intersectSegmentDistance(q, p2);
        ASSERT_EQ(segD.has_value(), segT.has_value()) << "segment " << qi;
        if (segD) ASSERT_DOUBLE_EQ(*segD, *segT * geom::length(geom::sub(p2, q)));
    }
}
}  // namespace

TEST(BoxTreeGeometry, ClosestPointOnTriangleRegions)
{
    Vec3 const a{0, 0, 0}, b{2, 0, 0}, c{0, 2, 0};
    auto cp = [&](Vec3 p) { return geom::closestPointOnTriangle(a, b, c, p); };
    EXPECT_EQ(cp({0.5, 0.5, 3}), (Vec3{0.5, 0.5, 0}));  // face
    EXPECT_EQ(cp({-1, -1, 1}), a);                      // vertex regions
    EXPECT_EQ(cp({5, -1, 0}), b);
    EXPECT_EQ(cp({-1, 5, 0}), c);
    EXPECT_EQ(cp({1, -3, 0}), (Vec3{1, 0, 0}));  // edge ab
    EXPECT_EQ(cp({-3, 1, 0}), (Vec3{0, 1, 0}));  // edge ac
    auto const bc = cp({2, 2, 0});               // edge bc
    EXPECT_NEAR(bc[0], 1, 1e-12);
    EXPECT_NEAR(bc[1], 1, 1e-12);
    // degenerate (collinear) triangle falls back to its edges
    auto const dg = geom::closestPointOnTriangle(Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{2, 0, 0}, Vec3{1.5, 1, 0});
    EXPECT_EQ(dg, (Vec3{1.5, 0, 0}));
}

TEST(BoxTreeGeometry, FilledTriangle2DAndRays)
{
    Triangle<2> const t{{0, 0}, {4, 0}, {0, 4}};
    using T = TraitsOf<Triangle<2>>;
    EXPECT_EQ(T::closestPoint(t, {1, 1}), (Vec2{1, 1}));  // inside: exact
    EXPECT_EQ(T::closestPoint(t, {2, -2}), (Vec2{2, 0}));
    EXPECT_EQ(T::intersectRay(t, {1, 1}, {1, 0}), 0.0);  // starts inside
    EXPECT_EQ(T::intersectRay(t, {-2, 1}, {1, 0}), 2.0);
    EXPECT_FALSE(T::intersectRay(t, {-2, 1}, {-1, 0}).has_value());

    EXPECT_EQ(geom::intersectRaySegment({0, -1}, {0, 1}, {-3, 0}, {1, 0}), 3.0);
    EXPECT_FALSE(geom::intersectRaySegment({0, -1}, {0, 1}, {-3, 0}, {0, 1}).has_value());  // parallel
    EXPECT_EQ(geom::intersectRayTriangle({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0.2, 0.2, 5}, {0, 0, -1}), 5.0);
    EXPECT_FALSE(geom::intersectRayTriangle({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {2, 2, 5}, {0, 0, -1}).has_value());

    Box<3> const box{{0, 0, 0}, {1, 1, 1}};
    // zero direction components must not produce NaN misses
    auto const h = box.intersectRay({0.5, 0.5, -2}, {0, 0, 1}, 0.0, 100.0);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(h->first, 2.0);
    EXPECT_EQ(h->second, 3.0);
    EXPECT_FALSE(box.intersectRay({1.5, 0.5, -2}, {0, 0, 1}, 0.0, 100.0).has_value());
    EXPECT_EQ(box.distanceSqr({2, 0.5, 3}), 5.0);
}

TEST(BoxTree, EmptyTree)
{
    libkdtree::ClosestPoint3D<> tree;
    EXPECT_TRUE(tree.empty());
    EXPECT_FALSE(tree.closestPoint(Vec3{0, 0, 0}).has_value());
    EXPECT_FALSE(tree.closestPointSignedDistance(Vec3{0, 0, 0}).has_value());
    EXPECT_FALSE(tree.intersectRayDistance(Vec3{0, 0, 0}, Vec3{1, 0, 0}).has_value());
    EXPECT_EQ(tree.distance(Vec3{0, 0, 0}), std::numeric_limits<double>::infinity());
    EXPECT_EQ(tree.distance(Vec3{0, 0, 0}, 2.5), 2.5);
    EXPECT_EQ(tree.distanceSqr(Vec3{0, 0, 0}, 2.5), 6.25);
    EXPECT_TRUE(tree.within(Vec3{0, 0, 0}, 10).empty());
}

TEST(BoxTree, TriangleSoupMedianMatchesBruteForce)
{
    auto const tris = randomTriangles(6000, 100.0, 6.0, 31);
    BoxTree<Triangle<3>> tree(tris);  // Median by default
    EXPECT_EQ(tree.size(), tris.size());
    EXPECT_LE(tree.depth(), 13u);  // ceil(log2(6000 / 4)) + 1
    crossCheckTriangles(tree, tris, 250, 100.0, 32);
}

TEST(BoxTree, TriangleSoupWeightedMatchesBruteForce)
{
    auto const tris = randomTriangles(6000, 100.0, 6.0, 33);
    libkdtree::ClosestPoint3D<> tree(tris, 64, SplitMetric::Weighted);
    EXPECT_EQ(tree.getSplitMetric(), SplitMetric::Weighted);
    crossCheckTriangles(tree, tris, 250, 100.0, 34);
}

TEST(BoxTree, LargeMixedScaleScene)
{
    // a few huge triangles over many tiny ones: stresses overlap-heavy boxes
    auto tris = randomTriangles(40000, 50.0, 0.5, 35);
    auto big = randomTriangles(40, 50.0, 60.0, 36);
    tris.insert(tris.end(), big.begin(), big.end());
    for (auto metric : {SplitMetric::Median, SplitMetric::Weighted}) {
        BoxTree<Triangle<3>> tree(tris, 64, metric);
        EXPECT_LE(tree.depth(), BoxTree<Triangle<3>>::kMaxDepthLimit);
        crossCheckTriangles(tree, tris, 60, 50.0, 37);
    }
}

TEST(BoxTree, SphereSignedDistance)
{
    auto const tris = sphereMesh(1.0, 24, 48);
    libkdtree::ClosestPoint3D<> tree(tris);
    std::mt19937 rng(38);
    std::uniform_real_distribution<double> inside(0.0, 0.85), outside(1.15, 4.0);
    for (int i = 0; i < 400; ++i) {
        Vec3 const dir = randomDirection(rng);
        Vec3 const pin = geom::mul(dir, inside(rng));
        Vec3 const pout = geom::mul(dir, outside(rng));
        Vec3 cp{};
        Triangle<3> const* tri = nullptr;
        auto const sin = tree.closestPointSignedDistance(pin, cp, tri);
        ASSERT_TRUE(sin.has_value());
        EXPECT_LT(*sin, 0) << i;
        EXPECT_DOUBLE_EQ(-*sin, std::sqrt(bruteNearest(tris, pin).first));
        auto const sout = tree.closestPointSignedDistance(pout);
        ASSERT_TRUE(sout.has_value());
        EXPECT_GT(*sout, 0) << i;
        // every ray from inside leaves through the surface at roughly radius 1
        auto const hit = tree.intersectRayDistance(pin, dir);
        ASSERT_TRUE(hit.has_value());
        EXPECT_NEAR(geom::length(pin) + *hit, 1.0, 0.02);
    }
    // bounded signed query misses beyond the bound
    EXPECT_FALSE(tree.closestPointSignedDistance(Vec3{0, 0, 3}, 1.5).has_value());
    EXPECT_NEAR(*tree.closestPointSignedDistance(Vec3{0, 0, 3}, 2.5), 2.0, 1e-9);
}

TEST(BoxTree, Polygon2DSignedDistance)
{
    // counter-clockwise circle: Segment<2> normals point outward
    std::vector<Segment<2>> segs;
    int const n = 720;
    for (int i = 0; i < n; ++i) {
        double const a0 = 2 * std::numbers::pi * i / n, a1 = 2 * std::numbers::pi * (i + 1) / n;
        segs.push_back({{5 * std::cos(a0), 5 * std::sin(a0)}, {5 * std::cos(a1), 5 * std::sin(a1)}});
    }
    libkdtree::ClosestPoint2D<> tree(segs);
    std::mt19937 rng(39);
    std::uniform_real_distribution<double> ang(0, 2 * std::numbers::pi), rad(0, 10);
    for (int i = 0; i < 2000; ++i) {
        double const a = ang(rng), r = rad(rng);
        Vec2 const p{r * std::cos(a), r * std::sin(a)};
        auto const [bestSq, idx] = bruteNearest(segs, p);
        auto const sd = tree.closestPointSignedDistance(p);
        ASSERT_TRUE(sd.has_value());
        EXPECT_DOUBLE_EQ(std::abs(*sd), std::sqrt(bestSq));
        if (std::abs(r - 5) > 0.01) EXPECT_EQ(*sd < 0, r < 5) << "r=" << r;
        // rays from the origin always hit the circle at distance ~5
        EXPECT_NEAR(*tree.intersectRayDistance(Vec2{0, 0}, Vec2{std::cos(a), std::sin(a)}), 5.0, 1e-3);
    }
}

TEST(BoxTree, Mixed2DVariantMatchesBruteForce)
{
    using Prim = std::variant<Segment<2>, Triangle<2>>;
    static_assert(!BoxTree<Prim>::kHasNormal, "filled triangles have no normal");
    static_assert(BoxTree<Prim>::kHasRay);
    std::mt19937 rng(40);
    std::uniform_real_distribution<double> c(-200, 200), s(-3, 3);
    std::vector<Prim> prims;
    for (int i = 0; i < 8000; ++i) {
        Vec2 const p{c(rng), c(rng)};
        auto off = [&] { return Vec2{p[0] + s(rng), p[1] + s(rng)}; };
        if (i % 3 == 0)
            prims.emplace_back(Triangle<2>{p, off(), off()});
        else
            prims.emplace_back(Segment<2>{p, off()});
    }
    for (auto metric : {SplitMetric::Median, SplitMetric::Weighted}) {
        BoxTree<Prim> tree(prims, 64, metric);
        std::mt19937 q(41);
        std::uniform_real_distribution<double> u(-220, 220);
        std::uniform_real_distribution<double> ang(0, 2 * std::numbers::pi);
        for (int i = 0; i < 300; ++i) {
            Vec2 const p{u(q), u(q)};
            auto const [bestSq, idx] = bruteNearest(prims, p);
            Prim const* hit = nullptr;
            auto const d = tree.closestPoint(p, hit);
            ASSERT_TRUE(d.has_value());
            ASSERT_DOUBLE_EQ(*d, std::sqrt(bestSq));
            ASSERT_EQ(tree.within(p, std::sqrt(bestSq) + 2).size(),
                      static_cast<std::size_t>(std::count_if(prims.begin(), prims.end(), [&](Prim const& pr) {
                          return geom::distanceSqr(TraitsOf<Prim>::closestPoint(pr, p), p) <=
                                 std::pow(std::sqrt(bestSq) + 2, 2);
                      })));
            double const a = ang(q);
            Vec2 const dir{std::cos(a), std::sin(a)};
            auto const t = bruteRay(prims, p, dir, std::numeric_limits<double>::infinity());
            auto const rd = tree.intersectRayDistance(p, dir);
            ASSERT_EQ(rd.has_value(), t.has_value());
            if (rd) ASSERT_DOUBLE_EQ(*rd, *t);
        }
        // a point inside a filled triangle is at distance 0
        auto const& tri = std::get<Triangle<2>>(prims[0]);
        Vec2 const centroid = geom::mul(geom::add(geom::add(tri.a, tri.b), tri.c), 1.0 / 3);
        EXPECT_EQ(tree.distance(centroid), 0.0);
    }
}

TEST(BoxTree, CustomPrimitiveViaTraits)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> c(-100, 100), r(0.1, 4);
    std::vector<Disk> disks(5000);
    for (auto& d : disks) d = {{c(rng), c(rng)}, r(rng)};
    BoxTree<Disk> tree(disks);
    static_assert(!BoxTree<Disk>::kHasRay && !BoxTree<Disk>::kHasNormal);
    for (int i = 0; i < 500; ++i) {
        Vec2 const p{c(rng) * 1.2, c(rng) * 1.2};
        Vec2 cp{};
        Disk const* hit = nullptr;
        auto const d2 = tree.closestPointSqr(p, cp, hit);
        ASSERT_EQ(*d2, bruteNearest(disks, p).first);
        if (*d2 == 0) {
            EXPECT_LE(geom::distanceSqr(p, hit->c), hit->r * hit->r);
        }
    }
}

TEST(BoxTree, BoxPrimitivesRays)
{
    std::mt19937 rng(43);
    std::uniform_real_distribution<double> c(-50, 50), e(0.1, 3);
    std::vector<Box<3>> boxes(3000);
    for (auto& b : boxes) {
        Vec3 const p{c(rng), c(rng), c(rng)};
        b = Box<3>::fromPoints(p, {p[0] + e(rng), p[1] + e(rng), p[2] + e(rng)});
    }
    BoxTree<Box<3>> tree(boxes, 64, SplitMetric::Weighted);
    for (int i = 0; i < 400; ++i) {
        Vec3 const o{c(rng), c(rng), c(rng)};
        Vec3 const d = randomDirection(rng);
        auto const t = bruteRay(boxes, o, d, std::numeric_limits<double>::infinity());
        auto const rd = tree.intersectRayDistance(o, d);
        ASSERT_EQ(rd.has_value(), t.has_value());
        if (rd) ASSERT_EQ(*rd, *t * geom::length(d));
        ASSERT_EQ(tree.distanceSqr(o), bruteNearest(boxes, o).first);
    }
}

TEST(BoxTree, DegenerateInputs)
{
    // identical primitives: every centroid coincides, so the root stays one big leaf
    std::vector<Triangle<3>> same(300, Triangle<3>{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
    BoxTree<Triangle<3>> t1(same);
    EXPECT_EQ(t1.nodeCount(), 1u);
    EXPECT_DOUBLE_EQ(*t1.closestPoint(Vec3{0.2, 0.2, 2}), 2.0);
    EXPECT_EQ(t1.within(Vec3{0.2, 0.2, 2}, 2.0).size(), 300u);

    // perfectly flat scene (z = 0) with SAH
    auto flat = randomTriangles(3000, 40.0, 2.0, 44);
    for (auto& t : flat) t.a[2] = t.b[2] = t.c[2] = 0;
    BoxTree<Triangle<3>> t2(flat, 64, SplitMetric::Weighted);
    crossCheckTriangles(t2, flat, 80, 40.0, 45);

    // depth cap forces large leaves; queries stay exact
    auto const tris = randomTriangles(2000, 40.0, 2.0, 46);
    BoxTree<Triangle<3>> t3(tris, 3);
    EXPECT_EQ(t3.depth(), 3u);
    EXPECT_EQ(t3.nodeCount(), 7u);
    crossCheckTriangles(t3, tris, 50, 40.0, 47);

    // a single primitive
    BoxTree<Segment<3>> t4(std::vector<Segment<3>>{{{0, 0, 0}, {0, 0, 10}}});
    EXPECT_DOUBLE_EQ(*t4.closestPoint(Vec3{3, 4, 5}), 5.0);
    EXPECT_FALSE(BoxTree<Segment<3>>::kHasRay);  // 3D segments have no ray support
}

TEST(BoxTree, AppendFinishAndIndices)
{
    auto const tris = randomTriangles(1500, 30.0, 3.0, 48);
    BoxTree<Triangle<3>> tree(SplitMetric::Median);
    for (std::size_t i = 0; i < 1000; ++i) tree.append(tris[i]);
    EXPECT_THROW((void)tree.closestPoint(Vec3{0, 0, 0}), std::logic_error);
    tree.finish();
    for (std::size_t i = 1000; i < tris.size(); ++i) tree.emplace(tris[i]);
    tree.finish();
    ASSERT_EQ(tree.size(), tris.size());

    // insertion indices survive the reordering of both builds
    for (std::size_t i = 0; i < tris.size(); ++i) {
        auto const& p = tree.primitive(i);
        EXPECT_EQ(p.a, tris[i].a);
        EXPECT_EQ(tree.index(&p), i);
    }
    crossCheckTriangles(tree, tris, 80, 30.0, 49);

    // rebuild with a different metric, then reset
    tree.build(std::vector<Triangle<3>>(tris.begin(), tris.begin() + 10), SplitMetric::Weighted);
    EXPECT_EQ(tree.size(), 10u);
    tree.reset();
    EXPECT_TRUE(tree.empty());
    EXPECT_FALSE(tree.closestPoint(Vec3{0, 0, 0}).has_value());
}
