#ifndef TACHYON_CRYPTO_COMMITMENTS_POLYNOMIAL_OPENINGS_H_
#define TACHYON_CRYPTO_COMMITMENTS_POLYNOMIAL_OPENINGS_H_

#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/ref.h"
#include "tachyon/math/polynomials/univariate/lagrange_interpolation.h"

namespace tachyon::crypto {

// A single polynomial with a single opening.
template <typename Poly>
struct PolynomialOpening {
  using Field = typename Poly::Field;
  using Point = typename Poly::Point;

  // P₀
  base::DeepRef<const Poly> poly;
  // x₀
  base::DeepRef<const Point> point;
  // P₀(x₀)
  Field opening;

  PolynomialOpening() = default;
  PolynomialOpening(base::DeepRef<const Poly> poly,
                    base::DeepRef<const Point> point, Field&& opening)
      : poly(poly), point(point), opening(std::move(opening)) {}
};

// A single polynomial with multi openings.
template <typename Poly>
struct PolynomialOpenings {
  using Field = typename Poly::Field;

  // P₀
  base::DeepRef<const Poly> poly;
  // [P₀(x₀), P₀(x₁), P₀(x₂)]
  std::vector<Field> openings;

  PolynomialOpenings() = default;
  PolynomialOpenings(base::DeepRef<const Poly> poly,
                     std::vector<Field>&& openings)
      : poly(poly), openings(std::move(openings)) {}
};

// Multi polynomials with multi openings grouped by shared points.
template <typename Poly>
struct GroupedPolynomialOpenings {
  using Field = typename Poly::Field;
  using Point = typename Poly::Point;

  // [{P₀, [P₀(x₀), P₀(x₁), P₀(x₂)]}, {P₁, [P₁(x₀), P₁(x₁), P₁(x₂)]}]
  std::vector<PolynomialOpenings<Poly>> poly_openings_vec;
  // [x₀, x₁, x₂]
  std::vector<base::DeepRef<const Point>> points;

  GroupedPolynomialOpenings() = default;
  GroupedPolynomialOpenings(
      std::vector<PolynomialOpenings<Poly>>&& poly_openings_vec,
      std::vector<base::DeepRef<const Point>>&& points)
      : poly_openings_vec(std::move(poly_openings_vec)),
        points(std::move(points)) {}

  // Create a low degree extension that is a linear combination with a set of
  // low degree extensions based on every |poly_openings_vec.openings| and
  // shared |points|.
  Poly CreateCombinedLowDegreeExtensions(
      const Field& r, std::vector<Poly>& low_degree_extensions) const {
    std::vector<Point> owned_points = CreateOwnedPoints();
    low_degree_extensions = CreateLowDegreeExtensions(owned_points);
    return CombineLowDegreeExtensions(r, owned_points, low_degree_extensions);
  }

 private:
  FRIEND_TEST(PolynomialOpeningsTest, CreateCombinedLowDegreeExtensions);

  // TODO(chokobole): Since |CreateLowDegreeExtensions()| and
  // |CombineLowDegreeExtensions()| internally access to |Point| with an
  // indexing operator, if we use a vector of |base::DeepRef<const Point>| it
  // can't access as we expect.
  std::vector<Point> CreateOwnedPoints() const {
    return base::Map(points,
                     [](const base::DeepRef<const Point>& p) { return *p; });
  }

  // Create a set of low degree extensions based on every
  // |poly_openings.openings| and shared |points|.
  std::vector<Poly> CreateLowDegreeExtensions(
      const std::vector<Point>& owned_points) const {
    return base::Map(
        poly_openings_vec,
        [&owned_points](const PolynomialOpenings<Poly>& poly_openings) {
          Poly low_degree_extension;
          CHECK(math::LagrangeInterpolate(owned_points, poly_openings.openings,
                                          &low_degree_extension));
          return low_degree_extension;
        });
  }

  Poly CombineLowDegreeExtensions(
      const Field& r, const std::vector<Point>& owned_points,
      const std::vector<Poly>& low_degree_extensions) const {
    // numerators: [P₀(X) - R₀(X), P₁(X) - R₁(X), P₂(X) - R₂(X)]
    std::vector<Poly> numerators =
        base::Map(poly_openings_vec,
                  [&low_degree_extensions](
                      size_t i, const PolynomialOpenings<Poly>& poly_openings) {
                    return *poly_openings.poly - low_degree_extensions[i];
                  });

    // Combine numerator polynomials with powers of |r|.
    // N(X) = (P₀(X) - R₀(X)) + r(P₁(X) - R₁(X)) + r²(P₂(X) - R₂(X))
    Poly& n = Poly::LinearizeInPlace(numerators, r);

    // Divide combined polynomial by vanishing polynomial of evaluation points.
    // H(X) = N(X) / (X - x₀)(X - x₁)(X - x₂)
    Poly vanishing_poly = Poly::FromRoots(owned_points);
    return n /= vanishing_poly;
  }
};

template <typename Poly>
class PolynomialOpeningGrouper {
 public:
  using Field = typename Poly::Field;
  using Point = typename Poly::Point;
  using PolyDeepRef = base::DeepRef<const Poly>;
  using PointDeepRef = base::DeepRef<const Point>;

  const std::vector<GroupedPolynomialOpenings<Poly>>&
  grouped_poly_openings_vec() const {
    return grouped_poly_openings_vec_;
  }
  const absl::btree_set<PointDeepRef>& super_point_set() const {
    return super_point_set_;
  }

  void GroupByPolyAndPoints(
      const std::vector<PolynomialOpening<Poly>>& poly_openings) {
    // Group |poly_openings| by polynomial.
    // {P₀, [x₀, x₁, x₂]}
    // {P₁, [x₀, x₁, x₂]}
    // {P₂, [x₀, x₁, x₂]}
    // {P₃, [x₂, x₃]}
    // {P₄, [x₄]}
    absl::flat_hash_map<PolyDeepRef, absl::btree_set<PointDeepRef>>
        poly_openings_grouped_by_poly = GroupByPoly(poly_openings);

    // Group |poly_openings_grouped_by_poly| by points.
    // [x₀, x₁, x₂]: [P₀, P₁, P₂]
    // [x₂, x₃]: [P₃]
    // [x₄]: [P₄]
    absl::flat_hash_map<absl::btree_set<PointDeepRef>, std::vector<PolyDeepRef>>
        poly_openings_grouped_by_poly_and_points =
            GroupByPoints(poly_openings_grouped_by_poly);

    // Construct opening sets from the flattened map.
    // Each contains oracles and the corresponding evaluation points.
    // grouped_poly_openings_vec_[0]: {[P₀, P₁, P₂], [x₀, x₁, x₂]}
    // grouped_poly_openings_vec_[1]: {[P₃], [x₂, x₃]}
    // grouped_poly_openings_vec_[2]: {[P₄], [x₄]}
    CreateMultiPolynomialOpenings(poly_openings,
                                  poly_openings_grouped_by_poly_and_points);
  }

 private:
  FRIEND_TEST(PolynomialOpeningsTest, GroupByPolyAndPoints);

  absl::flat_hash_map<PolyDeepRef, absl::btree_set<PointDeepRef>> GroupByPoly(
      const std::vector<PolynomialOpening<Poly>>& poly_openings) {
    absl::flat_hash_map<PolyDeepRef, absl::btree_set<PointDeepRef>> ret;
    for (const PolynomialOpening<Poly>& poly_opening : poly_openings) {
      super_point_set_.insert(poly_opening.point);
      ret[poly_opening.poly].insert(poly_opening.point);
    }
    return ret;
  }

  absl::flat_hash_map<absl::btree_set<PointDeepRef>, std::vector<PolyDeepRef>>
  GroupByPoints(
      const absl::flat_hash_map<PolyDeepRef, absl::btree_set<PointDeepRef>>&
          poly_openings_grouped_by_poly) {
    absl::flat_hash_map<absl::btree_set<PointDeepRef>, std::vector<PolyDeepRef>>
        ret;
    for (const auto& [poly, points] : poly_openings_grouped_by_poly) {
      ret[points].push_back(poly);
    }
    return ret;
  }

  void CreateMultiPolynomialOpenings(
      const std::vector<PolynomialOpening<Poly>>& poly_openings,
      const absl::flat_hash_map<absl::btree_set<PointDeepRef>,
                                std::vector<PolyDeepRef>>&
          poly_openings_grouped_by_poly_and_points) {
    grouped_poly_openings_vec_.reserve(
        poly_openings_grouped_by_poly_and_points.size());

    for (const auto& [points, polys] :
         poly_openings_grouped_by_poly_and_points) {
      std::vector<PointDeepRef> points_vec(points.begin(), points.end());

      std::vector<PolynomialOpenings<Poly>> poly_openings_vec =
          base::Map(polys, [&poly_openings, &points_vec](PolyDeepRef poly) {
            std::vector<Field> openings = base::Map(
                points_vec, [poly, &poly_openings](PointDeepRef point) {
                  return GetOpeningFromPolyOpenings(poly_openings, poly, point);
                });
            return PolynomialOpenings<Poly>(poly, std::move(openings));
          });

      grouped_poly_openings_vec_.push_back(GroupedPolynomialOpenings<Poly>(
          std::move(poly_openings_vec), std::move(points_vec)));
    }
  }

  static Field GetOpeningFromPolyOpenings(
      const std::vector<PolynomialOpening<Poly>>& poly_openings,
      PolyDeepRef poly, PointDeepRef point) {
    auto it = std::find_if(
        poly_openings.begin(), poly_openings.end(),
        [poly, point](const PolynomialOpening<Poly>& poly_opening) {
          return poly_opening.poly == poly && poly_opening.point == point;
        });
    CHECK(it != poly_openings.end());
    return it->opening;
  }

  // |grouped_poly_openings_vec_| is a list of
  // |GroupedPolynomialOpenings|, which is the result of list of
  // |PolynomialOpening| grouped by poly and points.
  // {[P₀, P₁, P₂], [x₀, x₁, x₂]}
  // {[P₃], [x₂, x₃]}
  // {[P₄], [x₄]}
  std::vector<GroupedPolynomialOpenings<Poly>> grouped_poly_openings_vec_;
  // |super_point_set_| is all the points that appear in opening.
  // [x₀, x₁, x₂, x₃, x₄]
  absl::btree_set<base::DeepRef<const Point>> super_point_set_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_POLYNOMIAL_OPENINGS_H_
