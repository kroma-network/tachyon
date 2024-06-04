#ifndef TACHYON_CRYPTO_COMMITMENTS_POLYNOMIAL_OPENINGS_H_
#define TACHYON_CRYPTO_COMMITMENTS_POLYNOMIAL_OPENINGS_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/strings/substitute.h"
#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/base/optional.h"
#include "tachyon/base/ref.h"
#include "tachyon/math/polynomials/univariate/lagrange_interpolation.h"

namespace tachyon::crypto {

// A single polynomial oracle with a single opening.
// The type of polynomial oracle can be either |Poly| in |CreateOpeningProof()|
// and |Commitment| in |VerifyOpeningProof()|.
template <typename Poly, typename PolyOracle = Poly>
struct PolynomialOpening {
  using Field = typename Poly::Field;
  using Point = typename Poly::Point;

  // polynomial Pᵢ or commitment Cᵢ
  base::Ref<const PolyOracle> poly_oracle;
  // xᵢ
  Point point;
  // Pᵢ(xᵢ)
  Field opening;

  PolynomialOpening() = default;
  PolynomialOpening(base::Ref<const PolyOracle> poly_oracle, const Point& point,
                    const Field& opening)
      : poly_oracle(poly_oracle), point(point), opening(opening) {}
  PolynomialOpening(base::Ref<const PolyOracle> poly_oracle, Point&& point,
                    Field&& opening)
      : poly_oracle(poly_oracle),
        point(std::move(point)),
        opening(std::move(opening)) {}

  bool operator==(const PolynomialOpening& other) const {
    return poly_oracle == other.poly_oracle && point == other.point &&
           opening == other.opening;
  }
  bool operator!=(const PolynomialOpening& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    if constexpr (std::is_same_v<Poly, PolyOracle>) {
      return absl::Substitute("{poly: $0, point: $1, opening: $2}",
                              poly_oracle->ToString(), point->ToString(),
                              opening.ToString());
    } else {
      return absl::Substitute("{commitment: $0, point: $1, opening: $2}",
                              poly_oracle->ToString(), point->ToString(),
                              opening.ToString());
    }
  }

  std::string ToHexString(bool pad_zero = false) const {
    if constexpr (std::is_same_v<Poly, PolyOracle>) {
      return absl::Substitute(
          "{poly: $0, point: $1, opening: $2}", poly_oracle->ToString(),
          point->ToHexString(pad_zero), opening.ToHexString(pad_zero));
    } else {
      return absl::Substitute("{commitment: $0, point: $1, opening: $2}",
                              poly_oracle->ToHexString(pad_zero),
                              point->ToHexString(pad_zero),
                              opening.ToHexString(pad_zero));
    }
  }
};

// A single polynomial oracle with multi openings.
template <typename Poly, typename PolyOracle = Poly>
struct PolynomialOpenings {
  using Field = typename Poly::Field;

  // polynomial Pᵢ or commitment Cᵢ
  base::Ref<const PolyOracle> poly_oracle;
  // [Pᵢ(x₀), Pᵢ(x₁), Pᵢ(x₂)]
  std::vector<Field> openings;

  PolynomialOpenings() = default;
  PolynomialOpenings(base::Ref<const PolyOracle> poly_oracle,
                     std::vector<Field>&& openings)
      : poly_oracle(poly_oracle), openings(std::move(openings)) {}

  bool operator==(const PolynomialOpenings& other) const {
    return poly_oracle == other.poly_oracle && openings == other.openings;
  }
  bool operator!=(const PolynomialOpenings& other) const {
    return !operator==(other);
  }
};

// Multi polynomial oracles with multi openings grouped by shared points.
template <typename Poly, typename PolyOracle = Poly>
struct GroupedPolynomialOpenings {
  using Field = typename Poly::Field;
  using Point = typename Poly::Point;

  // [{P₀, [P₀(x₀), P₀(x₁), P₀(x₂)]}, {P₁, [P₁(x₀), P₁(x₁), P₁(x₂)]}]
  std::vector<PolynomialOpenings<Poly, PolyOracle>> poly_openings_vec;
  // [x₀, x₁, x₂]
  std::vector<Point> points;

  GroupedPolynomialOpenings() = default;
  GroupedPolynomialOpenings(
      std::vector<PolynomialOpenings<Poly, PolyOracle>>&& poly_openings_vec,
      std::vector<Point>&& points)
      : poly_openings_vec(std::move(poly_openings_vec)),
        points(std::move(points)) {}

  // Create a low degree extension that is a linear combination with a set of
  // low degree extensions based on every |poly_openings_vec.openings| and
  // shared |points|.
  Poly CreateCombinedLowDegreeExtensions(
      const Field& r, std::vector<Poly>& low_degree_extensions) const {
    low_degree_extensions = CreateLowDegreeExtensions();
    return CombineLowDegreeExtensions(r, low_degree_extensions);
  }

  bool operator==(const GroupedPolynomialOpenings& other) const {
    return poly_openings_vec == other.poly_openings_vec &&
           points == other.points;
  }
  bool operator!=(const GroupedPolynomialOpenings& other) const {
    return !operator==(other);
  }

 private:
  FRIEND_TEST(PolynomialOpeningsTest, CreateCombinedLowDegreeExtensions);

  // Create a set of low degree extensions based on every
  // |poly_openings.openings| and shared |points|.
  std::vector<Poly> CreateLowDegreeExtensions() const {
    return base::Map(
        poly_openings_vec,
        [this](const PolynomialOpenings<Poly>& poly_openings) {
          Poly low_degree_extension;
          CHECK(math::LagrangeInterpolate(points, poly_openings.openings,
                                          &low_degree_extension));
          return low_degree_extension;
        });
  }

  Poly CombineLowDegreeExtensions(
      const Field& r, const std::vector<Poly>& low_degree_extensions) const {
    // numerators: [P₀(X) - R₀(X), P₁(X) - R₁(X), P₂(X) - R₂(X)]
    std::vector<Poly> numerators(low_degree_extensions.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < low_degree_extensions.size(); ++i) {
      numerators[i] =
          *poly_openings_vec[i].poly_oracle - low_degree_extensions[i];
    }

    // Combine numerator polynomials with powers of |r|.
    // N(X) = (P₀(X) - R₀(X)) + r(P₁(X) - R₁(X)) + r²(P₂(X) - R₂(X))
    Poly& n = Poly::template LinearCombinationInPlace</*forward=*/false>(
        numerators, r);

    // Divide combined polynomial by vanishing polynomial of evaluation points.
    // H(X) = N(X) / (X - x₀)(X - x₁)(X - x₂)
    Poly vanishing_poly = Poly::FromRoots(points);
    return unwrap<Poly>(n / vanishing_poly);
  }
};

template <typename Poly, typename PolyOracle = Poly>
class PolynomialOpeningGrouper {
 public:
  using Field = typename Poly::Field;
  using Point = typename Poly::Point;
  using PolyOracleRef = base::Ref<const PolyOracle>;

  const std::vector<GroupedPolynomialOpenings<Poly, PolyOracle>>&
  grouped_poly_openings_vec() const {
    return grouped_poly_openings_vec_;
  }
  const absl::btree_set<Point>& super_point_set() const {
    return super_point_set_;
  }

  // Used by GWC.
  void GroupBySinglePoint(
      const std::vector<PolynomialOpening<Poly, PolyOracle>>& poly_openings) {
    for (const PolynomialOpening<Poly, PolyOracle>& poly_opening :
         poly_openings) {
      super_point_set_.insert(poly_opening.point);

      auto it = std::find_if(
          grouped_poly_openings_vec_.begin(), grouped_poly_openings_vec_.end(),
          [&poly_opening](const GroupedPolynomialOpenings<Poly, PolyOracle>&
                              grouped_poly_openings) {
            return grouped_poly_openings.points[0] == poly_opening.point;
          });

      PolynomialOpenings<Poly, PolyOracle> poly_openings(
          poly_opening.poly_oracle, {poly_opening.opening});
      if (it != grouped_poly_openings_vec_.end()) {
        it->poly_openings_vec.push_back(std::move(poly_openings));
      } else {
        grouped_poly_openings_vec_.push_back(
            {{std::move(poly_openings)}, {poly_opening.point}});
      }
    }
  }

  // Used by SHPlonk.
  void GroupByPolyOracleAndPoints(
      const std::vector<PolynomialOpening<Poly, PolyOracle>>& poly_openings) {
    // Group |poly_openings| by polynomial.
    // grouped_poly_oracle_pairs[0]: {P₀, [x₀, x₁, x₂]}
    // grouped_poly_oracle_pairs[1]: {P₁, [x₀, x₁, x₂]}
    // grouped_poly_oracle_pairs[2]: {P₂, [x₀, x₁, x₂]}
    // grouped_poly_oracle_pairs[3]: {P₃, [x₂, x₃]}
    // grouped_poly_oracle_pairs[4]: {P₄, [x₄]}
    std::vector<GroupedPolyOraclePair> grouped_poly_oracle_pairs =
        GroupByPolyOracle(poly_openings);

    // Group |grouped_poly_oracle_pairs| by points.
    // grouped_point_pairs[0]: {[x₀, x₁, x₂], [P₀, P₁, P₂]}
    // grouped_point_pairs[1]: {[x₂, x₃], [P₃]}
    // grouped_point_pairs[2]: {[x₄], [P₄]}
    std::vector<GroupedPointPair> grouped_point_pairs =
        GroupByPoints(grouped_poly_oracle_pairs);

    // Construct openings vectors from the |grouped_poly_oracle_pairs|.
    // Each contains oracles and the corresponding evaluation points.
    // grouped_poly_openings_vec_[0]: {[P₀, P₁, P₂], [x₀, x₁, x₂]}
    // grouped_poly_openings_vec_[1]: {[P₃], [x₂, x₃]}
    // grouped_poly_openings_vec_[2]: {[P₄], [x₄]}
    CreateMultiPolynomialOpenings(poly_openings, grouped_point_pairs);
  }

 private:
  FRIEND_TEST(PolynomialOpeningsTest, GroupByPolyOracleAndPoints);

  struct GroupedPolyOraclePair {
    PolyOracleRef poly_oracle;
    absl::btree_set<Point> points;
  };

  std::vector<GroupedPolyOraclePair> GroupByPolyOracle(
      const std::vector<PolynomialOpening<Poly, PolyOracle>>& poly_openings) {
    std::vector<GroupedPolyOraclePair> ret;
    ret.reserve(poly_openings.size());
    for (const PolynomialOpening<Poly, PolyOracle>& poly_opening :
         poly_openings) {
      super_point_set_.insert(poly_opening.point);

      auto it = std::find_if(
          ret.begin(), ret.end(),
          [&poly_opening](
              const GroupedPolyOraclePair& poly_oracle_grouped_pair) {
            return poly_oracle_grouped_pair.poly_oracle ==
                   poly_opening.poly_oracle;
          });

      if (it != ret.end()) {
        it->points.insert(poly_opening.point);
      } else {
        GroupedPolyOraclePair new_pair;
        new_pair.poly_oracle = poly_opening.poly_oracle;
        new_pair.points.insert(poly_opening.point);
        ret.push_back(std::move(new_pair));
      }
    }
    return ret;
  }

  struct GroupedPointPair {
    absl::btree_set<Point> points;
    std::vector<PolyOracleRef> polys;
  };

  std::vector<GroupedPointPair> GroupByPoints(
      const std::vector<GroupedPolyOraclePair>& grouped_poly_oracle_pairs) {
    std::vector<GroupedPointPair> ret;
    ret.reserve(grouped_poly_oracle_pairs.size());
    for (const GroupedPolyOraclePair& grouped_poly_oracle_pair :
         grouped_poly_oracle_pairs) {
      auto it = std::find_if(ret.begin(), ret.end(),
                             [&grouped_poly_oracle_pair](
                                 const GroupedPointPair& point_grouped_pair) {
                               return point_grouped_pair.points ==
                                      grouped_poly_oracle_pair.points;
                             });

      if (it != ret.end()) {
        it->polys.push_back(grouped_poly_oracle_pair.poly_oracle);
      } else {
        GroupedPointPair new_pair;
        new_pair.points = grouped_poly_oracle_pair.points;
        new_pair.polys.push_back(grouped_poly_oracle_pair.poly_oracle);
        ret.push_back(std::move(new_pair));
      }
    }
    return ret;
  }

  void CreateMultiPolynomialOpenings(
      const std::vector<PolynomialOpening<Poly, PolyOracle>>& poly_openings,
      const std::vector<GroupedPointPair>& grouped_point_pairs) {
    grouped_poly_openings_vec_.reserve(grouped_point_pairs.size());

    for (const GroupedPointPair& grouped_point_pair : grouped_point_pairs) {
      const std::vector<PolyOracleRef>& polys = grouped_point_pair.polys;
      std::vector<Point> points(grouped_point_pair.points.begin(),
                                grouped_point_pair.points.end());
      std::vector<PolynomialOpenings<Poly, PolyOracle>> poly_openings_vec =
          base::Map(
              polys, [&poly_openings, &points](PolyOracleRef poly_oracle) {
                std::vector<Field> openings = base::Map(
                    points, [poly_oracle, &poly_openings](const Point& point) {
                      return GetOpeningFromPolyOpenings(poly_openings,
                                                        poly_oracle, point);
                    });
                return PolynomialOpenings<Poly, PolyOracle>(
                    poly_oracle, std::move(openings));
              });

      grouped_poly_openings_vec_.push_back(
          GroupedPolynomialOpenings<Poly, PolyOracle>(
              std::move(poly_openings_vec), std::move(points)));
    }
  }

  static Field GetOpeningFromPolyOpenings(
      const std::vector<PolynomialOpening<Poly, PolyOracle>>& poly_openings,
      PolyOracleRef poly_oracle, const Point& point) {
    auto it = std::find_if(
        poly_openings.begin(), poly_openings.end(),
        [poly_oracle,
         point](const PolynomialOpening<Poly, PolyOracle>& poly_opening) {
          return poly_opening.poly_oracle == poly_oracle &&
                 poly_opening.point == point;
        });
    CHECK(it != poly_openings.end());
    return it->opening;
  }

  // |grouped_poly_openings_vec_| is a list of
  // |GroupedPolynomialOpenings|, which is the result of list of
  // |PolynomialOpening| grouped by poly_oracle and points.
  // {[P₀, P₁, P₂], [x₀, x₁, x₂]}
  // {[P₃], [x₂, x₃]}
  // {[P₄], [x₄]}
  std::vector<GroupedPolynomialOpenings<Poly, PolyOracle>>
      grouped_poly_openings_vec_;
  // |super_point_set_| is all the points that appear in opening.
  // [x₀, x₁, x₂, x₃, x₄]
  absl::btree_set<Point> super_point_set_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_POLYNOMIAL_OPENINGS_H_
