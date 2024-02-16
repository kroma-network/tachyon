#include "tachyon/crypto/commitments/polynomial_openings.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::crypto {

using namespace math;

namespace {

const size_t kMaxDegree = 4;

using Point = GF7;
using Poly = UnivariateDensePolynomial<GF7, kMaxDegree>;
using Coeffs = UnivariateDenseCoefficients<GF7, kMaxDegree>;
using PolyRef = base::Ref<const Poly>;
using PointDeepRef = base::DeepRef<const Point>;

class PolynomialOpeningsTest : public FiniteFieldTest<GF7> {
 public:
  void SetUp() override {
    // NOTE(Insun35): For testing, I added the same polynomial to |polys_[2]|
    // and |polys_[3]| on purpose. Even if they have the same value, they should
    // be grouped separately because they are ShallowRef in
    // |PolynomialOpenings|.
    polys_.push_back(Poly(Coeffs({GF7(3), GF7(6), GF7(4), GF7(6), GF7(6)})));
    polys_.push_back(Poly(Coeffs({GF7(3), GF7(4), GF7(5), GF7(0), GF7(2)})));
    polys_.push_back(Poly(Coeffs({GF7(1), GF7(5), GF7(3), GF7(6), GF7(6)})));
    polys_.push_back(Poly(Coeffs({GF7(1), GF7(5), GF7(3), GF7(6), GF7(6)})));
    points_.push_back(GF7(1));
    points_.push_back(GF7(2));
    points_.push_back(GF7(3));
  }

 protected:
  std::vector<Poly> polys_;
  std::vector<Point> points_;
};

}  // namespace

TEST_F(PolynomialOpeningsTest, CreateCombinedLowDegreeExtensions) {
  std::vector<PolynomialOpenings<Poly>> poly_openings_vec = {
      {PolyRef(&polys_[0]),
       {polys_[0].Evaluate(points_[0]), polys_[0].Evaluate(points_[1])}},
      {PolyRef(&polys_[1]),
       {polys_[1].Evaluate(points_[0]), polys_[1].Evaluate(points_[1])}},
  };
  std::vector<PointDeepRef> point_refs = {
      PointDeepRef(&points_[0]),
      PointDeepRef(&points_[1]),
  };
  GroupedPolynomialOpenings<Poly> grouped_poly_opening(
      std::move(poly_openings_vec), std::move(point_refs));

  // NOTE(chokobole): Check whether the manually created low degree extensions
  // are constructed correctly.
  std::vector<Point> owned_points = grouped_poly_opening.CreateOwnedPoints();
  std::vector<Poly> low_degree_extensions =
      grouped_poly_opening.CreateLowDegreeExtensions(owned_points);
  for (size_t i = 0; i < low_degree_extensions.size(); ++i) {
    const Poly& low_degree_extension = low_degree_extensions[i];
    const PolynomialOpenings<Poly>& poly_openings =
        grouped_poly_opening.poly_openings_vec[i];
    for (size_t j = 0; j < poly_openings.openings.size(); ++j) {
      EXPECT_EQ(
          low_degree_extension.Evaluate(*grouped_poly_opening.point_refs[j]),
          poly_openings.openings[j]);
    }
  }

  // NOTE(chokobole): Check whether the manually combined low degree extension
  // and returned ones are same.
  GF7 r = GF7::Random();
  Poly combined_low_degree_extension =
      grouped_poly_opening.CombineLowDegreeExtensions(r, owned_points,
                                                      low_degree_extensions);
  std::vector<Poly> actual_low_degree_extensions;
  EXPECT_EQ(combined_low_degree_extension,
            grouped_poly_opening.CreateCombinedLowDegreeExtensions(
                r, actual_low_degree_extensions));
  EXPECT_EQ(actual_low_degree_extensions, low_degree_extensions);

  // NOTE(chokobole): Check whether the evaluations are same.
  Point x = Point::Random();
  GF7 actual_eval = combined_low_degree_extension.Evaluate(x);
  GF7 expected_eval =
      grouped_poly_opening.poly_openings_vec[0].poly_oracle->Evaluate(x) -
      low_degree_extensions[0].Evaluate(x);
  GF7 power = r;
  for (size_t i = 1; i < low_degree_extensions.size(); ++i) {
    expected_eval +=
        (grouped_poly_opening.poly_openings_vec[i].poly_oracle->Evaluate(x) -
         low_degree_extensions[i].Evaluate(x)) *
        power;
    power *= r;
  }
  for (const Point& point : owned_points) {
    expected_eval /= (x - point);
  }
  EXPECT_EQ(actual_eval, expected_eval);
}

#define OPENING(poly, point) \
  PolyRef(&poly), PointDeepRef(&point), poly.Evaluate(point)

TEST_F(PolynomialOpeningsTest, GroupBySinglePoint) {
  std::vector<PolynomialOpening<Poly>> poly_openings;
  poly_openings.emplace_back(OPENING(polys_[0], points_[0]));
  poly_openings.emplace_back(OPENING(polys_[0], points_[1]));
  poly_openings.emplace_back(OPENING(polys_[1], points_[0]));
  poly_openings.emplace_back(OPENING(polys_[1], points_[1]));
  poly_openings.emplace_back(OPENING(polys_[2], points_[2]));
  poly_openings.emplace_back(OPENING(polys_[3], points_[2]));
  PolynomialOpeningGrouper<Poly> grouper;
  grouper.GroupBySinglePoint(poly_openings);

  absl::btree_set<PointDeepRef> expected_points = {
      PointDeepRef(&points_[0]),
      PointDeepRef(&points_[1]),
      PointDeepRef(&points_[2]),
  };
  EXPECT_EQ(grouper.super_point_set(), expected_points);

  const std::vector<GroupedPolynomialOpenings<Poly>>&
      grouped_poly_openings_vec = grouper.grouped_poly_openings_vec();
  ASSERT_EQ(grouped_poly_openings_vec.size(), 3);
  std::vector<PointDeepRef> expected_points_vec = {
      PointDeepRef(&points_[0]),
  };
  std::vector<PointDeepRef> expected_points_vec2 = {
      PointDeepRef(&points_[1]),
  };
  std::vector<PointDeepRef> expected_points_vec3 = {
      PointDeepRef(&points_[2]),
  };
  std::vector<PolynomialOpenings<Poly>> expected_polys_vec = {
      PolynomialOpenings<Poly>(PolyRef(&polys_[0]), {poly_openings[0].opening}),
      PolynomialOpenings<Poly>(PolyRef(&polys_[1]), {poly_openings[2].opening}),
  };
  std::vector<PolynomialOpenings<Poly>> expected_polys_vec2 = {
      PolynomialOpenings<Poly>(PolyRef(&polys_[0]), {poly_openings[1].opening}),
      PolynomialOpenings<Poly>(PolyRef(&polys_[1]), {poly_openings[3].opening}),
  };
  std::vector<PolynomialOpenings<Poly>> expected_polys_vec3 = {
      PolynomialOpenings<Poly>(PolyRef(&polys_[2]), {poly_openings[4].opening}),
      PolynomialOpenings<Poly>(PolyRef(&polys_[3]), {poly_openings[5].opening}),
  };

  EXPECT_EQ(grouped_poly_openings_vec[0].point_refs, expected_points_vec);
  EXPECT_EQ(grouped_poly_openings_vec[1].point_refs, expected_points_vec2);
  EXPECT_EQ(grouped_poly_openings_vec[2].point_refs, expected_points_vec3);
  EXPECT_EQ(grouped_poly_openings_vec[0].poly_openings_vec, expected_polys_vec);
  EXPECT_EQ(grouped_poly_openings_vec[1].poly_openings_vec,
            expected_polys_vec2);
  EXPECT_EQ(grouped_poly_openings_vec[2].poly_openings_vec,
            expected_polys_vec3);
}

TEST_F(PolynomialOpeningsTest, GroupByPolyOracleAndPoints) {
  using GroupedPolyOraclePair =
      PolynomialOpeningGrouper<Poly>::GroupedPolyOraclePair;
  using GroupedPointPair = PolynomialOpeningGrouper<Poly>::GroupedPointPair;

  std::vector<PolynomialOpening<Poly>> poly_openings;
  poly_openings.emplace_back(OPENING(polys_[0], points_[0]));
  poly_openings.emplace_back(OPENING(polys_[0], points_[1]));
  poly_openings.emplace_back(OPENING(polys_[1], points_[0]));
  poly_openings.emplace_back(OPENING(polys_[1], points_[1]));
  poly_openings.emplace_back(OPENING(polys_[2], points_[2]));
  poly_openings.emplace_back(OPENING(polys_[3], points_[2]));
  PolynomialOpeningGrouper<Poly> grouper;
  std::vector<GroupedPolyOraclePair> poly_openings_grouped_by_poly =
      grouper.GroupByPolyOracle(poly_openings);

  // NOTE(Insun35): Although |polys_[2]| and |polys_[3]| have the same value,
  // they should be grouped separately because they are ShallowRef in
  // |PolynomialOpenings|.
  EXPECT_EQ(poly_openings_grouped_by_poly.size(), 4);

  for (const auto& poly_oracle_grouped_pair : poly_openings_grouped_by_poly) {
    absl::btree_set<PointDeepRef> expected_points;
    if (poly_oracle_grouped_pair.poly_oracle == PolyRef(&polys_[0]) ||
        poly_oracle_grouped_pair.poly_oracle == PolyRef(&polys_[1])) {
      expected_points = {
          PointDeepRef(&points_[0]),
          PointDeepRef(&points_[1]),
      };
    } else {
      expected_points = {
          PointDeepRef(&points_[2]),
      };
    }
    EXPECT_EQ(poly_oracle_grouped_pair.points, expected_points);
  }

  absl::btree_set<PointDeepRef> expected_points = {
      PointDeepRef(&points_[0]),
      PointDeepRef(&points_[1]),
      PointDeepRef(&points_[2]),
  };
  EXPECT_EQ(grouper.super_point_set(), expected_points);

  std::vector<GroupedPointPair> poly_openings_grouped_by_poly_and_points =
      grouper.GroupByPoints(poly_openings_grouped_by_poly);
  for (const auto& point_grouped_pair :
       poly_openings_grouped_by_poly_and_points) {
    std::vector<PolyRef> expected_polys;
    if (point_grouped_pair.points ==
        absl::btree_set<PointDeepRef>{PointDeepRef(&points_[0]),
                                      PointDeepRef(&points_[1])}) {
      expected_polys = {
          PolyRef(&polys_[0]),
          PolyRef(&polys_[1]),
      };
    } else {
      expected_polys = {
          PolyRef(&polys_[2]),
          PolyRef(&polys_[3]),
      };
    }
    EXPECT_EQ(point_grouped_pair.polys, expected_polys);
  }

  grouper.CreateMultiPolynomialOpenings(
      poly_openings, poly_openings_grouped_by_poly_and_points);
  const std::vector<GroupedPolynomialOpenings<Poly>>&
      grouped_poly_openings_vec = grouper.grouped_poly_openings_vec();
  ASSERT_EQ(grouped_poly_openings_vec.size(), 2);
  std::vector<PointDeepRef> expected_points_vec = {
      PointDeepRef(&points_[0]),
      PointDeepRef(&points_[1]),
  };
  std::vector<PointDeepRef> expected_points_vec2 = {
      PointDeepRef(&points_[2]),
  };
  std::vector<PolynomialOpenings<Poly>> expected_polys_vec = {
      PolynomialOpenings<Poly>(PolyRef(&polys_[0]), {poly_openings[0].opening,
                                                     poly_openings[1].opening}),
      PolynomialOpenings<Poly>(PolyRef(&polys_[1]), {poly_openings[2].opening,
                                                     poly_openings[3].opening}),
  };
  std::vector<PolynomialOpenings<Poly>> expected_polys_vec2 = {
      PolynomialOpenings<Poly>(PolyRef(&polys_[2]), {poly_openings[4].opening}),
      PolynomialOpenings<Poly>(PolyRef(&polys_[3]), {poly_openings[5].opening}),
  };

  EXPECT_EQ(grouped_poly_openings_vec[0].point_refs, expected_points_vec);
  EXPECT_EQ(grouped_poly_openings_vec[0].poly_openings_vec, expected_polys_vec);
  EXPECT_EQ(grouped_poly_openings_vec[1].point_refs, expected_points_vec2);
  EXPECT_EQ(grouped_poly_openings_vec[1].poly_openings_vec,
            expected_polys_vec2);
}

#undef OPENING

}  // namespace tachyon::crypto
