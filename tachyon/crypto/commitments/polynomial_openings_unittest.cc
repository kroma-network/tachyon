#include "tachyon/crypto/commitments/polynomial_openings.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::crypto {

using namespace math;

namespace {

const size_t kMaxDegree = 4;

using Point = GF7;
using Poly = UnivariateDensePolynomial<GF7, kMaxDegree>;
using Coeffs = UnivariateDenseCoefficients<GF7, kMaxDegree>;
using PolyDeepRef = base::DeepRef<const Poly>;
using PointDeepRef = base::DeepRef<const Point>;

class PolynomialOpeningsTest : public testing::Test {
 public:
  static void SetUpTestSuite() { GF7::Init(); }

  void SetUp() override {
    polys_.push_back(Poly(Coeffs({GF7(3), GF7(6), GF7(4), GF7(6), GF7(6)})));
    polys_.push_back(Poly(Coeffs({GF7(3), GF7(4), GF7(5), GF7(0), GF7(2)})));
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
      {PolyDeepRef(&polys_[0]),
       {polys_[0].Evaluate(points_[0]), polys_[0].Evaluate(points_[1])}},
      {PolyDeepRef(&polys_[1]),
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

TEST_F(PolynomialOpeningsTest, GroupByPolyAndPoints) {
  using PolyOracleGroupedPair =
      PolynomialOpeningGrouper<Poly>::PolyOracleGroupedPair;
  using PointGroupedPair = PolynomialOpeningGrouper<Poly>::PointGroupedPair;

  std::vector<PolynomialOpening<Poly>> poly_openings;
  poly_openings.emplace_back(PolyDeepRef(&polys_[0]), PointDeepRef(&points_[0]),
                             polys_[0].Evaluate(points_[0]));
  poly_openings.emplace_back(PolyDeepRef(&polys_[0]), PointDeepRef(&points_[1]),
                             polys_[0].Evaluate(points_[1]));
  poly_openings.emplace_back(PolyDeepRef(&polys_[1]), PointDeepRef(&points_[0]),
                             polys_[0].Evaluate(points_[0]));
  poly_openings.emplace_back(PolyDeepRef(&polys_[1]), PointDeepRef(&points_[1]),
                             polys_[0].Evaluate(points_[1]));
  poly_openings.emplace_back(PolyDeepRef(&polys_[2]), PointDeepRef(&points_[2]),
                             polys_[0].Evaluate(points_[2]));
  PolynomialOpeningGrouper<Poly> grouper;
  std::vector<PolyOracleGroupedPair> poly_openings_grouped_by_poly =
      grouper.GroupByPoly(poly_openings);

  for (const auto& poly_oracle_grouped_pair : poly_openings_grouped_by_poly) {
    absl::btree_set<PointDeepRef> expected_points;
    if (poly_oracle_grouped_pair.poly_oracle == PolyDeepRef(&polys_[0]) ||
        poly_oracle_grouped_pair.poly_oracle == PolyDeepRef(&polys_[1])) {
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

  std::vector<PointGroupedPair> poly_openings_grouped_by_poly_and_points =
      grouper.GroupByPoints(poly_openings_grouped_by_poly);
  for (const auto& point_grouped_pair :
       poly_openings_grouped_by_poly_and_points) {
    std::vector<PolyDeepRef> expected_polys;
    if (point_grouped_pair.points ==
        absl::btree_set<PointDeepRef>{PointDeepRef(&points_[0]),
                                      PointDeepRef(&points_[1])}) {
      expected_polys = {
          PolyDeepRef(&polys_[0]),
          PolyDeepRef(&polys_[1]),
      };
    } else {
      expected_polys = {
          PolyDeepRef(&polys_[2]),
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
  // TODO(chokobole): Test validity of
  // grouped_poly_openings_vec[i].poly_openings_vec
  if (grouped_poly_openings_vec[0].point_refs.size() == 2) {
    EXPECT_EQ(grouped_poly_openings_vec[1].point_refs.size(), 1);
    EXPECT_THAT(grouped_poly_openings_vec[0].point_refs,
                testing::UnorderedElementsAreArray(expected_points_vec));
    EXPECT_THAT(grouped_poly_openings_vec[1].point_refs,
                testing::UnorderedElementsAreArray(expected_points_vec2));
  } else {
    ASSERT_EQ(grouped_poly_openings_vec[0].point_refs.size(), 1);
    EXPECT_EQ(grouped_poly_openings_vec[1].point_refs.size(), 2);
    EXPECT_THAT(grouped_poly_openings_vec[0].point_refs,
                testing::UnorderedElementsAreArray(expected_points_vec2));
    EXPECT_THAT(grouped_poly_openings_vec[1].point_refs,
                testing::UnorderedElementsAreArray(expected_points_vec));
  }
}

}  // namespace tachyon::crypto
