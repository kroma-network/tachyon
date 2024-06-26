#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/sw_curve_config.h"
#include "tachyon/math/elliptic_curves/test/random.h"

namespace tachyon::math {

namespace {

class PointXYZZTest : public testing::Test {
 public:
  static void SetUpTestSuite() { test::G1Curve::Init(); }
};

}  // namespace

TEST_F(PointXYZZTest, IsZero) {
  EXPECT_TRUE(test::PointXYZZ(GF7(1), GF7(2), GF7(0), GF7(0)).IsZero());
  EXPECT_FALSE(test::PointXYZZ(GF7(1), GF7(2), GF7(1), GF7(0)).IsZero());
  EXPECT_TRUE(test::PointXYZZ(GF7(1), GF7(2), GF7(0), GF7(1)).IsZero());
}

TEST_F(PointXYZZTest, Generator) {
  EXPECT_EQ(test::PointXYZZ::Generator(),
            test::PointXYZZ(test::PointXYZZ::Curve::Config::kGenerator.x,
                            test::PointXYZZ::Curve::Config::kGenerator.y,
                            GF7::One(), GF7::One()));
}

TEST_F(PointXYZZTest, Random) {
  bool success = false;
  test::PointXYZZ r = test::PointXYZZ::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != test::PointXYZZ::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(PointXYZZTest, EqualityOperators) {
  {
    SCOPED_TRACE("p.IsZero() && p2.IsZero()");
    test::PointXYZZ p(GF7(1), GF7(2), GF7(0), GF7(0));
    test::PointXYZZ p2(GF7(3), GF7(4), GF7(0), GF7(0));
    EXPECT_EQ(p, p2);
    EXPECT_EQ(p2, p);
  }

  {
    SCOPED_TRACE("!p.IsZero() && p2.IsZero()");
    test::PointXYZZ p(GF7(1), GF7(2), GF7(1), GF7(0));
    test::PointXYZZ p2(GF7(3), GF7(4), GF7(0), GF7(0));
    EXPECT_NE(p, p2);
    EXPECT_NE(p2, p);
  }

  {
    SCOPED_TRACE("other");
    test::PointXYZZ p(GF7(1), GF7(2), GF7(2), GF7(6));
    test::PointXYZZ p2(GF7(1), GF7(2), GF7(2), GF7(6));
    EXPECT_EQ(p, p2);
    EXPECT_EQ(p2, p);
  }
}

TEST_F(PointXYZZTest, AdditiveGroupOperators) {
  test::PointXYZZ p =
      test::PointXYZZ::CreateChecked(GF7(5), GF7(5), GF7(1), GF7(1));
  test::PointXYZZ p2 =
      test::PointXYZZ::CreateChecked(GF7(3), GF7(2), GF7(1), GF7(1));
  test::PointXYZZ p3 =
      test::PointXYZZ::CreateChecked(GF7(3), GF7(5), GF7(1), GF7(1));
  test::PointXYZZ p4 =
      test::PointXYZZ::CreateChecked(GF7(6), GF7(5), GF7(1), GF7(1));
  test::AffinePoint ap = p.ToAffine();
  test::AffinePoint ap2 = p2.ToAffine();

  EXPECT_EQ(p + p2, p3);
  EXPECT_EQ(p - p3, -p2);
  EXPECT_EQ(p + p, p4);
  EXPECT_EQ(p - p4, -p);

  {
    test::PointXYZZ p_tmp = p;
    p_tmp += p2;
    EXPECT_EQ(p_tmp, p3);
    p_tmp -= p2;
    EXPECT_EQ(p_tmp, p);
  }

  EXPECT_EQ(p + ap2, p3);
  EXPECT_EQ(p + ap, p4);
  EXPECT_EQ(p - p3, -p2);
  EXPECT_EQ(p - p4, -p);

  EXPECT_EQ(p.Double(), p4);
  {
    test::PointXYZZ p_tmp = p;
    p_tmp.DoubleInPlace();
    EXPECT_EQ(p_tmp, p4);
  }

  EXPECT_EQ(-p, test::PointXYZZ(GF7(5), GF7(2), GF7(1), GF7(1)));
  {
    test::PointXYZZ p_tmp = p;
    p_tmp.NegateInPlace();
    EXPECT_EQ(p_tmp, test::PointXYZZ(GF7(5), GF7(2), GF7(1), GF7(1)));
  }

  EXPECT_EQ(p * GF7(2), p4);
  EXPECT_EQ(GF7(2) * p, p4);
  EXPECT_EQ(p *= GF7(2), p4);
}

TEST_F(PointXYZZTest, ScalarMulOperator) {
  std::vector<test::AffinePoint> points;
  for (size_t i = 0; i < 7; ++i) {
    points.push_back((GF7(i) * test::PointXYZZ::Generator()).ToAffine());
  }

  EXPECT_THAT(
      points,
      testing::UnorderedElementsAreArray(std::vector<test::AffinePoint>{
          test::AffinePoint(GF7(0), GF7(0)), test::AffinePoint(GF7(3), GF7(2)),
          test::AffinePoint(GF7(5), GF7(2)), test::AffinePoint(GF7(6), GF7(2)),
          test::AffinePoint(GF7(3), GF7(5)), test::AffinePoint(GF7(5), GF7(5)),
          test::AffinePoint(GF7(6), GF7(5))}));
}

TEST_F(PointXYZZTest, ToAffine) {
  EXPECT_EQ(test::PointXYZZ(GF7(1), GF7(2), GF7(0), GF7(0)).ToAffine(),
            test::AffinePoint::Zero());
  EXPECT_EQ(test::PointXYZZ(GF7(1), GF7(2), GF7(1), GF7(1)).ToAffine(),
            test::AffinePoint(GF7(1), GF7(2)));
  EXPECT_EQ(test::PointXYZZ(GF7(1), GF7(2), GF7(2), GF7(6)).ToAffine(),
            test::AffinePoint(GF7(4), GF7(5)));
}

TEST_F(PointXYZZTest, ToProjective) {
  EXPECT_EQ(test::PointXYZZ(GF7(1), GF7(2), GF7(0), GF7(0)).ToProjective(),
            test::ProjectivePoint::Zero());
  EXPECT_EQ(test::PointXYZZ(GF7(1), GF7(2), GF7(1), GF7(1)).ToProjective(),
            test::ProjectivePoint(GF7(1), GF7(2), GF7(1)));
  EXPECT_EQ(test::PointXYZZ(GF7(1), GF7(2), GF7(2), GF7(6)).ToProjective(),
            test::ProjectivePoint(GF7(6), GF7(4), GF7(5)));
}

TEST_F(PointXYZZTest, ToJacobian) {
  EXPECT_EQ(test::PointXYZZ(GF7(1), GF7(2), GF7(0), GF7(0)).ToJacobian(),
            test::JacobianPoint::Zero());
  EXPECT_EQ(test::PointXYZZ(GF7(1), GF7(2), GF7(1), GF7(1)).ToJacobian(),
            test::JacobianPoint(GF7(1), GF7(2), GF7(1)));
  EXPECT_EQ(test::PointXYZZ(GF7(1), GF7(2), GF7(2), GF7(6)).ToJacobian(),
            test::JacobianPoint(GF7(2), GF7(2), GF7(5)));
}

#if defined(TACHYON_HAS_OPENMP)
TEST_F(PointXYZZTest, BatchNormalize) {
  size_t size = size_t{1} << (static_cast<size_t>(omp_get_max_threads()) /
                              GF7::kParallelBatchInverseDivisorThreshold);
  for (size_t i = 0; i < 1; ++i) {
    // NOTE(chokobole): if i == 0 runs in parallel, otherwise runs in serial.
    std::vector<test::PointXYZZ> point_xyzzs =
        CreatePseudoRandomPoints<test::PointXYZZ>(size - i);

    std::vector<test::AffinePoint> affine_points;
    affine_points.resize(point_xyzzs.size() - 1);
    ASSERT_FALSE(test::PointXYZZ::BatchNormalize(point_xyzzs, &affine_points));

    affine_points.resize(point_xyzzs.size());
    ASSERT_TRUE(test::PointXYZZ::BatchNormalize(point_xyzzs, &affine_points));

    std::vector<test::AffinePoint> expected_affine_points = base::Map(
        point_xyzzs,
        [](const test::PointXYZZ& point) { return point.ToAffine(); });
    EXPECT_EQ(affine_points, expected_affine_points);
  }
}
#endif  // defined(TACHYON_HAS_OPENMP)

TEST_F(PointXYZZTest, BatchNormalizeSerial) {
  std::vector<test::PointXYZZ> point_xyzzs = {
      test::PointXYZZ(GF7(1), GF7(2), GF7(0), GF7(0)),
      test::PointXYZZ(GF7(1), GF7(2), GF7(1), GF7(1)),
      test::PointXYZZ(GF7(1), GF7(2), GF7(2), GF7(6))};

  std::vector<test::AffinePoint> affine_points;
  affine_points.resize(2);
  ASSERT_FALSE(
      test::PointXYZZ::BatchNormalizeSerial(point_xyzzs, &affine_points));

  affine_points.resize(3);
  ASSERT_TRUE(
      test::PointXYZZ::BatchNormalizeSerial(point_xyzzs, &affine_points));

  std::vector<test::AffinePoint> expected_affine_points = {
      test::AffinePoint::Zero(), test::AffinePoint(GF7(1), GF7(2)),
      test::AffinePoint(GF7(4), GF7(5))};
  EXPECT_EQ(affine_points, expected_affine_points);
}

TEST_F(PointXYZZTest, IsOnCurve) {
  test::PointXYZZ invalid_point(GF7(1), GF7(2), GF7(1), GF7(1));
  EXPECT_FALSE(invalid_point.IsOnCurve());
  test::PointXYZZ valid_point(GF7(3), GF7(2), GF7(1), GF7(1));
  EXPECT_TRUE(valid_point.IsOnCurve());
  valid_point = test::PointXYZZ(GF7(3), GF7(5), GF7(1), GF7(1));
  EXPECT_TRUE(valid_point.IsOnCurve());
}

TEST_F(PointXYZZTest, CreateFromX) {
  {
    std::optional<test::PointXYZZ> p =
        test::PointXYZZ::CreateFromX(GF7(3), /*pick_odd=*/true);
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->y(), GF7(5));
  }
  {
    std::optional<test::PointXYZZ> p =
        test::PointXYZZ::CreateFromX(GF7(3), /*pick_odd=*/false);
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->y(), GF7(2));
  }
  {
    std::optional<test::PointXYZZ> p =
        test::PointXYZZ::CreateFromX(GF7(1), /*pick_odd=*/false);
    ASSERT_FALSE(p.has_value());
  }
}

TEST_F(PointXYZZTest, Copyable) {
  test::PointXYZZ expected = test::PointXYZZ::Random();

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  test::PointXYZZ value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST_F(PointXYZZTest, JsonValueConverter) {
  test::PointXYZZ expected_point(GF7(1), GF7(2), GF7(3), GF7(4));
  std::string expected_json = R"({"x":1,"y":2,"zz":3,"zzz":4})";

  test::PointXYZZ p;
  std::string error;
  ASSERT_TRUE(base::ParseJson(expected_json, &p, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(p, expected_point);

  std::string json = base::WriteToJson(p);
  EXPECT_EQ(json, expected_json);
}

}  // namespace tachyon::math
