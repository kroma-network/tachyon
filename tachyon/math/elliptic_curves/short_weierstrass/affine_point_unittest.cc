#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/sw_curve_config.h"

namespace tachyon::math {

namespace {

class AffinePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { test::G1Curve::Init(); }
};

}  // namespace

TEST_F(AffinePointTest, Zero) {
  EXPECT_TRUE(test::AffinePoint::Zero().infinity());
  EXPECT_TRUE(test::AffinePoint::Zero().IsZero());
  EXPECT_FALSE(test::AffinePoint(GF7(1), GF7(2)).IsZero());
}

TEST_F(AffinePointTest, Generator) {
  EXPECT_EQ(test::AffinePoint::Generator(),
            test::AffinePoint(test::G1Curve::Config::kGenerator.x,
                              test::G1Curve::Config::kGenerator.y));
}

TEST_F(AffinePointTest, Montgomery) {
  test::AffinePoint r = test::AffinePoint::Random();
  while (r.infinity()) {
    r = test::AffinePoint::Random();
  }
  EXPECT_EQ(r, test::AffinePoint::FromMontgomery(r.ToMontgomery()));
}

TEST_F(AffinePointTest, Random) {
  bool success = false;
  test::AffinePoint r = test::AffinePoint::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != test::AffinePoint::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(AffinePointTest, EqualityOperators) {
  test::AffinePoint p(GF7(1), GF7(2));
  test::AffinePoint p2(GF7(3), GF7(4));
  EXPECT_TRUE(p == p);
  EXPECT_TRUE(p != p2);
}

TEST_F(AffinePointTest, AdditiveGroupOperators) {
  test::AffinePoint ap = test::AffinePoint::CreateChecked(GF7(5), GF7(5));
  test::AffinePoint ap2 = test::AffinePoint::CreateChecked(GF7(3), GF7(2));
  test::AffinePoint ap3 = test::AffinePoint::CreateChecked(GF7(3), GF7(5));
  test::AffinePoint ap4 = test::AffinePoint::CreateChecked(GF7(6), GF7(5));
  test::JacobianPoint jp = ap.ToJacobian();
  test::JacobianPoint jp2 = ap2.ToJacobian();
  test::JacobianPoint jp3 = ap3.ToJacobian();
  test::JacobianPoint jp4 = ap4.ToJacobian();

  EXPECT_EQ(ap + ap2, jp3);
  EXPECT_EQ(ap + ap, jp4);
  EXPECT_EQ(ap3 - ap2, jp);
  EXPECT_EQ(ap4 - ap, jp);

  EXPECT_EQ(ap + jp2, jp3);
  EXPECT_EQ(ap + jp, jp4);
  EXPECT_EQ(ap - jp3, -jp2);
  EXPECT_EQ(ap - jp4, -jp);

  EXPECT_EQ(ap.Double(), jp4);
  EXPECT_EQ(ap.DoubleProjective(), ap4.ToProjective());
  EXPECT_EQ(ap.DoubleXYZZ(), ap4.ToXYZZ());

  EXPECT_EQ(-ap, test::AffinePoint(GF7(5), GF7(2)));
  {
    test::AffinePoint ap_tmp = ap;
    ap_tmp.NegInPlace();
    EXPECT_EQ(ap_tmp, test::AffinePoint(GF7(5), GF7(2)));
  }

  EXPECT_EQ(ap * GF7(2), jp4);
  EXPECT_EQ(GF7(2) * ap, jp4);
}

TEST_F(AffinePointTest, ToProjective) {
  EXPECT_EQ(test::AffinePoint::Zero().ToProjective(),
            test::ProjectivePoint::Zero());
  test::AffinePoint p(GF7(3), GF7(2));
  EXPECT_EQ(p.ToProjective(), test::ProjectivePoint(GF7(3), GF7(2), GF7(1)));
}

TEST_F(AffinePointTest, ToJacobian) {
  EXPECT_EQ(test::AffinePoint::Zero().ToJacobian(),
            test::JacobianPoint::Zero());
  test::AffinePoint p(GF7(3), GF7(2));
  EXPECT_EQ(p.ToJacobian(), test::JacobianPoint(GF7(3), GF7(2), GF7(1)));
}

TEST_F(AffinePointTest, ToPointXYZZ) {
  EXPECT_EQ(test::AffinePoint::Zero().ToXYZZ(), test::PointXYZZ::Zero());
  test::AffinePoint p(GF7(3), GF7(2));
  EXPECT_EQ(p.ToXYZZ(), test::PointXYZZ(GF7(3), GF7(2), GF7(1), GF7(1)));
}

#if defined(TACHYON_HAS_OPENMP)
TEST_F(AffinePointTest, BatchMapScalarFieldToPoint) {
  size_t size = size_t{1} << (static_cast<size_t>(omp_get_max_threads()) /
                              GF7::kParallelBatchInverseDivisorThreshold);
  std::vector<GF7> scalar_fields =
      base::CreateVector(size, [](int i) { return GF7(i % 7); });
  test::AffinePoint point = test::AffinePoint::Generator();

  std::vector<test::AffinePoint> affine_points;
  affine_points.resize(scalar_fields.size() - 1);
  ASSERT_FALSE(test::AffinePoint::BatchMapScalarFieldToPoint(
      point, scalar_fields, &affine_points));

  affine_points.resize(scalar_fields.size());
  ASSERT_TRUE(test::AffinePoint::BatchMapScalarFieldToPoint(
      point, scalar_fields, &affine_points));

  std::vector<test::AffinePoint> expected_affine_points =
      base::Map(scalar_fields, [&point](const GF7& scalar_field) {
        return (scalar_field * point).ToAffine();
      });
  EXPECT_EQ(affine_points, expected_affine_points);
}
#endif  // defined(TACHYON_HAS_OPENMP)

TEST_F(AffinePointTest, BatchMapScalarFieldToPointSerial) {
  std::vector<GF7> scalar_fields =
      base::CreateVector(7, [](int i) { return GF7(i); });
  test::AffinePoint point = test::AffinePoint::Generator();

  std::vector<test::AffinePoint> affine_points;
  affine_points.resize(6);
  ASSERT_FALSE(test::AffinePoint::BatchMapScalarFieldToPointSerial(
      point, scalar_fields, &affine_points));

  affine_points.resize(7);
  ASSERT_TRUE(test::AffinePoint::BatchMapScalarFieldToPointSerial(
      point, scalar_fields, &affine_points));

  std::vector<test::AffinePoint> expected_affine_points =
      base::Map(scalar_fields, [&point](const GF7& scalar_field) {
        return (scalar_field * point).ToAffine();
      });
  EXPECT_EQ(affine_points, expected_affine_points);
}

TEST_F(AffinePointTest, IsOnCurve) {
  test::AffinePoint invalid_point(GF7(1), GF7(2));
  EXPECT_FALSE(invalid_point.IsOnCurve());
  test::AffinePoint valid_point(GF7(3), GF7(2));
  EXPECT_TRUE(valid_point.IsOnCurve());
  valid_point = test::AffinePoint(GF7(3), GF7(5));
  EXPECT_TRUE(valid_point.IsOnCurve());
}

TEST_F(AffinePointTest, CreateFromX) {
  {
    std::optional<test::AffinePoint> p =
        test::AffinePoint::CreateFromX(GF7(3), /*pick_odd=*/true);
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->y(), GF7(5));
  }
  {
    std::optional<test::AffinePoint> p =
        test::AffinePoint::CreateFromX(GF7(3), /*pick_odd=*/false);
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->y(), GF7(2));
  }
  {
    std::optional<test::AffinePoint> p =
        test::AffinePoint::CreateFromX(GF7(1), /*pick_odd=*/false);
    ASSERT_FALSE(p.has_value());
  }
}

TEST_F(AffinePointTest, Copyable) {
  test::AffinePoint expected = test::AffinePoint::Random();

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  test::AffinePoint value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST_F(AffinePointTest, JsonValueConverter) {
  test::AffinePoint expected_point(GF7(1), GF7(2));
  std::string expected_json = R"({"x":{"value":"0x1"},"y":{"value":"0x2"}})";

  test::AffinePoint p;
  std::string error;
  ASSERT_TRUE(base::ParseJson(expected_json, &p, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(p, expected_point);

  std::string json = base::WriteToJson(p);
  EXPECT_EQ(json, expected_json);
}

}  // namespace tachyon::math
