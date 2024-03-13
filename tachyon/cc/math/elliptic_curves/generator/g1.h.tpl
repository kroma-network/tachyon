// clang-format off
#include <string.h>

#include <ostream>

#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/g1.h"
#include "tachyon/cc/export.h"
#include "tachyon/cc/math/elliptic_curves/%{header_dir_name}/fq.h"

namespace tachyon::cc::math::%{type} {

class TACHYON_CC_EXPORT G1ProjectivePoint {
 public:
  G1ProjectivePoint() = default;
  explicit G1ProjectivePoint(const tachyon_%{type}_g1_projective& point)
      : G1ProjectivePoint(point.x, point.y, point.z) {}
  G1ProjectivePoint(const tachyon_%{type}_fq& x, const tachyon_%{type}_fq& y, const tachyon_%{type}_fq& z) {
    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(z_.value().limbs, z.limbs, sizeof(uint64_t) * %{fq_limb_nums});
  }
  G1ProjectivePoint(const Fq& x, const Fq& y, const Fq& z)
      : x_(x), y_(y), z_(z) {}

  const Fq& x() const { return x_; }
  Fq& x() { return x_; }

  const Fq& y() const { return y_; }
  Fq& y() { return y_; }

  const Fq& z() const { return z_; }
  Fq& z() { return z_; }

  static G1ProjectivePoint Zero() {
    return G1ProjectivePoint(tachyon_%{type}_g1_projective_zero());
  }

  static G1ProjectivePoint Generator() {
    return G1ProjectivePoint(tachyon_%{type}_g1_projective_generator());
  }

  static G1ProjectivePoint Random() {
    return G1ProjectivePoint(tachyon_%{type}_g1_projective_random());
  }

  G1ProjectivePoint operator+(const G1ProjectivePoint& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return G1ProjectivePoint(tachyon_%{type}_g1_projective_add(&a, &b));
  }

  G1ProjectivePoint& operator+=(const G1ProjectivePoint& other) {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    *this = G1ProjectivePoint(tachyon_%{type}_g1_projective_add(&a, &b));
    return *this;
  }

  G1ProjectivePoint operator-(const G1ProjectivePoint& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return G1ProjectivePoint(tachyon_%{type}_g1_projective_sub(&a, &b));
  }

  G1ProjectivePoint& operator-=(const G1ProjectivePoint& other) {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    *this = G1ProjectivePoint(tachyon_%{type}_g1_projective_sub(&a, &b));
    return *this;
  }

  G1ProjectivePoint operator-() const {
    return {x_, -y_, z_};
  }

  G1ProjectivePoint Double() const {
    auto a = ToCPoint();
    return G1ProjectivePoint(tachyon_%{type}_g1_projective_dbl(&a));
  }

  bool operator==(const G1ProjectivePoint& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return tachyon_%{type}_g1_projective_eq(&a, &b);
  }

  bool operator!=(const G1ProjectivePoint& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return tachyon_%{type}_g1_projective_ne(&a, &b);
  }

  tachyon_%{type}_g1_projective ToCPoint() const {
    tachyon_%{type}_g1_projective ret;
    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.z.limbs, z_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    return ret;
  }

  std::string ToString() const;

 private:
  Fq x_;
  Fq y_;
  Fq z_;
};

TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1ProjectivePoint& value);

class TACHYON_CC_EXPORT G1JacobianPoint {
 public:
  G1JacobianPoint() = default;
  explicit G1JacobianPoint(const tachyon_%{type}_g1_jacobian& point)
      : G1JacobianPoint(point.x, point.y, point.z) {}
  G1JacobianPoint(const tachyon_%{type}_fq& x, const tachyon_%{type}_fq& y, const tachyon_%{type}_fq& z) {
    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(z_.value().limbs, z.limbs, sizeof(uint64_t) * %{fq_limb_nums});
  }
  G1JacobianPoint(const Fq& x, const Fq& y, const Fq& z)
      : x_(x), y_(y), z_(z) {}

  const Fq& x() const { return x_; }
  Fq& x() { return x_; }

  const Fq& y() const { return y_; }
  Fq& y() { return y_; }

  const Fq& z() const { return z_; }
  Fq& z() { return z_; }

  static G1JacobianPoint Zero() {
    return G1JacobianPoint(tachyon_%{type}_g1_jacobian_zero());
  }

  static G1JacobianPoint Generator() {
    return G1JacobianPoint(tachyon_%{type}_g1_jacobian_generator());
  }

  static G1JacobianPoint Random() {
    return G1JacobianPoint(tachyon_%{type}_g1_jacobian_random());
  }

  G1JacobianPoint operator+(const G1JacobianPoint& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return G1JacobianPoint(tachyon_%{type}_g1_jacobian_add(&a, &b));
  }

  G1JacobianPoint& operator+=(const G1JacobianPoint& other) {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    *this = G1JacobianPoint(tachyon_%{type}_g1_jacobian_add(&a, &b));
    return *this;
  }

  G1JacobianPoint operator-(const G1JacobianPoint& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return G1JacobianPoint(tachyon_%{type}_g1_jacobian_sub(&a, &b));
  }

  G1JacobianPoint& operator-=(const G1JacobianPoint& other) {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    *this = G1JacobianPoint(tachyon_%{type}_g1_jacobian_sub(&a, &b));
    return *this;
  }

  G1JacobianPoint operator-() const {
    return {x_, -y_, z_};
  }

  G1JacobianPoint Double() const {
    auto a = ToCPoint();
    return G1JacobianPoint(tachyon_%{type}_g1_jacobian_dbl(&a));
  }

  bool operator==(const G1JacobianPoint& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return tachyon_%{type}_g1_jacobian_eq(&a, &b);
  }

  bool operator!=(const G1JacobianPoint& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return tachyon_%{type}_g1_jacobian_ne(&a, &b);
  }

  tachyon_%{type}_g1_jacobian ToCPoint() const {
    tachyon_%{type}_g1_jacobian ret;
    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.z.limbs, z_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    return ret;
  }

  std::string ToString() const;

 private:
  Fq x_;
  Fq y_;
  Fq z_;
};

TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1JacobianPoint& value);

class TACHYON_CC_EXPORT G1AffinePoint {
 public:
  G1AffinePoint() = default;
  explicit G1AffinePoint(const tachyon_%{type}_g1_affine& point)
      : G1AffinePoint(point.x, point.y, point.infinity) {}
  G1AffinePoint(const tachyon_%{type}_fq& x, const tachyon_%{type}_fq& y, bool infinity = false)
      : infinity_(infinity) {
    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});
  }
  G1AffinePoint(const Fq& x, const Fq& y, bool infinity = false)
      : x_(x), y_(y), infinity_(infinity) {}

  const Fq& x() const { return x_; }
  Fq& x() { return x_; }

  const Fq& y() const { return y_; }
  Fq& y() { return y_; }

  const bool infinity() const { return infinity_; }
  bool infinity() { return infinity_; }

  static G1AffinePoint Zero() {
    return G1AffinePoint(tachyon_%{type}_g1_affine_zero());
  }

  static G1AffinePoint Generator() {
    return G1AffinePoint(tachyon_%{type}_g1_affine_generator());
  }

  static G1AffinePoint Random() {
    return G1AffinePoint(tachyon_%{type}_g1_affine_random());
  }

  G1JacobianPoint operator+(const G1AffinePoint& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return G1JacobianPoint(tachyon_%{type}_g1_affine_add(&a, &b));
  }

  G1JacobianPoint operator-(const G1AffinePoint& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return G1JacobianPoint(tachyon_%{type}_g1_affine_sub(&a, &b));
  }

  G1AffinePoint operator-() const {
    return {x_, -y_, infinity_};
  }

  G1JacobianPoint Double() const {
    auto a = ToCPoint();
    return G1JacobianPoint(tachyon_%{type}_g1_affine_dbl(&a));
  }

  bool operator==(const G1AffinePoint& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return tachyon_%{type}_g1_affine_eq(&a, &b);
  }

  bool operator!=(const G1AffinePoint& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return tachyon_%{type}_g1_affine_ne(&a, &b);
  }

  tachyon_%{type}_g1_affine ToCPoint() const {
    tachyon_%{type}_g1_affine ret;
    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    ret.infinity = infinity_;
    return ret;
  }

  std::string ToString() const;

 private:
  Fq x_;
  Fq y_;
  bool infinity_ = false;
};

TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1AffinePoint& value);

class TACHYON_CC_EXPORT G1PointXYZZ {
 public:
  G1PointXYZZ() = default;
  explicit G1PointXYZZ(const tachyon_%{type}_g1_xyzz& point)
      : G1PointXYZZ(point.x, point.y, point.zz, point.zzz) {}
  G1PointXYZZ(const tachyon_%{type}_fq& x, const tachyon_%{type}_fq& y, const tachyon_%{type}_fq& zz, const tachyon_%{type}_fq& zzz) {
    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(zz_.value().limbs, zz.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(zzz_.value().limbs, zzz.limbs, sizeof(uint64_t) * %{fq_limb_nums});
  }
  G1PointXYZZ(const Fq& x, const Fq& y, const Fq& zz, const Fq& zzz)
      : x_(x), y_(y), zz_(zz), zzz_(zzz) {}

  const Fq& x() const { return x_; }
  Fq& x() { return x_; }

  const Fq& y() const { return y_; }
  Fq& y() { return y_; }

  const Fq& zz() const { return zz_; }
  Fq& zz() { return zz_; }

  const Fq& zzz() const { return zzz_; }
  Fq& zzz() { return zzz_; }

  static G1PointXYZZ Zero() {
    return G1PointXYZZ(tachyon_%{type}_g1_xyzz_zero());
  }

  static G1PointXYZZ Generator() {
    return G1PointXYZZ(tachyon_%{type}_g1_xyzz_generator());
  }

  static G1PointXYZZ Random() {
    return G1PointXYZZ(tachyon_%{type}_g1_xyzz_random());
  }

  G1PointXYZZ operator+(const G1PointXYZZ& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return G1PointXYZZ(tachyon_%{type}_g1_xyzz_add(&a, &b));
  }

  G1PointXYZZ& operator+=(const G1PointXYZZ& other) {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    *this = G1PointXYZZ(tachyon_%{type}_g1_xyzz_add(&a, &b));
    return *this;
  }

  G1PointXYZZ operator-(const G1PointXYZZ& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return G1PointXYZZ(tachyon_%{type}_g1_xyzz_sub(&a, &b));
  }

  G1PointXYZZ& operator-=(const G1PointXYZZ& other) {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    *this = G1PointXYZZ(tachyon_%{type}_g1_xyzz_sub(&a, &b));
    return *this;
  }

  G1PointXYZZ operator-() const {
    return {x_, -y_, zz_, zzz_};
  }

  G1PointXYZZ Double() const {
    auto a = ToCPoint();
    return G1PointXYZZ(tachyon_%{type}_g1_xyzz_dbl(&a));
  }

  bool operator==(const G1PointXYZZ& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return tachyon_%{type}_g1_xyzz_eq(&a, &b);
  }

  bool operator!=(const G1PointXYZZ& other) const {
    auto a = ToCPoint();
    auto b = other.ToCPoint();
    return tachyon_%{type}_g1_xyzz_ne(&a, &b);
  }

  tachyon_%{type}_g1_xyzz ToCPoint() const {
    tachyon_%{type}_g1_xyzz ret;
    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.zz.limbs, zz_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.zzz.limbs, zzz_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    return ret;
  }

  std::string ToString() const;

 private:
  Fq x_;
  Fq y_;
  Fq zz_;
  Fq zzz_;
};

TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1PointXYZZ& value);

class TACHYON_CC_EXPORT G1Point2 {
 public:
  G1Point2() = default;
  G1Point2(const tachyon_%{type}_g1_point2& point)
      : G1Point2(point.x, point.y) {}
  G1Point2(const tachyon_%{type}_fq& x, const tachyon_%{type}_fq& y) {
    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});
  }
  G1Point2(const Fq& x, const Fq& y)
      : x_(x), y_(y) {}

  const Fq& x() const { return x_; }
  Fq& x() { return x_; }

  const Fq& y() const { return y_; }
  Fq& y() { return y_; }

  tachyon_%{type}_g1_point2 ToCPoint() const {
    tachyon_%{type}_g1_point2 ret;
    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    return ret;
  }

  std::string ToString() const;

 private:
  Fq x_;
  Fq y_;
};

TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1Point2& value);

class TACHYON_CC_EXPORT G1Point3 {
 public:
  G1Point3() = default;
  G1Point3(const tachyon_%{type}_g1_point3& point)
      : G1Point3(point.x, point.y, point.z) {}
  G1Point3(const tachyon_%{type}_fq& x, const tachyon_%{type}_fq& y, const tachyon_%{type}_fq& z) {
    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(z_.value().limbs, z.limbs, sizeof(uint64_t) * %{fq_limb_nums});
  }
  G1Point3(const Fq& x, const Fq& y, const Fq& z)
      : x_(x), y_(y), z_(z) {}

  const Fq& x() const { return x_; }
  Fq& x() { return x_; }

  const Fq& y() const { return y_; }
  Fq& y() { return y_; }

  const Fq& z() const { return z_; }
  Fq& z() { return z_; }

  tachyon_%{type}_g1_point3 ToCPoint() const {
    tachyon_%{type}_g1_point3 ret;
    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.z.limbs, z_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    return ret;
  }

  std::string ToString() const;

 private:
  Fq x_;
  Fq y_;
  Fq z_;
};

TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1Point2& value);

class TACHYON_CC_EXPORT G1Point4 {
 public:
  G1Point4() = default;
  G1Point4(const tachyon_%{type}_g1_point4& point)
      : G1Point4(point.x, point.y, point.z, point.w) {}
  G1Point4(const tachyon_%{type}_fq& x, const tachyon_%{type}_fq& y,const tachyon_%{type}_fq& z, const tachyon_%{type}_fq& w) {
    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(z_.value().limbs, z.limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(w_.value().limbs, w.limbs, sizeof(uint64_t) * %{fq_limb_nums});
  }
  G1Point4(const Fq& x, const Fq& y, const Fq& z, const Fq& w)
      : x_(x), y_(y), z_(z), w_(w) {}

  const Fq& x() const { return x_; }
  Fq& x() { return x_; }

  const Fq& y() const { return y_; }
  Fq& y() { return y_; }

  const Fq& z() const { return z_; }
  Fq& z() { return z_; }

  const Fq& w() const { return w_; }
  Fq& w() { return w_; }

  tachyon_%{type}_g1_point4 ToCPoint() const {
    tachyon_%{type}_g1_point4 ret;
    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.z.limbs, z_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    memcpy(ret.w.limbs, w_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});
    return ret;
  }

  std::string ToString() const;

 private:
  Fq x_;
  Fq y_;
  Fq z_;
  Fq w_;
};

TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1Point4& value);

} // namespace tachyon::cc::math::%{type}
// clang-format on
