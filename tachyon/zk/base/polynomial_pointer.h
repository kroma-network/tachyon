#ifndef TACHYON_ZK_BASE_POLYNOMIAL_POINTER_H_
#define TACHYON_ZK_BASE_POLYNOMIAL_POINTER_H_

namespace tachyon::zk {

template <typename PCSTy>
class PolynomialPointer {
 public:
  using F = typename PCSTy::Field;
  using Poly = typename PCSTy::Poly;

  PolynomialPointer(const Poly& poly, const F& blind)
      : poly_(poly), blind_(blind) {}

  const Poly& poly() const { return poly_; }
  const F& blind() const { return blind_; }

  // Returns true if |this| and |other| have the same reference as a member.
  bool operator==(const PolynomialPointer& other) const {
    return &poly_ == &other.poly_ && &blind_ == &other.blind_;
  }

  bool operator!=(const PolynomialPointer& other) const {
    return !operator==(other);
  }

 private:
  const Poly& poly_;
  const F& blind_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_POLYNOMIAL_POINTER_H_
