#ifndef TACHYON_MATH_FINITE_FIELDS_QUADRATIC_EXTENSION_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_QUADRATIC_EXTENSION_FIELD_H_

namespace tachyon {
namespace math {

template <typename F>
class QuadraticExtensionField {
 public:
  using BaseField = F;

  constexpr QuadraticExtensionField() = default;
  constexpr QuadraticExtensionField(const BaseField& c0, const BaseField& c1)
      : c0_(c0), c1_(c1) {}
  constexpr QuadraticExtensionField(BaseField&& c0, BaseField&& c1)
      : c0_(std::move(c0)), c1_(std::move(c1)) {}

  QuadraticExtensionField& ConjugateInPlace() {
    c1_.NegInPlace();
    return *this;
  }

 private:
  // c = c0_ + c1_ * X
  BaseField c0_;
  BaseField c1_;
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_QUADRATIC_EXTENSION_FIELD_H_
