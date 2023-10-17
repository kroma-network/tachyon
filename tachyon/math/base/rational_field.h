#ifndef TACHYON_MATH_BASE_RATIONAL_FIELD_H_
#define TACHYON_MATH_BASE_RATIONAL_FIELD_H_

#include <string>
#include <utility>

#include "tachyon/math/base/field.h"

namespace tachyon::math {

// RationalField class can be used to optimize performance by delaying an
// expensive inverse computation. The motivation is very similar to why you use
// JacobianPoint or others instead of AffinePoint.
template <typename F>
class RationalField : public Field<RationalField<F>> {
 public:
  constexpr RationalField() = default;
  constexpr explicit RationalField(const F& numerator)
      : numerator_(numerator) {}
  constexpr explicit RationalField(F&& numerator)
      : numerator_(std::move(numerator)) {}
  constexpr RationalField(const F& numerator, const F& denominator)
      : numerator_(numerator), denominator_(denominator) {}
  constexpr RationalField(F&& numerator, F&& denominator)
      : numerator_(std::move(numerator)),
        denominator_(std::move(denominator)) {}

  constexpr static RationalField Zero() { return RationalField(); }

  constexpr static RationalField One() { return RationalField(F::One()); }

  constexpr static RationalField Random() {
    F denominator = F::Random();
    while (denominator.IsZero()) {
      denominator = F::Random();
    }
    return {F::Random(), std::move(denominator)};
  }

  template <typename InputContainer, typename OutputContainer>
  constexpr static bool BatchEvaluate(const InputContainer& ration_fields,
                                      OutputContainer* results,
                                      const F& coeff = F::One()) {
    static_assert(
        std::is_same_v<typename InputContainer::value_type, RationalField>);
    static_assert(std::is_same_v<typename OutputContainer::value_type, F>);

    if (std::size(ration_fields) != std::size(*results)) {
      LOG(ERROR) << "Size of |ration_fields| and |results| do not match";
      return false;
    }
    size_t size = std::size(ration_fields);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < size; ++i) {
      (*results)[i] = ration_fields[i].denominator_;
    }
    F::BatchInverseInPlace(*results, coeff);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < size; ++i) {
      (*results)[i] *= ration_fields[i].numerator_;
    }
    return true;
  }

  constexpr const F& numerator() const { return numerator_; }
  constexpr const F& denominator() const { return denominator_; }

  constexpr bool IsZero() const { return numerator_.IsZero(); }

  constexpr bool IsOne() const {
    return numerator_.IsOne() || numerator_ == denominator_;
  }

  std::string ToString() const {
    return absl::Substitute("$0 / $1", numerator_.ToString(),
                            denominator_.ToString());
  }

  constexpr bool operator==(const RationalField& other) const {
    return numerator_ * other.denominator_ == other.numerator_ * denominator_;
  }
  constexpr bool operator!=(const RationalField& other) const {
    return numerator_ * other.denominator_ != other.numerator_ * denominator_;
  }
  constexpr bool operator<(const RationalField& other) const {
    return numerator_ * other.denominator_ < other.numerator_ * denominator_;
  }
  constexpr bool operator>(const RationalField& other) const {
    return numerator_ * other.denominator_ > other.numerator_ * denominator_;
  }
  constexpr bool operator<=(const RationalField& other) const {
    return numerator_ * other.denominator_ <= other.numerator_ * denominator_;
  }
  constexpr bool operator>=(const RationalField& other) const {
    return numerator_ * other.denominator_ >= other.numerator_ * denominator_;
  }

  F Evaluate() const { return numerator_ / denominator_; }

  // AdditiveSemigroup methods
  constexpr RationalField Add(const RationalField& other) const {
    return {numerator_ * other.denominator_ + other.numerator_ * denominator_,
            denominator_ * other.denominator_};
  }

  constexpr RationalField& AddInPlace(const RationalField& other) {
    numerator_ =
        numerator_ * other.denominator_ + other.numerator_ * denominator_;
    denominator_ *= other.denominator_;
    return *this;
  }

  constexpr RationalField& DoubleInPlace() {
    numerator_.DoubleInPlace();
    return *this;
  }

  // AdditiveGroup methods
  constexpr RationalField Sub(const RationalField& other) const {
    return {numerator_ * other.denominator_ - other.numerator_ * denominator_,
            denominator_ * other.denominator_};
  }

  constexpr RationalField& SubInPlace(const RationalField& other) {
    numerator_ =
        numerator_ * other.denominator_ - other.numerator_ * denominator_;
    denominator_ *= other.denominator_;
    return *this;
  }

  constexpr RationalField& NegInPlace() {
    numerator_.NegInPlace();
    return *this;
  }

  // MultiplicativeSemigroup methods
  constexpr RationalField Mul(const RationalField& other) const {
    return {numerator_ * other.numerator_, denominator_ * other.denominator_};
  }

  constexpr RationalField& MulInPlace(const RationalField& other) {
    numerator_ *= other.numerator_;
    denominator_ *= other.denominator_;
    return *this;
  }

  constexpr RationalField& SquareInPlace() {
    numerator_.SquareInPlace();
    denominator_.SquareInPlace();
    return *this;
  }

  // MultiplicativeGroup methods
  constexpr RationalField& InverseInPlace() {
    std::swap(numerator_, denominator_);
    return *this;
  }

 private:
  F numerator_;
  F denominator_ = F::One();
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_RATIONAL_FIELD_H_
