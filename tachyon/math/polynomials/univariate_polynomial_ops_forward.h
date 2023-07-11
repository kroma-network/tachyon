#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_OPS_FORWARD_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_OPS_FORWARD_H_

namespace tachyon {
namespace math {
namespace internal {

template <typename Coefficients, typename SFINAE = void>
class UnivariatePolynomialOp;

template <typename Coefficients, typename L, typename R, typename = void>
struct SupportsPolyAdd : std::false_type {};

template <typename Coefficients, typename L, typename R>
struct SupportsPolyAdd<Coefficients, L, R,
                       decltype(void(UnivariatePolynomialOp<Coefficients>::Add(
                           std::declval<const L&>(),
                           std::declval<const R&>())))> : std::true_type {};

template <typename Coefficients, typename L, typename R, typename = void>
struct SupportsPolyAddInPlace : std::false_type {};

template <typename Coefficients, typename L, typename R>
struct SupportsPolyAddInPlace<
    Coefficients, L, R,
    decltype(void(UnivariatePolynomialOp<Coefficients>::AddInPlace(
        std::declval<L&>(), std::declval<const R&>())))> : std::true_type {};

template <typename Coefficients, typename L, typename R, typename = void>
struct SupportsPolySub : std::false_type {};

template <typename Coefficients, typename L, typename R>
struct SupportsPolySub<Coefficients, L, R,
                       decltype(void(UnivariatePolynomialOp<Coefficients>::Sub(
                           std::declval<const L&>(),
                           std::declval<const R&>())))> : std::true_type {};

template <typename Coefficients, typename L, typename R, typename = void>
struct SupportsPolySubInPlace : std::false_type {};

template <typename Coefficients, typename L, typename R>
struct SupportsPolySubInPlace<
    Coefficients, L, R,
    decltype(void(UnivariatePolynomialOp<Coefficients>::SubInPlace(
        std::declval<L&>(), std::declval<const R&>())))> : std::true_type {};

template <typename Coefficients, typename L, typename R, typename = void>
struct SupportsPolyMul : std::false_type {};

template <typename Coefficients, typename L, typename R>
struct SupportsPolyMul<Coefficients, L, R,
                       decltype(void(UnivariatePolynomialOp<Coefficients>::Mul(
                           std::declval<const L&>(),
                           std::declval<const R&>())))> : std::true_type {};

template <typename Coefficients, typename L, typename R, typename = void>
struct SupportsPolyMulInPlace : std::false_type {};

template <typename Coefficients, typename L, typename R>
struct SupportsPolyMulInPlace<
    Coefficients, L, R,
    decltype(void(UnivariatePolynomialOp<Coefficients>::MulInPlace(
        std::declval<L&>(), std::declval<const R&>())))> : std::true_type {};

}  // namespace internal
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_OPS_FORWARD_H_
