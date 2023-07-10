#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_OPS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_OPS_H_

#include <algorithm>
#include <iterator>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/polynomials/univariate_polynomial.h"

namespace tachyon {
namespace math {
namespace internal {

template <typename F, size_t MAX_DEGREE>
class UnivariatePolynomialOp<DenseCoefficients<F, MAX_DEGREE>> {
 public:
  using D = DenseCoefficients<F, MAX_DEGREE>;
  using S = SparseCoefficients<F, MAX_DEGREE>;
  using Element = typename S::Element;

  static UnivariatePolynomial<D>& AddInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<D>& other) {
    std::vector<F>& l_coefficients = self.coefficients_.coefficients_;
    if (self.IsZero()) {
      const std::vector<F>& r_coefficients = other.coefficients_.coefficients_;
      l_coefficients = r_coefficients;
      return self;
    } else if (other.IsZero()) {
      return self;
    }

    size_t degree = self.Degree();
    size_t other_degree = other.Degree();
    if (degree < other_degree) {
      size_t size = other.Degree() + 1;
      l_coefficients.reserve(size);
      std::fill_n(std::back_inserter(l_coefficients), size, F::Zero());
    }
    size_t max_degree = std::max(degree, other_degree);
    for (size_t i = 0; i < max_degree + 1; ++i) {
      l_coefficients[i] += (other[i] == nullptr) ? F::Zero() : *other[i];
    }

    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<D>& AddInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<S>& other) {
    std::vector<F>& l_coefficients = self.coefficients_.coefficients_;
    const std::vector<Element>& r_elements = other.coefficients().elements_;
    if (self.IsZero()) {
      size_t size = other.Degree() + 1;
      l_coefficients.reserve(size);
      std::fill_n(std::back_inserter(l_coefficients), size, F::Zero());

      for (const Element& r_elem : r_elements) {
        l_coefficients[r_elem.degree] = r_elem.coefficient;
      }
      return self;
    } else if (other.IsZero()) {
      return self;
    }

    size_t degree = self.Degree();
    size_t other_degree = other.Degree();
    std::vector<F> upper_coeffs;
    if (degree < other_degree) {
      upper_coeffs = base::CreateVector(other_degree - degree, F::Zero());
    }
    for (const Element& r_elem : r_elements) {
      if (r_elem.degree <= degree) {
        l_coefficients[r_elem.degree] += r_elem.coefficient;
      } else {
        upper_coeffs[r_elem.degree - degree - 1] = r_elem.coefficient;
      }
    }
    l_coefficients.insert(l_coefficients.end(),
                          std::make_move_iterator(upper_coeffs.begin()),
                          std::make_move_iterator(upper_coeffs.end()));

    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<D>& SubInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<D>& other) {
    std::vector<F>& l_coefficients = self.coefficients_.coefficients_;
    if (self.IsZero()) {
      const std::vector<F>& r_coefficients = other.coefficients_.coefficients_;
      l_coefficients = base::CreateVector(
          r_coefficients.size(),
          std::function<F(size_t)>(
              [&r_coefficients](size_t idx) { return -r_coefficients[idx]; }));
      return self;
    } else if (other.IsZero()) {
      return self;
    }

    size_t degree = self.Degree();
    size_t other_degree = other.Degree();
    if (degree < other_degree) {
      size_t size = other.Degree() + 1;
      l_coefficients.reserve(size);
      std::fill_n(std::back_inserter(l_coefficients), size, F::Zero());
    }
    size_t max_degree = std::max(degree, other_degree);
    for (size_t i = 0; i < max_degree + 1; ++i) {
      l_coefficients[i] -= (other[i] == nullptr) ? F::Zero() : *other[i];
    }

    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<D>& SubInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<S>& other) {
    std::vector<F>& l_coefficients = self.coefficients_.coefficients_;
    const std::vector<Element>& r_elements = other.coefficients().elements_;
    if (self.IsZero()) {
      size_t size = other.Degree() + 1;
      l_coefficients.reserve(size);
      std::fill_n(std::back_inserter(l_coefficients), size, F::Zero());

      for (const Element& r_elem : r_elements) {
        l_coefficients[r_elem.degree] = -r_elem.coefficient;
      }
      return self;
    } else if (other.IsZero()) {
      return self;
    }

    size_t degree = self.Degree();
    size_t other_degree = other.Degree();
    std::vector<F> upper_coeffs;
    if (degree < other_degree) {
      upper_coeffs = base::CreateVector(other_degree - degree, F::Zero());
    }
    for (const Element& r_elem : r_elements) {
      if (r_elem.degree <= degree) {
        l_coefficients[r_elem.degree] -= r_elem.coefficient;
      } else {
        upper_coeffs[r_elem.degree - degree - 1] = -r_elem.coefficient;
      }
    }
    l_coefficients.insert(l_coefficients.end(),
                          std::make_move_iterator(upper_coeffs.begin()),
                          std::make_move_iterator(upper_coeffs.end()));

    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<D>& NegativeInPlace(
      UnivariatePolynomial<D>& self) {
    std::vector<F>& coefficients = self.coefficients_.coefficients_;
    for (F& coefficient : coefficients) {
      coefficient.NegativeInPlace();
    }
    return self;
  }

  static UnivariatePolynomial<D> ToDensePolynomial(
      const UnivariatePolynomial<D>& self) {
    return self;
  }

  static UnivariatePolynomial<S> ToSparsePolynomial(
      const UnivariatePolynomial<D>& self) {
    std::vector<Element> elements;
    size_t size = self.Degree() + 1;
    // TODO(chokobole): What if this dense polynomial is really sparse..?
    elements.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      if (self[i] == nullptr || self[i]->IsZero()) {
        continue;
      }
      elements.push_back({i, *self[i]});
    }
    return UnivariatePolynomial<S>(S(std::move(elements)));
  }
};

template <typename F, size_t MAX_DEGREE>
class UnivariatePolynomialOp<SparseCoefficients<F, MAX_DEGREE>> {
 public:
  using D = DenseCoefficients<F, MAX_DEGREE>;
  using S = SparseCoefficients<F, MAX_DEGREE>;
  using Element = typename S::Element;

  static UnivariatePolynomial<D> Add(const UnivariatePolynomial<S>& self,
                                     const UnivariatePolynomial<D>& other) {
    return other + self;
  }

  static UnivariatePolynomial<S>& AddInPlace(
      UnivariatePolynomial<S>& self, const UnivariatePolynomial<S>& other) {
    std::vector<Element>& l_elements = self.coefficients_.elements_;
    const std::vector<Element>& r_elements = other.coefficients_.elements_;
    if (self.IsZero()) {
      l_elements = r_elements;
      return self;
    } else if (other.IsZero()) {
      return self;
    }

    auto l_it = l_elements.begin();
    auto r_it = r_elements.begin();
    std::vector<Element> ret;
    while (l_it != l_elements.end() || r_it != r_elements.end()) {
      if (l_it == l_elements.end()) {
        ret.push_back(*r_it);
        ++r_it;
        continue;
      }
      if (r_it == r_elements.end()) {
        ret.push_back(*l_it);
        ++l_it;
        continue;
      }
      if (l_it->degree < r_it->degree) {
        ret.push_back(*l_it);
        ++l_it;
      } else if (l_it->degree > r_it->degree) {
        ret.push_back(*r_it);
        ++r_it;
      } else {
        F coeff = l_it->coefficient + r_it->coefficient;
        if (!coeff.IsZero()) {
          ret.push_back({l_it->degree, std::move(coeff)});
        }
        ++l_it;
        ++r_it;
      }
    }
    l_elements = std::move(ret);
    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<D> Sub(const UnivariatePolynomial<S>& self,
                                     const UnivariatePolynomial<D>& other) {
    return -other + self;
  }

  static UnivariatePolynomial<S>& SubInPlace(
      UnivariatePolynomial<S>& self, const UnivariatePolynomial<S>& other) {
    std::vector<Element>& l_elements = self.coefficients_.elements_;
    const std::vector<Element>& r_elements = other.coefficients_.elements_;
    if (self.IsZero()) {
      l_elements = base::CreateVector(
          r_elements.size(),
          std::function<Element(size_t)>(
              [&r_elements](size_t idx) { return -r_elements[idx]; }));
      return self;
    } else if (other.IsZero()) {
      return self;
    }

    auto l_it = l_elements.begin();
    auto r_it = r_elements.begin();
    std::vector<Element> ret;
    while (l_it != l_elements.end() || r_it != r_elements.end()) {
      if (l_it == l_elements.end()) {
        ret.push_back(-(*r_it));
        ++r_it;
        continue;
      }
      if (r_it == r_elements.end()) {
        ret.push_back(*l_it);
        ++l_it;
        continue;
      }
      if (l_it->degree < r_it->degree) {
        ret.push_back(*l_it);
        ++l_it;
      } else if (l_it->degree > r_it->degree) {
        ret.push_back(-(*r_it));
        ++r_it;
      } else {
        F coeff = l_it->coefficient - r_it->coefficient;
        if (!coeff.IsZero()) {
          ret.push_back({l_it->degree, std::move(coeff)});
        }
        ++l_it;
        ++r_it;
      }
    }
    l_elements = std::move(ret);
    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<S>& NegativeInPlace(
      UnivariatePolynomial<S>& self) {
    std::vector<Element>& elements = self.coefficients_.elements_;
    for (Element& elem : elements) {
      elem.coefficient.NegativeInPlace();
    }
    return self;
  }

  static UnivariatePolynomial<D> ToDensePolynomial(
      const UnivariatePolynomial<S>& self) {
    std::vector<F> coefficients;
    size_t size = self.Degree() + 1;
    coefficients.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      coefficients.push_back(self[i] == nullptr ? F::Zero() : *self[i]);
    }
    return UnivariatePolynomial<D>(D(std::move(coefficients)));
  }

  static UnivariatePolynomial<S> ToSparsePolynomial(
      const UnivariatePolynomial<S>& self) {
    return self;
  }
};

}  // namespace internal
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_OPS_H_
