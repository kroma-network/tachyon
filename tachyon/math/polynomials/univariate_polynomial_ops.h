#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_OPS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_OPS_H_

#include <algorithm>
#include <iterator>
#include <unordered_map>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/base/arithmetics_results.h"
#include "tachyon/math/polynomials/univariate_polynomial.h"

namespace tachyon {
namespace math {
namespace internal {

template <typename F, size_t MaxDegree>
class UnivariatePolynomialOp<DenseCoefficients<F, MaxDegree>> {
 public:
  using D = DenseCoefficients<F, MaxDegree>;
  using S = SparseCoefficients<F, MaxDegree>;
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
    if (self.IsZero()) {
      return Copy<false>(self, other);
    } else if (other.IsZero()) {
      return self;
    }

    size_t degree = self.Degree();
    size_t other_degree = other.Degree();
    std::vector<F> upper_coeffs;
    if (degree < other_degree) {
      upper_coeffs = base::CreateVector(other_degree - degree, F::Zero());
    }

    std::vector<F>& l_coefficients = self.coefficients_.coefficients_;
    const std::vector<Element>& r_elements = other.coefficients().elements_;
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
    if (self.IsZero()) {
      return Copy<true>(self, other);
    } else if (other.IsZero()) {
      return self;
    }

    size_t degree = self.Degree();
    size_t other_degree = other.Degree();
    std::vector<F> upper_coeffs;
    if (degree < other_degree) {
      upper_coeffs = base::CreateVector(other_degree - degree, F::Zero());
    }

    std::vector<F>& l_coefficients = self.coefficients_.coefficients_;
    const std::vector<Element>& r_elements = other.coefficients().elements_;
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

  static UnivariatePolynomial<D>& NegInPlace(UnivariatePolynomial<D>& self) {
    std::vector<F>& coefficients = self.coefficients_.coefficients_;
    for (F& coefficient : coefficients) {
      coefficient.NegInPlace();
    }
    return self;
  }

  static UnivariatePolynomial<D>& MulInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<D>& other) {
    std::vector<F>& l_coefficients = self.coefficients_.coefficients_;
    const std::vector<F>& r_coefficients = other.coefficients_.coefficients_;
    if (self.IsZero() || other.IsOne()) {
      return self;
    } else if (self.IsOne()) {
      l_coefficients = r_coefficients;
      return self;
    } else if (other.IsZero()) {
      l_coefficients = {};
      return self;
    }

    size_t degree = self.Degree();
    size_t other_degree = other.Degree();
    std::vector<F> coefficients =
        base::CreateVector(degree + other_degree + 1, F::Zero());
    for (size_t i = 0; i < r_coefficients.size(); ++i) {
      const F& r = r_coefficients[i];
      if (r.IsZero()) {
        continue;
      } else {
        for (size_t j = 0; j < l_coefficients.size(); ++j) {
          coefficients[i + j] += l_coefficients[j] * r;
        }
      }
    }

    l_coefficients = std::move(coefficients);
    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<D>& MulInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<S>& other) {
    std::vector<F>& l_coefficients = self.coefficients_.coefficients_;
    if (self.IsZero() || other.IsOne()) {
      return self;
    } else if (self.IsOne()) {
      return Copy<false>(self, other);
    } else if (other.IsZero()) {
      l_coefficients = {};
      return self;
    }

    size_t degree = self.Degree();
    size_t other_degree = other.Degree();
    std::vector<F> coefficients =
        base::CreateVector(degree + other_degree + 1, F::Zero());

    const std::vector<Element>& r_elements = other.coefficients().elements_;
    for (size_t i = 0; i < r_elements.size(); ++i) {
      const F& r = r_elements[i].coefficient;
      if (r.IsZero()) {
        continue;
      } else {
        size_t r_degree = r_elements[i].degree;
        for (size_t j = 0; j < l_coefficients.size(); ++j) {
          coefficients[r_degree + j] += l_coefficients[j] * r;
        }
      }
    }

    l_coefficients = std::move(coefficients);
    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  template <typename DOrS>
  static UnivariatePolynomial<D>& DivInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<DOrS>& other) {
    DivResult<UnivariatePolynomial<D>> result = Divide(self, other);
    self = std::move(result.quotient);
    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  template <typename DOrS>
  static UnivariatePolynomial<D>& ModInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<DOrS>& other) {
    DivResult<UnivariatePolynomial<D>> result = Divide(self, other);
    self = std::move(result.remainder);
    self.coefficients_.RemoveHighDegreeZeros();
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

 private:
  template <bool NEGATION>
  static UnivariatePolynomial<D>& Copy(UnivariatePolynomial<D>& self,
                                       const UnivariatePolynomial<S>& other) {
    std::vector<F>& l_coefficients = self.coefficients_.coefficients_;
    l_coefficients = base::CreateVector(other.Degree() + 1, F::Zero());

    const std::vector<Element>& r_elements = other.coefficients().elements_;
    for (const Element& r_elem : r_elements) {
      if constexpr (NEGATION) {
        l_coefficients[r_elem.degree] = -r_elem.coefficient;
      } else {
        l_coefficients[r_elem.degree] = r_elem.coefficient;
      }
    }
    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  template <typename DOrS>
  static DivResult<UnivariatePolynomial<D>> Divide(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<DOrS>& other) {
    if (self.IsZero()) {
      return {UnivariatePolynomial<D>::Zero(), UnivariatePolynomial<D>::Zero()};
    } else if (other.IsZero()) {
      NOTREACHED() << "Divide by zero polynomial";
    } else if (self.Degree() < other.Degree()) {
      return {UnivariatePolynomial<D>::Zero(), self.ToDense()};
    }
    std::vector<F> quotient =
        base::CreateVector(self.Degree() - other.Degree() + 1, F::Zero());
    UnivariatePolynomial<D> remainder = self.ToDense();
    std::vector<F>& r_coefficients = remainder.coefficients_.coefficients_;
    // Can unwrap here because we know self is not zero.
    F divisor_leading_inv = other.GetLeadingCoefficient()->Inverse();

    while (!remainder.IsZero() && remainder.Degree() >= other.Degree()) {
      F q_coeff =
          remainder.coefficients_.coefficients_.back() * divisor_leading_inv;
      size_t degree = remainder.Degree() - other.Degree();
      quotient[degree] = q_coeff;

      if constexpr (std::is_same_v<DOrS, D>) {
        const std::vector<F>& d_elements = other.coefficients_.coefficients_;
        for (size_t i = 0; i < d_elements.size(); ++i) {
          r_coefficients[degree + i] -= q_coeff * d_elements[i];
        }
      } else {
        const std::vector<Element>& d_elements = other.coefficients().elements_;
        for (const Element& d_elem : d_elements) {
          r_coefficients[degree + d_elem.degree] -=
              q_coeff * d_elem.coefficient;
        }
      }
      remainder.coefficients_.RemoveHighDegreeZeros();
    }
    D d(std::move(quotient));
    d.RemoveHighDegreeZeros();
    return {UnivariatePolynomial<D>(std::move(d)), std::move(remainder)};
  }
};

template <typename F, size_t MaxDegree>
class UnivariatePolynomialOp<SparseCoefficients<F, MaxDegree>> {
 public:
  using D = DenseCoefficients<F, MaxDegree>;
  using S = SparseCoefficients<F, MaxDegree>;
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

    return DoAddition<false>(self, other);
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

    return DoAddition<true>(self, other);
  }

  static UnivariatePolynomial<S>& NegInPlace(UnivariatePolynomial<S>& self) {
    std::vector<Element>& elements = self.coefficients_.elements_;
    for (Element& elem : elements) {
      elem.coefficient.NegInPlace();
    }
    return self;
  }

  static UnivariatePolynomial<D> Mul(const UnivariatePolynomial<S>& self,
                                     const UnivariatePolynomial<D>& other) {
    return other * self;
  }

  static UnivariatePolynomial<S>& MulInPlace(
      UnivariatePolynomial<S>& self, const UnivariatePolynomial<S>& other) {
    std::vector<Element>& l_elements = self.coefficients_.elements_;
    const std::vector<Element>& r_elements = other.coefficients_.elements_;
    if (self.IsZero() || other.IsOne()) {
      return self;
    } else if (self.IsOne()) {
      l_elements = r_elements;
      return self;
    } else if (other.IsZero()) {
      l_elements = {};
      return self;
    }

    std::vector<Element> records;
    for (const Element& l_elem : l_elements) {
      for (const Element& r_elem : r_elements) {
        F f = l_elem.coefficient * r_elem.coefficient;
        if (f.IsZero()) continue;
        size_t degree = l_elem.degree + r_elem.degree;
        auto it =
            base::ranges::find_if(records, [degree](const Element& element) {
              return element.degree == degree;
            });
        if (it != records.end()) {
          it->coefficient += f;
          if (it->coefficient.IsZero()) {
            records.erase(it);
          }
        } else {
          records.push_back({degree, std::move(f)});
        }
      }
    }
    l_elements = std::move(records);
    base::ranges::sort(l_elements);
    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  template <typename DOrS>
  static UnivariatePolynomial<D> Div(const UnivariatePolynomial<S>& self,
                                     const UnivariatePolynomial<DOrS>& other) {
    return self.ToDense() / other;
  }

  template <typename DOrS>
  static UnivariatePolynomial<D> Mod(const UnivariatePolynomial<S>& self,
                                     const UnivariatePolynomial<DOrS>& other) {
    return self.ToDense() % other;
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

 private:
  template <bool NEGATION>
  static UnivariatePolynomial<S>& DoAddition(
      UnivariatePolynomial<S>& self, const UnivariatePolynomial<S>& other) {
    std::vector<Element>& l_elements = self.coefficients_.elements_;
    const std::vector<Element>& r_elements = other.coefficients_.elements_;

    auto l_it = l_elements.begin();
    auto r_it = r_elements.begin();
    std::vector<Element> ret;
    while (l_it != l_elements.end() || r_it != r_elements.end()) {
      if (l_it == l_elements.end()) {
        if constexpr (NEGATION) {
          ret.push_back(-(*r_it));
        } else {
          ret.push_back(*r_it);
        }
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
        if constexpr (NEGATION) {
          ret.push_back(-(*r_it));
        } else {
          ret.push_back(*r_it);
        }
        ++r_it;
      } else {
        F coeff;
        if constexpr (NEGATION) {
          coeff = l_it->coefficient - r_it->coefficient;
        } else {
          coeff = l_it->coefficient + r_it->coefficient;
        }
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
};

}  // namespace internal
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_OPS_H_
