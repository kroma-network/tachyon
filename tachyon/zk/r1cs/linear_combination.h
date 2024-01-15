// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_LINEAR_COMBINATION_H_
#define TACHYON_ZK_R1CS_LINEAR_COMBINATION_H_

#include <stddef.h>

#include <algorithm>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/zk/r1cs/term.h"

namespace tachyon::zk::r1cs {

template <typename F>
class LinearCombination {
 public:
  constexpr LinearCombination() = default;
  constexpr explicit LinearCombination(const std::vector<Term<F>>& terms)
      : terms_(terms) {}
  constexpr explicit LinearCombination(std::vector<Term<F>>&& terms)
      : terms_(std::move(terms)) {}

  const std::vector<Term<F>>& terms() const { return terms_; }

  constexpr static LinearCombination CreateDeduplicated(
      const std::vector<Term<F>>& terms) {
    LinearCombination lc(terms);
    lc.Deduplicate();
    return lc;
  }

  constexpr static LinearCombination CreateDeduplicated(
      std::vector<Term<F>>&& terms) {
    LinearCombination lc(std::move(terms));
    lc.Deduplicate();
    return lc;
  }

  void AppendTerm(const Term<F>& term) { terms_.push_back(term); }
  void AppendTerm(Term<F>&& term) { terms_.push_back(std::move(term)); }

  void AppendTerms(const std::vector<Term<F>>& terms) {
    terms_.insert(terms_.end(), terms.begin(), terms.end());
  }
  void AppendTerms(std::vector<Term<F>>&& terms) {
    terms_.insert(terms_.end(), std::make_move_iterator(terms.begin()),
                  std::make_move_iterator(terms.end()));
  }

  std::vector<Term<F>>&& TakeTerms() && { return std::move(terms_); }

  void Deduplicate() {
    std::sort(terms_.begin(), terms_.end(),
              [](const Term<F>& a, const Term<F>& b) {
                return a.variable < b.variable;
              });
    bool is_first = true;
    auto cur_var_first_it = terms_.begin();
    auto it = terms_.begin();
    while (it != terms_.end()) {
      if (!is_first && cur_var_first_it->variable == it->variable) {
        cur_var_first_it->coefficient += it->coefficient;
      } else {
        cur_var_first_it = it;
        is_first = false;
      }
      ++it;
    }
    auto last = std::unique(terms_.begin(), terms_.end(),
                            [](const Term<F>& a, const Term<F>& b) {
                              return a.variable == b.variable;
                            });
    terms_.erase(last, terms_.end());
  }

  bool IsSorted() const {
    return base::ranges::is_sorted(terms_.begin(), terms_.end(),
                                   [](const Term<F>& a, const Term<F>& b) {
                                     return a.variable < b.variable;
                                   });
  }

  LinearCombination operator+(const Term<F>& term) const {
    LinearCombination ret = *this;
    ret += term;
    return ret;
  }

  LinearCombination& operator+=(const Term<F>& term) {
    size_t index = 0;
    if (GetVarLocation(term.variable, &index)) {
      terms_[index].coefficient += term.coefficient;
    } else {
      terms_.insert(terms_.begin() + index, term);
    }
    return *this;
  }

  LinearCombination operator-(const Term<F>& term) const {
    return operator+(-term);
  }

  LinearCombination& operator-=(const Term<F>& term) {
    return operator+=(-term);
  }

  template <typename T,
            std::enable_if_t<!std::is_same_v<std::remove_cv_t<T>, Term<F>&> &&
                             !std::is_same_v<T, Term<F>> &&
                             std::is_constructible_v<Term<F>, T>>* = nullptr>
  LinearCombination operator+(T&& value) const {
    return operator+(Term<F>(std::forward<T>(value)));
  }

  template <typename T,
            std::enable_if_t<!std::is_same_v<std::remove_cv_t<T>, Term<F>&> &&
                             !std::is_same_v<T, Term<F>> &&
                             std::is_constructible_v<Term<F>, T>>* = nullptr>
  LinearCombination& operator+=(T&& value) {
    return operator+=(Term<F>(std::forward<T>(value)));
  }

  template <typename T,
            std::enable_if_t<!std::is_same_v<std::remove_cv_t<T>, Term<F>&> &&
                             !std::is_same_v<T, Term<F>> &&
                             std::is_constructible_v<Term<F>, T>>* = nullptr>
  LinearCombination operator-(T&& value) const {
    return operator-(Term<F>(std::forward<T>(value)));
  }

  template <typename T,
            std::enable_if_t<!std::is_same_v<std::remove_cv_t<T>, Term<F>&> &&
                             !std::is_same_v<T, Term<F>> &&
                             std::is_constructible_v<Term<F>, T>>* = nullptr>
  LinearCombination& operator-=(T&& value) {
    return operator-=(Term<F>(std::forward<T>(value)));
  }

  LinearCombination operator-() const {
    return LinearCombination(
        base::Map(terms_, [](const Term<F>& term) { return -term; }));
  }
  LinearCombination& NegInPlace() {
    for (Term<F>& term : terms_) {
      term.NegInPlace();
    }
    return *this;
  }

  LinearCombination operator+(const LinearCombination& other) const {
    if (terms_.empty()) {
      return other;
    } else if (other.terms_.empty()) {
      return *this;
    }
    return DoAddition</*NEGATION=*/false>(other);
  }
  LinearCombination& operator+=(const LinearCombination& other) {
    if (terms_.empty()) {
      return *this = other;
    } else if (other.terms_.empty()) {
      return *this;
    }
    return *this = DoAddition</*NEGATION=*/false>(other);
  }

  LinearCombination operator-(const LinearCombination& other) const {
    if (terms_.empty()) {
      return -other;
    } else if (other.terms_.empty()) {
      return *this;
    }
    return DoAddition</*NEGATION=*/true>(other);
  }
  LinearCombination& operator-=(const LinearCombination& other) {
    if (terms_.empty()) {
      return *this = -other;
    } else if (other.terms_.empty()) {
      return *this;
    }
    return *this = DoAddition</*NEGATION=*/true>(other);
  }

  // TODO(chokobole): Let |LinearCombination| have an additional member to
  // accumulate scalar multiplications and do real multiplication lazily.
  LinearCombination operator*(const F& scalar) const {
    LinearCombination ret = *this;
    ret *= scalar;
    return ret;
  }
  LinearCombination& operator*=(const F& scalar) {
    for (Term<F>& term : terms_) {
      term *= scalar;
    }
    return *this;
  }

  bool operator==(const LinearCombination& other) const {
    return terms_ == other.terms_;
  }
  bool operator!=(const LinearCombination& other) const {
    return terms_ != other.terms_;
  }

  std::string ToString() const {
    std::vector<std::string> term_strs =
        base::Map(terms_, [](const Term<F>& term) { return term.ToString(); });
    return absl::StrJoin(term_strs, " + ");
  }

 private:
  FRIEND_TEST(LinearCombinationTest, BinarySearch);

  bool GetVarLocation(const Variable& variable, size_t* index) const {
    constexpr static size_t kBinarySearchThreshold = 6;
    if (terms_.size() < kBinarySearchThreshold) {
      for (size_t i = 0; i < terms_.size(); ++i) {
        const Variable& cur_variable = terms_[i].variable;
        if (cur_variable == variable) {
          *index = i;
          return true;
        } else if (cur_variable > variable) {
          *index = i;
          return false;
        }
      }
      *index = terms_.size();
      return false;
    } else {
      return BinarySearch(variable, index);
    }
  }

  // Returns true if the |variable| is found and populates |index| with the
  // index of the |variable|. Otherwise, returns false and and populates |index|
  // with the index where a matching |variable| could be inserted while
  // maintaining sorted order. See
  // https://doc.rust-lang.org/std/primitive.slice.html#method.binary_search_by
  bool BinarySearch(const Variable& variable, size_t* index) const {
    size_t size = terms_.size();
    size_t left = 0;
    size_t right = size;
    while (left < right) {
      size_t mid = left + size / 2;
      // SAFETY: the while condition means |size| is strictly positive, so
      // |size / 2 < size|. Thus |left + size / 2 < left + size|, which
      // coupled with the |left + size <= terms_.size()| invariant means
      // we have |left + size / 2 < terms_.size()|, and this is in-bounds.
      const Variable& mid_variable = terms_[mid].variable;

      if (mid_variable < variable) {
        left = mid + 1;
      } else if (mid_variable > variable) {
        right = mid;
      } else {
        *index = mid;
        return true;
      }
      size = right - left;
    }

    *index = left;
    return false;
  }

  template <bool NEGATION>
  LinearCombination DoAddition(const LinearCombination& other) const {
    const std::vector<Term<F>>& l_terms = terms_;
    const std::vector<Term<F>>& r_terms = other.terms_;

    auto l_it = l_terms.begin();
    auto r_it = r_terms.begin();
    std::vector<Term<F>> ret;
    ret.reserve(std::max(l_terms.size(), r_terms.size()));
    while (l_it != l_terms.end() || r_it != r_terms.end()) {
      if (l_it == l_terms.end()) {
        if constexpr (NEGATION) {
          ret.push_back(-(*r_it));
        } else {
          ret.push_back(*r_it);
        }
        ++r_it;
        continue;
      }
      if (r_it == r_terms.end()) {
        ret.push_back(*l_it);
        ++l_it;
        continue;
      }
      if (l_it->variable < r_it->variable) {
        ret.push_back(*l_it);
        ++l_it;
      } else if (l_it->variable > r_it->variable) {
        if constexpr (NEGATION) {
          ret.push_back(-(*r_it));
        } else {
          ret.push_back(*r_it);
        }
        ++r_it;
      } else {
        F coefficient;
        if constexpr (NEGATION) {
          coefficient = l_it->coefficient - r_it->coefficient;
        } else {
          coefficient = l_it->coefficient + r_it->coefficient;
        }
        ret.push_back({
            std::move(coefficient),
            l_it->variable,
        });
        ++l_it;
        ++r_it;
      }
    }
    return LinearCombination(std::move(ret));
  }

  std::vector<Term<F>> terms_;
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_LINEAR_COMBINATION_H_
