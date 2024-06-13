// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_EXPRESSIONS_CHALLENGE_EXPRESSION_H_
#define TACHYON_ZK_EXPRESSIONS_CHALLENGE_EXPRESSION_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"

#include "tachyon/zk/expressions/expression.h"
#include "tachyon/zk/plonk/constraint_system/challenge.h"

namespace tachyon::zk {

template <typename F>
class ChallengeExpression : public Expression<F> {
 public:
  static std::unique_ptr<ChallengeExpression> CreateForTesting(
      plonk::Challenge challenge) {
    return absl::WrapUnique(new ChallengeExpression(challenge));
  }

  plonk::Challenge challenge() const { return challenge_; }

  // Expression methods
  size_t Degree() const override { return 0; }

  uint64_t Complexity() const override { return 0; }

  std::unique_ptr<Expression<F>> Clone() const override {
    return absl::WrapUnique(new ChallengeExpression(challenge_));
  }

  std::string ToString() const override {
    return absl::Substitute("{type: $0, challenge: $1}",
                            ExpressionTypeToString(this->type_),
                            challenge_.ToString());
  }

  void WriteIdentifier(std::ostream& out) const override {
    out << "challenge[" << challenge_.index() << "]";
  }

  bool operator==(const Expression<F>& other) const override {
    if (!Expression<F>::operator==(other)) return false;
    const ChallengeExpression* challenge = other.ToChallenge();
    return challenge_ == challenge->challenge_;
  }

 private:
  friend class ExpressionFactory<F>;

  explicit ChallengeExpression(plonk::Challenge challenge)
      : Expression<F>(ExpressionType::kChallenge), challenge_(challenge) {}

  plonk::Challenge challenge_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_EXPRESSIONS_CHALLENGE_EXPRESSION_H_
