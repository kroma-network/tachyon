// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_SYNTHESIS_MODE_H_
#define TACHYON_ZK_R1CS_SYNTHESIS_MODE_H_

namespace tachyon::zk::r1cs {

class SynthesisMode {
 public:
  enum class Type {
    kSetup,
    kProve,
  };

  SynthesisMode() = default;

  constexpr static SynthesisMode Setup() { return SynthesisMode(Type::kSetup); }
  constexpr static SynthesisMode Prove(bool construct_matrices) {
    return SynthesisMode(Type::kProve, construct_matrices);
  }

  Type type() const { return type_; }

  bool IsSetup() const { return type_ == Type::kSetup; }
  bool IsProve() const { return type_ == Type::kProve; }

  bool ShouldConstructMatrices() const {
    if (type_ == Type::kSetup) return true;
    return construct_matrices_;
  }

 private:
  constexpr explicit SynthesisMode(Type type) : type_(type) {}
  constexpr SynthesisMode(Type type, bool construct_matrices)
      : type_(type), construct_matrices_(construct_matrices) {}

  Type type_ = Type::kSetup;
  bool construct_matrices_ = false;
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_SYNTHESIS_MODE_H_
