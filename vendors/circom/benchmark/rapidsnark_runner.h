#ifndef VENDORS_CIRCOM_BENCHMARK_RAPIDSNARK_RUNNER_H_
#define VENDORS_CIRCOM_BENCHMARK_RAPIDSNARK_RUNNER_H_

#include <string.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fileloader.hpp"  // NOLINT(build/include_subdir)
#include "groth16.hpp"     // NOLINT(build/include_subdir)
#include "zkey_utils.hpp"  // NOLINT(build/include_subdir)

// clang-format off
#include "benchmark/runner.h"
// clang-format on
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/openmp_util.h"

namespace tachyon::circom {

namespace {

template <typename Engine>
void parse_key(Groth16::VerificationKey<Engine>& key, const char* key_str) {
  try {
    json key_json = json::parse(key_str);

    auto protocol = key_json["protocol"].template get<std::string>();

    CHECK_EQ(protocol, "groth16");

    key.fromJson(key_json);

    CHECK(!key.IC.empty());
  } catch (...) {
    NOTREACHED();
  }
}

}  // namespace

template <typename Curve, size_t MaxDegree, typename Engine>
class RapidsnarkRunner : public Runner<Curve, MaxDegree> {
 public:
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;
  using F = typename Curve::G1Curve::ScalarField;
  using Domain = math::UnivariateEvaluationDomain<F, MaxDegree>;

  explicit RapidsnarkRunner(const base::FilePath& vk_path)
      : verification_key_(Engine::engine) {
    BinFileUtils::FileLoader key(vk_path.value());
    parse_key(verification_key_, key.dataAsString().c_str());
  }

  void LoadZkey(const base::FilePath& zkey_path) override {
    zkey_ = BinFileUtils::openExisting(zkey_path.value(), "zkey", 1);
    zkey_header_ = ZKeyUtils::loadHeader(zkey_.get());
    prover_ = Groth16::makeProver<Engine>(
        zkey_header_->nVars, zkey_header_->nPublic, zkey_header_->domainSize,
        zkey_header_->nCoefs, zkey_header_->vk_alpha1, zkey_header_->vk_beta1,
        zkey_header_->vk_beta2, zkey_header_->vk_delta1,
        zkey_header_->vk_delta2,
        zkey_->getSectionData(4),  // Coeffs
        zkey_->getSectionData(5),  // pointsA
        zkey_->getSectionData(6),  // pointsB1
        zkey_->getSectionData(7),  // pointsB2
        zkey_->getSectionData(8),  // pointsC
        zkey_->getSectionData(9)   // pointsH1
    );                             // NOLINT(whitespace/parens)
  }

  zk::r1cs::groth16::Proof<Curve> Run(const Domain*,
                                      const std::vector<F>& full_assignments_in,
                                      absl::Span<const F> public_inputs_in,
                                      base::TimeDelta& delta) override {
    using FrElement = typename Engine::Fr::Element;

    base::TimeTicks now = base::TimeTicks::Now();

    std::vector<FrElement> full_assignments(full_assignments_in.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < full_assignments_in.size(); ++i) {
      using BigInt = typename F::BigIntTy;
      BigInt bigint = full_assignments_in[i].ToBigInt();
      memcpy(full_assignments[i].v, bigint.limbs, BigInt::kByteNums);
    }

    std::unique_ptr<Groth16::Proof<Engine>> proof =
        prover_->prove(full_assignments.data());

    delta = base::TimeTicks::Now() - now;

    std::vector<FrElement> public_inputs =
        base::Map(public_inputs_in, [](const F& public_input) {
          return reinterpret_cast<const FrElement&>(public_input);
        });

    Groth16::Verifier<AltBn128::Engine> verifier;
    CHECK(verifier.verify(*proof, public_inputs, verification_key_));

    return {
        G1AffinePointToNative(proof->A),
        G2AffinePointToNative(proof->B),
        G1AffinePointToNative(proof->C),
    };
  }

 private:
  using G1PointAffine = typename Engine::G1PointAffine;
  using G2PointAffine = typename Engine::G2PointAffine;

  static G1AffinePoint G1AffinePointToNative(const G1PointAffine& point) {
    using BaseField = typename Curve::G1Curve::BaseField;
    using BigInt = typename BaseField::BigIntTy;
    BigInt x_bigint;
    memcpy(x_bigint.limbs, &point.x, BigInt::kByteNums);
    BigInt y_bigint;
    memcpy(y_bigint.limbs, &point.y, BigInt::kByteNums);
    return {BaseField::FromMontgomery(x_bigint),
            BaseField::FromMontgomery(y_bigint)};
  }

  static G2AffinePoint G2AffinePointToNative(const G2PointAffine& point) {
    using BaseField = typename Curve::G2Curve::BaseField;
    using BasePrimeField = typename BaseField::BasePrimeField;
    using BigInt = typename BasePrimeField::BigIntTy;
    BaseField x;
    {
      BigInt a_bigint;
      memcpy(a_bigint.limbs, &point.x.a, BigInt::kByteNums);
      BasePrimeField a = BasePrimeField::FromMontgomery(a_bigint);
      BigInt b_bigint;
      memcpy(b_bigint.limbs, &point.x.b, BigInt::kByteNums);
      BasePrimeField b = BasePrimeField::FromMontgomery(b_bigint);
      x = BaseField(std::move(a), std::move(b));
    }
    BaseField y;
    {
      BigInt a_bigint;
      memcpy(a_bigint.limbs, &point.y.a, BigInt::kByteNums);
      BasePrimeField a = BasePrimeField::FromMontgomery(a_bigint);
      BigInt b_bigint;
      memcpy(b_bigint.limbs, &point.y.b, BigInt::kByteNums);
      BasePrimeField b = BasePrimeField::FromMontgomery(b_bigint);
      y = BaseField(std::move(a), std::move(b));
    }
    return {std::move(x), std::move(y)};
  }

  Groth16::VerificationKey<AltBn128::Engine> verification_key_;
  std::unique_ptr<BinFileUtils::BinFile> zkey_;
  std::unique_ptr<ZKeyUtils::Header> zkey_header_;
  std::unique_ptr<Groth16::Prover<Engine>> prover_;
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_BENCHMARK_RAPIDSNARK_RUNNER_H_
