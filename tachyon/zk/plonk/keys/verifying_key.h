#ifndef TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_
#define TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_

#include <stddef.h>

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/circuit/assembly.h"
#include "tachyon/zk/plonk/constraint_system.h"
#include "tachyon/zk/plonk/permutation/permutation_verifying_key.h"

namespace tachyon::zk {

template <typename PCSTy>
class VerifyingKey {
 public:
  constexpr static size_t kMaxDegree = PCSTy::kMaxDegree;

  using F = typename PCSTy::Field;
  using Domain = typename PCSTy::Domain;
  using Commitment = typename PCSTy::Commitment;
  using Commitments = std::vector<Commitment>;

  VerifyingKey() = default;
  VerifyingKey(
      std::unique_ptr<math::UnivariateEvaluationDomain<F, kMaxDegree>> domain,
      Commitments fixed_commitments,
      PermutationVerifyingKey<PCSTy> permutation_verifying_key,
      ConstraintSystem<F> constraint_system)
      : domain_(std::move(domain)),
        fixed_commitments_(std::move(fixed_commitments)),
        permutation_verifying_Key_(std::move(permutation_verifying_key)),
        constraint_system_(std::move(constraint_system)) {}

  static VerifyingKey FromParts(
      std::unique_ptr<math::UnivariateEvaluationDomain<F, kMaxDegree>> domain,
      Commitments fixed_commitments,
      PermutationVerifyingKey<PCSTy> permutation_verifying_key,
      ConstraintSystem<F> constraint_system) {
    VerifyingKey ret(std::move(domain), std::move(fixed_commitments),
                     std::move(permutation_verifying_key),
                     std::move(constraint_system));
    // TODO(chokobole): Implement blake transcript.
    // See
    // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk.rs#L176-L211.
    return ret;
  }

  template <typename CircuitTy>
  static Error Generate(const PCSTy& pcs, const CircuitTy& circuit,
                        VerifyingKey* verifying_key);

  const math::UnivariateEvaluationDomain<F, kMaxDegree>* domain() const {
    return domain_.get();
  }

  const Commitments& fixed_commitments() const { return fixed_commitments_; }

  const PermutationVerifyingKey<PCSTy>& permutation_verifying_key() const {
    return permutation_verifying_Key_;
  }

  const ConstraintSystem<F>& constraint_system() const {
    return constraint_system_;
  }

  const F& transcript_repr() const { return transcript_repr_; }

 private:
  std::unique_ptr<Domain> domain_;
  Commitments fixed_commitments_;
  PermutationVerifyingKey<PCSTy> permutation_verifying_Key_;
  ConstraintSystem<F> constraint_system_;
  // The representative of this |VerifyingKey| in transcripts.
  F transcript_repr_ = F::Zero();
};

// static
template <typename PCSTy>
template <typename CircuitTy>
Error VerifyingKey<PCSTy>::Generate(const PCSTy& pcs, const CircuitTy& circuit,
                                    VerifyingKey* verifying_key) {
  using Config = typename CircuitTy::Config;
  using FloorPlanner = typename CircuitTy::FloorPlanner;
  using DomainTy = math::UnivariateEvaluationDomain<F, kMaxDegree>;
  using DensePoly =
      math::UnivariateDensePolynomial<math::RationalField<F>, kMaxDegree>;
  using Evals = math::UnivariateEvaluations<F, kMaxDegree>;

  ConstraintSystem<F> constraint_system;
  Config config = CircuitTy::Configure(constraint_system);
  std::unique_ptr<DomainTy> domain =
      math::UnivariateEvaluationDomainFactory<F, kMaxDegree>::Create(pcs.n());

  if (pcs.N() < constraint_system.ComputeMinimumRows()) {
    return Error::kNotEnoughRowsAvailable;
  }

  Assembly<PCSTy> assembly(
      pcs.K(),
      base::CreateVector(constraint_system.num_fixed_columns(),
                         DensePoly::Zero()),
      PermutationAssembly<PCSTy>(constraint_system.permutation()),
      base::CreateVector(constraint_system.num_selectors(),
                         base::CreateVector(pcs.N(), false)),
      base::Range<size_t>::Until(
          pcs.N() - (constraint_system.ComputeBlindingFactors() + 1)));

  Error error =
      FloorPlanner::Synthesize(&assembly, constraint_system.constants());
  if (error != Error::kNone) return error;

  std::vector<Evals> fixeds =
      base::Map(assembly.fixeds(), [](const DensePoly& poly) {
        std::vector<F> result;
        CHECK(math::RationalField<F>::BatchEvaluate(poly.coefficients(),
                                                    &result));
        return Evals(std::move(result));
      });

  // TODO(chokobole): Implement selector compression.
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/keygen.rs#L236-L241.

  PermutationVerifyingKey<PCSTy> permutation_vk =
      assembly.permutation().BuildVerifyingKey(domain.get());

  // TODO(chokobole): Parallelize this.
  Commitments fixed_commitments = base::Map(fixeds, [&pcs](const Evals& evals) {
    Commitment commitment;
    CHECK(pcs.CommitLagrange(evals, &commitment));
    return commitment;
  });

  *verifying_key = VerifyingKey::FromParts(
      std::move(domain), std::move(fixed_commitments),
      std::move(permutation_vk), std::move(constraint_system));
  return Error::kNone;
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_
