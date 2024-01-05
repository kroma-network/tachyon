#ifndef TACHYON_ZK_PLONK_HALO2_PROOF_H_
#define TACHYON_ZK_PLONK_HALO2_PROOF_H_

#include <optional>
#include <utility>
#include <vector>

#include "tachyon/zk/lookup/lookup_pair.h"
#include "tachyon/zk/lookup/lookup_verification_data.h"
#include "tachyon/zk/plonk/permutation/permutation_verification_data.h"
#include "tachyon/zk/plonk/vanishing/vanishing_verification_data.h"

namespace tachyon::zk::halo2 {

template <typename F, typename C>
struct Proof {
  std::vector<std::vector<C>> advices_commitments_vec;
  std::vector<F> challenges;
  F theta;
  std::vector<std::vector<LookupPair<C>>> lookup_permuted_commitments_vec;
  F beta;
  F gamma;
  std::vector<std::vector<C>> permutation_product_commitments_vec;
  std::vector<std::vector<C>> lookup_product_commitments_vec;
  C vanishing_random_poly_commitment;
  F y;
  std::vector<C> vanishing_h_poly_commitments;
  F x;
  std::vector<std::vector<F>> instance_evals_vec;
  std::vector<std::vector<F>> advice_evals_vec;
  std::vector<F> fixed_evals;
  F vanishing_eval;
  std::vector<F> common_permutation_evals;
  std::vector<std::vector<F>> permutation_product_evals_vec;
  std::vector<std::vector<F>> permutation_product_next_evals_vec;
  std::vector<std::vector<std::optional<F>>> permutation_product_last_evals_vec;
  std::vector<std::vector<F>> lookup_product_evals_vec;
  std::vector<std::vector<F>> lookup_product_next_evals_vec;
  std::vector<std::vector<F>> lookup_permuted_input_evals_vec;
  std::vector<std::vector<F>> lookup_permuted_input_inv_evals_vec;
  std::vector<std::vector<F>> lookup_permuted_table_evals_vec;

  VanishingVerificationData<F> ToVanishingVerificationData(size_t i) const {
    VanishingVerificationData<F> ret;
    ret.fixed_evals = absl::MakeConstSpan(fixed_evals);
    ret.advice_evals = absl::MakeConstSpan(advice_evals_vec[i]);
    ret.instance_evals = absl::MakeConstSpan(instance_evals_vec[i]);
    ret.challenges = absl::MakeConstSpan(challenges);
    return ret;
  }

  PermutationVerificationData<F> ToPermutationVerificationData(
      size_t i, const F& l_first, const F& l_blind, const F& l_last) const {
    PermutationVerificationData<F> ret;
    ret.fixed_evals = absl::MakeConstSpan(fixed_evals);
    ret.advice_evals = absl::MakeConstSpan(advice_evals_vec[i]);
    ret.instance_evals = absl::MakeConstSpan(instance_evals_vec[i]);
    ret.challenges = absl::MakeConstSpan(challenges);
    ret.common_evals = absl::MakeConstSpan(common_permutation_evals);
    ret.product_evals = absl::MakeConstSpan(permutation_product_evals_vec[i]);
    ret.product_next_evals =
        absl::MakeConstSpan(permutation_product_next_evals_vec[i]);
    ret.product_last_evals =
        absl::MakeConstSpan(permutation_product_last_evals_vec[i]);
    ret.beta = beta;
    ret.gamma = gamma;
    ret.x = x;
    ret.l_first = l_first;
    ret.l_blind = l_blind;
    ret.l_last = l_last;
    return ret;
  }

  LookupVerificationData<F> ToLookupVerificationData(size_t i, size_t j,
                                                     const F& l_first,
                                                     const F& l_blind,
                                                     const F& l_last) const {
    LookupVerificationData<F> ret;
    ret.fixed_evals = absl::MakeConstSpan(fixed_evals);
    ret.advice_evals = absl::MakeConstSpan(advice_evals_vec[i]);
    ret.instance_evals = absl::MakeConstSpan(instance_evals_vec[i]);
    ret.challenges = absl::MakeConstSpan(challenges);
    ret.product_eval = lookup_product_evals_vec[i][j];
    ret.product_next_eval = lookup_product_next_evals_vec[i][j];
    ret.permuted_input_eval = lookup_permuted_input_evals_vec[i][j];
    ret.permuted_input_inv_eval = lookup_permuted_input_inv_evals_vec[i][j];
    ret.permuted_table_eval = lookup_permuted_table_evals_vec[i][j];
    ret.theta = theta;
    ret.beta = beta;
    ret.gamma = gamma;
    ret.x = x;
    ret.l_first = l_first;
    ret.l_blind = l_blind;
    ret.l_last = l_last;
    return ret;
  }
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_PROOF_H_
