#ifndef TACHYON_ZK_PLONK_HALO2_PROOF_H_
#define TACHYON_ZK_PLONK_HALO2_PROOF_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/json/json.h"
#include "tachyon/zk/lookup/lookup_pair.h"
#include "tachyon/zk/lookup/lookup_verification_data.h"
#include "tachyon/zk/plonk/permutation/permutation_verification_data.h"
#include "tachyon/zk/plonk/vanishing/vanishing_verification_data.h"

namespace tachyon {
namespace zk::plonk::halo2 {

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
  F vanishing_random_eval;
  std::vector<F> common_permutation_evals;
  std::vector<std::vector<F>> permutation_product_evals_vec;
  std::vector<std::vector<F>> permutation_product_next_evals_vec;
  std::vector<std::vector<std::optional<F>>> permutation_product_last_evals_vec;
  std::vector<std::vector<F>> lookup_product_evals_vec;
  std::vector<std::vector<F>> lookup_product_next_evals_vec;
  std::vector<std::vector<F>> lookup_permuted_input_evals_vec;
  std::vector<std::vector<F>> lookup_permuted_input_inv_evals_vec;
  std::vector<std::vector<F>> lookup_permuted_table_evals_vec;

  // auxiliary values
  F l_first;
  F l_blind;
  F l_last;
  F x_next;
  F x_prev;
  F x_last;
  F x_n;

  bool operator==(const Proof& other) const {
    return advices_commitments_vec == other.advices_commitments_vec &&
           challenges == other.challenges && theta == other.theta &&
           lookup_permuted_commitments_vec ==
               other.lookup_permuted_commitments_vec &&
           beta == other.beta && gamma == other.gamma &&
           permutation_product_commitments_vec ==
               other.permutation_product_commitments_vec &&
           lookup_product_commitments_vec ==
               other.lookup_product_commitments_vec &&
           vanishing_random_poly_commitment ==
               other.vanishing_random_poly_commitment &&
           y == other.y &&
           vanishing_h_poly_commitments == other.vanishing_h_poly_commitments &&
           x == other.x && instance_evals_vec == other.instance_evals_vec &&
           advice_evals_vec == other.advice_evals_vec &&
           fixed_evals == other.fixed_evals &&
           vanishing_random_eval == other.vanishing_random_eval &&
           common_permutation_evals == other.common_permutation_evals &&
           permutation_product_evals_vec ==
               other.permutation_product_evals_vec &&
           permutation_product_next_evals_vec ==
               other.permutation_product_next_evals_vec &&
           permutation_product_last_evals_vec ==
               other.permutation_product_last_evals_vec &&
           lookup_product_evals_vec == other.lookup_product_evals_vec &&
           lookup_product_next_evals_vec ==
               other.lookup_product_next_evals_vec &&
           lookup_permuted_input_evals_vec ==
               other.lookup_permuted_input_evals_vec &&
           lookup_permuted_input_inv_evals_vec ==
               other.lookup_permuted_input_inv_evals_vec &&
           lookup_permuted_table_evals_vec ==
               other.lookup_permuted_table_evals_vec;
  }
  bool operator!=(const Proof& other) const { return !operator==(other); }

  VanishingVerificationData<F> ToVanishingVerificationData(size_t i) const {
    VanishingVerificationData<F> ret;
    ret.fixed_evals = fixed_evals;
    ret.advice_evals = advice_evals_vec[i];
    ret.instance_evals = instance_evals_vec[i];
    ret.challenges = challenges;
    return ret;
  }

  PermutationVerificationData<F, C> ToPermutationVerificationData(
      size_t i) const {
    PermutationVerificationData<F, C> ret;
    ret.fixed_evals = fixed_evals;
    ret.advice_evals = advice_evals_vec[i];
    ret.instance_evals = instance_evals_vec[i];
    ret.challenges = challenges;
    ret.product_commitments = permutation_product_commitments_vec[i];
    ret.common_evals = common_permutation_evals;
    ret.product_evals = permutation_product_evals_vec[i];
    ret.product_next_evals = permutation_product_next_evals_vec[i];
    ret.product_last_evals = permutation_product_last_evals_vec[i];
    ret.beta = &beta;
    ret.gamma = &gamma;
    ret.x = &x;
    ret.x_next = &x_next;
    ret.x_last = &x_last;
    ret.l_first = &l_first;
    ret.l_blind = &l_blind;
    ret.l_last = &l_last;
    return ret;
  }

  LookupVerificationData<F, C> ToLookupVerificationData(size_t i,
                                                        size_t j) const {
    LookupVerificationData<F, C> ret;
    ret.fixed_evals = fixed_evals;
    ret.advice_evals = advice_evals_vec[i];
    ret.instance_evals = instance_evals_vec[i];
    ret.challenges = challenges;
    ret.permuted_commitment = &lookup_permuted_commitments_vec[i][j];
    ret.product_commitment = &lookup_product_commitments_vec[i][j];
    ret.product_eval = &lookup_product_evals_vec[i][j];
    ret.product_next_eval = &lookup_product_next_evals_vec[i][j];
    ret.permuted_input_eval = &lookup_permuted_input_evals_vec[i][j];
    ret.permuted_input_inv_eval = &lookup_permuted_input_inv_evals_vec[i][j];
    ret.permuted_table_eval = &lookup_permuted_table_evals_vec[i][j];
    ret.theta = &theta;
    ret.beta = &beta;
    ret.gamma = &gamma;
    ret.x = &x;
    ret.x_next = &x_next;
    ret.x_prev = &x_prev;
    ret.l_first = &l_first;
    ret.l_blind = &l_blind;
    ret.l_last = &l_last;
    return ret;
  }
};

}  // namespace zk::plonk::halo2

namespace base {

template <typename F, typename C>
class RapidJsonValueConverter<zk::plonk::halo2::Proof<F, C>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const zk::plonk::halo2::Proof<F, C>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "advices_commitments_vec",
                   value.advices_commitments_vec, allocator);
    AddJsonElement(object, "challenges", value.challenges, allocator);
    AddJsonElement(object, "theta", value.theta, allocator);
    AddJsonElement(object, "lookup_permuted_commitments_vec",
                   value.lookup_permuted_commitments_vec, allocator);
    AddJsonElement(object, "beta", value.beta, allocator);
    AddJsonElement(object, "gamma", value.gamma, allocator);
    AddJsonElement(object, "permutation_product_commitments_vec",
                   value.permutation_product_commitments_vec, allocator);
    AddJsonElement(object, "lookup_product_commitments_vec",
                   value.lookup_product_commitments_vec, allocator);
    AddJsonElement(object, "vanishing_random_poly_commitment",
                   value.vanishing_random_poly_commitment, allocator);
    AddJsonElement(object, "y", value.y, allocator);
    AddJsonElement(object, "vanishing_h_poly_commitments",
                   value.vanishing_h_poly_commitments, allocator);
    AddJsonElement(object, "x", value.x, allocator);
    AddJsonElement(object, "instance_evals_vec", value.instance_evals_vec,
                   allocator);
    AddJsonElement(object, "advice_evals_vec", value.advice_evals_vec,
                   allocator);
    AddJsonElement(object, "fixed_evals", value.fixed_evals, allocator);
    AddJsonElement(object, "vanishing_random_eval", value.vanishing_random_eval,
                   allocator);
    AddJsonElement(object, "common_permutation_evals",
                   value.common_permutation_evals, allocator);
    AddJsonElement(object, "permutation_product_evals_vec",
                   value.permutation_product_evals_vec, allocator);
    AddJsonElement(object, "permutation_product_next_evals_vec",
                   value.permutation_product_next_evals_vec, allocator);
    AddJsonElement(object, "permutation_product_last_evals_vec",
                   value.permutation_product_last_evals_vec, allocator);
    AddJsonElement(object, "lookup_product_evals_vec",
                   value.lookup_product_evals_vec, allocator);
    AddJsonElement(object, "lookup_product_next_evals_vec",
                   value.lookup_product_next_evals_vec, allocator);
    AddJsonElement(object, "lookup_permuted_input_evals_vec",
                   value.lookup_permuted_input_evals_vec, allocator);
    AddJsonElement(object, "lookup_permuted_input_inv_evals_vec",
                   value.lookup_permuted_input_inv_evals_vec, allocator);
    AddJsonElement(object, "lookup_permuted_table_evals_vec",
                   value.lookup_permuted_table_evals_vec, allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 zk::plonk::halo2::Proof<F, C>* proof_out, std::string* error) {
    zk::plonk::halo2::Proof<F, C> proof;
    if (!ParseJsonElement(json_value, "advices_commitments_vec",
                          &proof.advices_commitments_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "challenges", &proof.challenges, error))
      return false;
    if (!ParseJsonElement(json_value, "theta", &proof.theta, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_permuted_commitments_vec",
                          &proof.lookup_permuted_commitments_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "beta", &proof.beta, error)) return false;
    if (!ParseJsonElement(json_value, "gamma", &proof.gamma, error))
      return false;
    if (!ParseJsonElement(json_value, "permutation_product_commitments_vec",
                          &proof.permutation_product_commitments_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_product_commitments_vec",
                          &proof.lookup_product_commitments_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "vanishing_random_poly_commitment",
                          &proof.vanishing_random_poly_commitment, error))
      return false;
    if (!ParseJsonElement(json_value, "y", &proof.y, error)) return false;
    if (!ParseJsonElement(json_value, "vanishing_h_poly_commitments",
                          &proof.vanishing_h_poly_commitments, error))
      return false;
    if (!ParseJsonElement(json_value, "x", &proof.x, error)) return false;
    if (!ParseJsonElement(json_value, "instance_evals_vec",
                          &proof.instance_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "advice_evals_vec",
                          &proof.advice_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "fixed_evals", &proof.fixed_evals, error))
      return false;
    if (!ParseJsonElement(json_value, "vanishing_random_eval",
                          &proof.vanishing_random_eval, error))
      return false;
    if (!ParseJsonElement(json_value, "common_permutation_evals",
                          &proof.common_permutation_evals, error))
      return false;
    if (!ParseJsonElement(json_value, "permutation_product_evals_vec",
                          &proof.permutation_product_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "permutation_product_next_evals_vec",
                          &proof.permutation_product_next_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "common_permutation_evals",
                          &proof.common_permutation_evals, error))
      return false;
    if (!ParseJsonElement(json_value, "permutation_product_evals_vec",
                          &proof.permutation_product_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "permutation_product_next_evals_vec",
                          &proof.permutation_product_next_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "permutation_product_last_evals_vec",
                          &proof.permutation_product_last_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_product_evals_vec",
                          &proof.lookup_product_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_product_next_evals_vec",
                          &proof.lookup_product_next_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_permuted_input_evals_vec",
                          &proof.lookup_permuted_input_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_permuted_input_inv_evals_vec",
                          &proof.lookup_permuted_input_inv_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_permuted_table_evals_vec",
                          &proof.lookup_permuted_table_evals_vec, error))
      return false;

    *proof_out = std::move(proof);
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_HALO2_PROOF_H_
