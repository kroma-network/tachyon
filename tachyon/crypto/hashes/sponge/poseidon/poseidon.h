#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_

#include <vector>

// #include "tachyon/crypto/hashes/sponge/sponge.h"
#include "tachyon/math/finite_fields/prime_field_forward.h"

namespace tachyon::crypto {

template <typename PrimeFieldTy>
class PoseidonConfig {
 public:
  size_t full_rounds;
  size_t partial_rounds;
  uint64_t alpha;
  std::vector<std::vector<PrimeFieldTy>> ark;
  std::vector<std::vector<PrimeFieldTy>> mds;
  size_t rate;
  size_t capacity;
  PoseidonConfig(size_t full_rounds_, size_t partial_rounds_, uint64_t alpha_,
                 std::vector<std::vector<PrimeFieldTy>> ark_,
                 std::vector<std::vector<PrimeFieldTy>> mds_, size_t rate_,
                 size_t capacity_)
      : full_rounds(full_rounds_),
        partial_rounds(partial_rounds_),
        alpha(alpha_),
        ark(ark_),
        mds(mds_),
        rate(rate_),
        capacity(capacity_) {}
};

// template <typename Config>
// class PoseidonSponge {
//  public:
//   using PrimeField = tachyon::math::PrimeField<Config>;
//   PoseidonSponge(const PoseidonConfig<Config>& parameters_)
//       : parameters_(parameters), mode_(DuplexSpongeMode::Absorbing) {}

//   void ApplySBox(std::vector<PrimeField>& state_, bool is_full_round) {
//     if (is_full_round) {
//       for (auto& element : state_) {
//         element = element.Pow(parameters_.alpha);
//       }
//     } else {
//       state_[0] = state_[0].Pow(parameters_.alpha);
//     }
//   }

//   void ApplyArk(std::vector<PrimeField>& state_, size_t round_number) {
//     for (size_t i = 0; i < state_.size(); ++i) {
//       state_[i] += parameters_.ark[round_number][i];
//     }
//   }

//   void ApplyMds(std::vector<PrimeField>& state_) {
//     std::vector<PrimeField> new_state(state_.size());
//     for (size_t i = 0; i < state_.size(); ++i) {
//       PrimeField cur = PrimeField::Zero();
//       for (size_t j = 0; j < state_.size(); ++j) {
//         cur[i] += state_[j] * parameters_.mds[i][j];
//       }
//       new_state[i] = cur;
//     }
//     state_ = std::move(new_state);
//   }

//   void Permute() {
//     size_t full_rounds_over_2 = parameters_.full_rounds / 2;
//     std::vector<PrimeField> state = state_;

//     for (size_t i = 0; i < full_rounds_over_2; ++i) {
//       ApplySBox(state, true);
//       ApplyArk(state, i);
//       ApplyMds(state);
//     }

//     for (size_t i = full_rounds_over_2;
//          i < full_rounds_over_2 + parameters_.partial_rounds; ++i) {
//       ApplySBox(state, false);
//       ApplyArk(state, i);
//       ApplyMds(state);
//     }

//     for (size_t i = full_rounds_over_2 + parameters_.partial_rounds;
//          i < parameters_.full_rounds; ++i) {
//       ApplySBox(state, true);
//       ApplyArk(state, i);
//       ApplyMds(state);
//     }

//     state_ = state;
//   }

//   void AbsorbInternal(size_t rate_start_index,
//                       const std::vector<PrimeField>& elements) {
//     std::vector<PrimeField> remaining_elements = elements;
//     while (true) {
//       if (rate_start_index + remaining_elements.size() <= parameters_.rate) {
//         for (size_t i = 0; i < remaining_elements.size(); ++i) {
//           state_[parameters_.capacity + i + rate_start_index] +=
//               remaining_elements[i];
//         }
//         mode_ = DuplexSpongeMode::Absorbing(rate_start_index +
//                                             remaining_elements.size());
//         return;
//       }

//       size_t num_elements_absorbed = parameters_.rate - rate_start_index;
//       for (size_t i = 0; i < num_elements_absorbed; ++i) {
//         state_[parameters_.capacity + i + rate_start_index] +=
//             remaining_elements[i];
//       }
//       Permute();
//       remaining_elements.erase(
//           remaining_elements.begin(),
//           remaining_elements.begin() + num_elements_absorbed);
//       rate_start_index = 0;
//     }
//   }

//   void SqueezeInternal(size_t rate_start_index,
//                        std::vector<PrimeField> output) {
//     size_t output_size = output.size();
//     size_t output_idx = 0;
//     while (true) {
//       if (rate_start_index + (output_size - output_idx) <= parameters.rate) {
//         for (size_t i = 0; i < (output_size - output_idx); ++i) {
//           output[output_idx + i] =
//               state[parameters.capacity + rate_start_index + i];
//         }
//         mode_ = DuplexSpongeMode::Squeezing(rate_start_index + output_size -
//                                             output_idx);
//         return;
//       }

//       size_t num_elements_squeezed = parameters.rate - rate_start_index;
//       for (size_t i = 0; i < num_elements_squeezed; ++i) {
//         output[output_idx + i] =
//             state[parameters.capacity + rate_start_index + i];
//       }

//       if (output_size - output_idx != parameters.rate) {
//         Permute();
//       }
//       output_idx += num_elements_squeezed;
//       rate_start_index = 0;
//     }
//   }

//  private:
//   PoseidonConfig<Config> parameters_;
//   std::vector<tachyon::math::PrimeField<Config>> state_;
//   DuplexSpongeMode mode_;
// };

// // CryptographicSponge for Poseidon.

// // FieldBasedCryptographicSponge for Poseidon.

// // PoseidonSpongeState

// // SpongeExt PoseidonSpongeSponge

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_
