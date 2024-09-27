#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_HINTABLE_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_HINTABLE_H_

#include <stddef.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_fields.h"
#include "tachyon/c/zk/air/sp1/block.h"
#include "tachyon/crypto/challenger/duplex_challenger.h"
#include "tachyon/crypto/commitments/fri/fri_proof.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::c::zk::air::sp1::baby_bear {

template <typename T, typename SFINAE = void>
class Hintable;

template <typename T>
auto WriteHint(T&& value) {
  return Hintable<std::decay_t<T>>::Write(std::forward<T>(value));
}

template <typename... Args>
size_t EstimateSize(const Args&... args) {
  return (... + Hintable<Args>::EstimateSize(args));
}

template <>
class Hintable<size_t> {
 public:
  static std::vector<std::vector<Block<F>>> Write(size_t value) {
    return {{Block<F>::From(F(static_cast<uint32_t>(value)))}};
  }

  constexpr static size_t EstimateSize(size_t) { return 1; }
};

template <>
class Hintable<Eigen::Index> {
 public:
  static std::vector<std::vector<Block<F>>> Write(Eigen::Index value) {
    return {{Block<F>::From(F(static_cast<uint32_t>(value)))}};
  }

  constexpr static size_t EstimateSize(size_t) { return 1; }
};

template <size_t N>
class Hintable<std::array<F, N>> {
 public:
  static std::vector<std::vector<Block<F>>> Write(
      const std::array<F, N>& values) {
    return {tachyon::base::Map(values,
                               [](F value) { return Block<F>::From(value); })};
  }

  constexpr static size_t EstimateSize(const std::array<F, N>& values) {
    return 1;
  }
};

template <>
class Hintable<std::vector<F>> {
 public:
  static std::vector<std::vector<Block<F>>> Write(
      const std::vector<F>& values) {
    return {tachyon::base::Map(values,
                               [](F value) { return Block<F>::From(value); })};
  }

  constexpr static size_t EstimateSize(const std::vector<F>& values) {
    return 1;
  }
};

template <>
class Hintable<math::Vector<F>> {
 public:
  static std::vector<std::vector<Block<F>>> Write(
      const math::Vector<F>& values) {
    return {tachyon::base::Map(values,
                               [](F value) { return Block<F>::From(value); })};
  }

  constexpr static size_t EstimateSize(const math::Vector<F>& values) {
    return 1;
  }
};

template <size_t N>
class Hintable<absl::InlinedVector<F, N>> {
 public:
  static std::vector<std::vector<Block<F>>> Write(
      const absl::InlinedVector<F, N>& values) {
    return {tachyon::base::Map(values,
                               [](F value) { return Block<F>::From(value); })};
  }

  constexpr static size_t EstimateSize(
      const absl::InlinedVector<F, N>& values) {
    return 1;
  }
};

template <typename T>
class Hintable<std::vector<T>> {
 public:
  static std::vector<std::vector<Block<F>>> Write(
      const std::vector<T>& values) {
    std::vector<std::vector<Block<F>>> ret;
    ret.reserve(EstimateSize(values));
    ret.push_back({Block<F>::From(F(values.size()))});
    for (size_t i = 0; i < values.size(); ++i) {
      tachyon::base::Extend(ret, WriteHint(values[i]));
    }
    return ret;
  }

  constexpr static size_t EstimateSize(const std::vector<T>& values) {
    return std::accumulate(values.begin(), values.end(), 1,
                           [](size_t total, const T& value) {
                             return total + baby_bear::EstimateSize(value);
                           });
  }
};

template <typename Params>
class Hintable<tachyon::crypto::SpongeState<Params>> {
 public:
  static std::vector<std::vector<Block<F>>> Write(
      const tachyon::crypto::SpongeState<Params>& value) {
    return WriteHint(value.elements);
  }

  constexpr static size_t EstimateSize(
      const tachyon::crypto::SpongeState<Params>& value) {
    return baby_bear::EstimateSize(value.elements);
  }
};

template <typename Permutation, size_t R>
class Hintable<tachyon::crypto::DuplexChallenger<Permutation, R>> {
 public:
  using Params = typename Permutation::Params;

  static std::vector<std::vector<Block<F>>> Write(
      const tachyon::crypto::DuplexChallenger<Permutation, R>& value) {
    std::vector<std::vector<Block<F>>> ret;
    ret.reserve(EstimateSize(value));
    tachyon::base::Extend(ret, WriteHint(value.state()));
    tachyon::base::Extend(ret, WriteHint(value.input_buffer().size()));
    absl::InlinedVector<F, R> input_buffer_padded = value.input_buffer();
    input_buffer_padded.resize(Params::kWidth, F::Zero());
    tachyon::base::Extend(ret, WriteHint(input_buffer_padded));
    tachyon::base::Extend(ret, WriteHint(value.output_buffer().size()));
    absl::InlinedVector<F, Params::kWidth> output_buffer_padded =
        value.output_buffer();
    output_buffer_padded.resize(Params::kWidth, F::Zero());
    tachyon::base::Extend(ret, WriteHint(output_buffer_padded));
    return ret;
  }

  constexpr static size_t EstimateSize(
      const tachyon::crypto::DuplexChallenger<Permutation, R>& value) {
    return baby_bear::EstimateSize(
        value.state(), value.input_buffer().size(), value.input_buffer(),
        value.output_buffer().size(), value.output_buffer());
  }
};

template <typename PCS>
class Hintable<tachyon::crypto::BatchOpening<PCS>> {
 public:
  static std::vector<std::vector<Block<F>>> Write(
      const tachyon::crypto::BatchOpening<PCS>& value) {
    std::vector<std::vector<Block<F>>> ret;
    ret.reserve(EstimateSize(value));
    tachyon::base::Extend(ret, WriteHint(value.opened_values));
    tachyon::base::Extend(ret, WriteHint(value.opening_proof));
    return ret;
  }

  constexpr static size_t EstimateSize(
      const tachyon::crypto::BatchOpening<PCS>& value) {
    return baby_bear::EstimateSize(value.opened_values, value.opening_proof);
  }
};

template <typename PCS>
class Hintable<tachyon::crypto::CommitPhaseProofStep<PCS>> {
 public:
  static std::vector<std::vector<Block<F>>> Write(
      const tachyon::crypto::CommitPhaseProofStep<PCS>& value) {
    std::vector<std::vector<Block<F>>> ret;
    ret.reserve(EstimateSize(value));
    ret.push_back({Block<F>::From(value.sibling_value)});
    tachyon::base::Extend(ret, WriteHint(value.opening_proof));
    return ret;
  }

  constexpr static size_t EstimateSize(
      const tachyon::crypto::CommitPhaseProofStep<PCS>& value) {
    return 1 + baby_bear::EstimateSize(value.opening_proof);
  }
};

template <typename PCS>
class Hintable<tachyon::crypto::QueryProof<PCS>> {
 public:
  static std::vector<std::vector<Block<F>>> Write(
      const tachyon::crypto::QueryProof<PCS>& value) {
    std::vector<std::vector<Block<F>>> ret;
    ret.reserve(EstimateSize(value));
    // NOTE(chokobole): |input_proof| shouldn't be included here.
    // See
    // https://github.com/succinctlabs/sp1/blob/6f67afd/crates/recursion/program/src/fri/hints.rs#L123-L129.
    tachyon::base::Extend(ret, WriteHint(value.commit_phase_openings));
    return ret;
  }

  constexpr static size_t EstimateSize(
      const tachyon::crypto::QueryProof<PCS>& value) {
    return baby_bear::EstimateSize(value.commit_phase_openings);
  }
};

template <typename PCS>
class Hintable<tachyon::crypto::FRIProof<PCS>> {
 public:
  static std::vector<std::vector<Block<F>>> Write(
      const tachyon::crypto::FRIProof<PCS>& value) {
    std::vector<std::vector<Block<F>>> ret;
    size_t ret_size = EstimateSize(value);
    ret.reserve(ret_size);
    tachyon::base::Extend(ret, WriteHint(value.commit_phase_commits));
    tachyon::base::Extend(ret, WriteHint(value.query_proofs));
    ret.push_back({Block<F>::From(value.final_eval)});
    ret.push_back({Block<F>::From(value.pow_witness)});
    // NOTE(chokobole): |query_proofs[i].input_proof| should be included here.
    // See
    // https://github.com/succinctlabs/sp1/blob/6f67afd/crates/recursion/program/src/fri/hints.rs#L279.
    std::vector<std::vector<Block<F>>> query_openings;
    size_t query_openings_size = std::accumulate(
        value.query_proofs.begin(), value.query_proofs.end(), 1,
        [](size_t total, const tachyon::crypto::QueryProof<PCS>& proof) {
          return total + baby_bear::EstimateSize(proof.input_proof);
        });
    query_openings.reserve(query_openings_size);
    query_openings.push_back({Block<F>::From(F(value.query_proofs.size()))});
    for (size_t i = 0; i < value.query_proofs.size(); ++i) {
      tachyon::base::Extend(query_openings,
                            WriteHint(value.query_proofs[i].input_proof));
    }
    CHECK_EQ(query_openings.size(), query_openings_size);
    tachyon::base::Extend(ret, std::move(query_openings));
    CHECK_EQ(ret.size(), ret_size);
    return ret;
  }

  constexpr static size_t EstimateSize(
      const tachyon::crypto::FRIProof<PCS>& value) {
    size_t query_openings_size = std::accumulate(
        value.query_proofs.begin(), value.query_proofs.end(), 1,
        [](size_t total, const tachyon::crypto::QueryProof<PCS>& proof) {
          return total + baby_bear::EstimateSize(proof.input_proof);
        });
    return baby_bear::EstimateSize(value.commit_phase_commits,
                                   value.query_proofs) +
           2 + query_openings_size;
  }
};

}  // namespace tachyon::c::zk::air::sp1::baby_bear

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_HINTABLE_H_
