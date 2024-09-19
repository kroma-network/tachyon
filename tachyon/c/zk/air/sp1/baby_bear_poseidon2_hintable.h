#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_HINTABLE_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_HINTABLE_H_

#include <stddef.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_fields.h"
#include "tachyon/c/zk/air/sp1/block.h"
#include "tachyon/crypto/challenger/duplex_challenger.h"
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

}  // namespace tachyon::c::zk::air::sp1::baby_bear

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_HINTABLE_H_
