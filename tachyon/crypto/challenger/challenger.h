// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_CHALLENGER_CHALLENGER_H_
#define TACHYON_CRYPTO_CHALLENGER_CHALLENGER_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/base/range.h"
#include "tachyon/crypto/challenger/challenger_traits_forward.h"

namespace tachyon::crypto {

template <typename Derived>
class Challenger {
 public:
  using Field = typename ChallengerTraits<Derived>::Field;

  template <typename T>
  void Observe(const T& value) {
    Derived* derived = static_cast<Derived*>(this);
    derived->DoObserve(value);
  }

  template <typename Container>
  void ObserveContainer(const Container& container) {
    Derived* derived = static_cast<Derived*>(this);
    for (const auto& value : container) {
      derived->DoObserve(value);
    }
  }

  template <typename Container>
  void ObserveContainer2D(const Container& container_2d) {
    Derived* derived = static_cast<Derived*>(this);
    for (const auto& container : container_2d) {
      for (const auto& value : container) {
        derived->DoObserve(value);
      }
    }
  }

  Field Sample() {
    Derived* derived = static_cast<Derived*>(this);
    return derived->DoSample();
  }

  template <size_t N>
  std::array<Field, N> SampleArray() {
    return base::CreateArray<N>([this]() { return Sample(); });
  }

  template <typename ExtField>
  ExtField SampleExtElement() {
    constexpr size_t N = ExtField::kDegreeOverBasePrimeField;
    using F = typename ExtField::BasePrimeField;
    static_assert(std::is_same_v<F, Field>);
    std::array<F, N> prime_fields = SampleArray<N>();
    return ExtField::FromBasePrimeFields(prime_fields);
  }

  uint32_t SampleBits(uint32_t bits) {
    static_assert(Field::Config::kModulusBits <= 32);
    DCHECK_LT(bits, sizeof(uint32_t) * 8);
    DCHECK_LT(uint32_t{1} << bits, Field::Config::kModulus);
    Field rand_f = Sample();
    uint32_t rand_size;
    if constexpr (Field::Config::kUseMontgomery) {
      rand_size = Field::Config::FromMontgomery(rand_f.value());
    } else {
      rand_size = rand_f.value();
    }
    return rand_size & ((uint32_t{1} << bits) - 1);
  }

  Field Grind(uint32_t bits) {
    return Grind(bits, base::Range<uint32_t>::Until(Field::Config::kModulus));
  }

  Field Grind(uint32_t bits, base::Range<uint32_t> range) {
    std::vector<uint32_t> ret = base::ParallelizeMap(
        range.GetSize(),
        [this, bits, range](uint32_t len, uint32_t chunk_offset,
                            uint32_t chunk_size) {
          uint32_t ret = std::numeric_limits<uint32_t>::max();
          uint32_t start = range.from + chunk_offset * chunk_size;
          uint32_t end = start + std::min(range.to - start, chunk_size);
          Field f(start);
          for (uint32_t i = start; i < end; ++i) {
            Derived derived = *static_cast<Derived*>(this);
            if (derived.CheckWitness(bits, f)) {
              ret = i;
              break;
            }
            f += Field::One();
          }
          return ret;
        });
    auto it = std::find_if(ret.begin(), ret.end(), [](uint32_t v) {
      return v != std::numeric_limits<uint32_t>::max();
    });
    CHECK(it != ret.end());
    CHECK(CheckWitness(bits, Field(*it)));
    return Field(*it);
  }

  [[nodiscard]] bool CheckWitness(uint32_t bits, const Field& witness) {
    Observe(witness);
    return SampleBits(bits) == 0;
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_CHALLENGER_CHALLENGER_H_
