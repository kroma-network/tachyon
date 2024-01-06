// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_H_

#include <numeric>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"

namespace tachyon::crypto {

// Specifying the output field element size.
class TACHYON_EXPORT FieldElementSize {
 public:
  static FieldElementSize Full() { return FieldElementSize(false); }
  static FieldElementSize Truncated(size_t num_bits) {
    return {true, num_bits};
  }

  template <typename PrimeField>
  size_t NumBits() {
    static_assert(math::FiniteFieldTraits<PrimeField>::kIsPrimeField,
                  "NumBits() is only supported for PrimeField");
    if (is_truncated_) {
      CHECK_LE(num_bits_, PrimeField::kModulusBits)
          << "num_bits is greater than the bit size of the field.";
      return num_bits_;
    }
    return PrimeField::kModulusBits - 1;
  }

  // Calculate the sum of prime field element sizes in |elements|.
  template <typename PrimeField>
  static size_t Sum(const std::vector<FieldElementSize>& elements) {
    static_assert(math::FiniteFieldTraits<PrimeField>::kIsPrimeField,
                  "Sum() is only supported for PrimeField");
    return (PrimeField::kModulusBits - 1) * elements.size();
  }

  bool IsFull() const { return !is_truncated_; }
  bool IsTruncated() const { return is_truncated_; }

 private:
  explicit FieldElementSize(bool is_truncated)
      : FieldElementSize(is_truncated, 0) {}
  FieldElementSize(bool is_truncated, size_t num_bits)
      : is_truncated_(is_truncated), num_bits_(num_bits) {}

  // If |is_truncated_| is false, sample field elements from the entire field.
  // If |is_truncated_| is true, sample field elements from a subset of the
  // field, specified by the maximum number of bit.
  bool is_truncated_;
  size_t num_bits_ = 0;
};

template <typename T>
struct CryptographicSpongeTraits;

// The interface for a cryptographic sponge.
// A sponge can |Absorb| and later |Squeeze| bytes of field elements.
// The outputs are dependent on previous |Absorb| and |Squeeze| calls.
template <typename Derived>
class CryptographicSponge {
 public:
  using F = typename CryptographicSpongeTraits<Derived>::F;

  // Squeeze |num_elements| nonnative field elements from the sponge.
  std::vector<F> SqueezeFieldElements(size_t num_elements) {
    Derived* derived = static_cast<Derived*>(this);
    return derived->SqueezeFieldElementsWithSizes(base::CreateVector(
        num_elements, []() { return FieldElementSize::Full(); }));
  }

  // Creates a new sponge with applied domain separation.
  Derived Fork(absl::Span<const uint8_t> domain) const {
    const Derived* derived = static_cast<const Derived*>(this);
    CHECK(derived->Absorb(domain));
    return *derived;
  }

 protected:
  std::vector<F> SqueezeFieldElementsWithSizesDefaultImpl(
      const std::vector<FieldElementSize>& sizes) {
    if constexpr (math::FiniteFieldTraits<F>::kIsPrimeField) {
      if (sizes.empty()) {
        return {};
      }

      size_t total_num_bits = FieldElementSize::Sum<F>(sizes);

      Derived derived = static_cast<Derived*>(this);
      std::vector<bool> bits = derived->SqueezeBits(total_num_bits);
      auto bits_window = bits.begin();

      std::vector<F> output;
      output.reserve(sizes.size());
      for (const FieldElementSize& size : sizes) {
        size_t num_bits = size.NumBits<F>();

        std::bitset<F::kModulusBits> field_element_bits;
        for (size_t i = 0; i < num_bits; ++i) {
          field_element_bits[i] = *(bits_window + i);
        }
        bits_window += num_bits;

        output.push_back(F::FromBitsLE(field_element_bits));
      }
      return output;
    } else {
      NOTIMPLEMENTED();
      return {};
    }
  }
};

// The interface for field-based cryptographic sponge.
template <typename Derived>
class FieldBasedCryptographicSponge : public CryptographicSponge<Derived> {
 public:
  using NativeField = typename CryptographicSpongeTraits<Derived>::F;

  // Squeeze |sizes.size()| field elements from the sponge.
  // where the |i|-th element of the output has |sizes[i]|.
  std::vector<NativeField> SqueezeNativeFieldElementsWithSizes(
      const std::vector<FieldElementSize>& sizes) {
    bool all_full_size =
        std::all_of(sizes.begin(), sizes.end(),
                    [](const FieldElementSize& size) { return size.IsFull(); });
    Derived* derived = static_cast<Derived*>(this);
    if (all_full_size) {
      return derived->SqueezeNativeFieldElements(sizes.size());
    } else {
      return derived->SqueezeFieldElementsWithSizesDefaultImpl(sizes);
    }
  }
};

// The mode structure for duplex sponge.
struct TACHYON_EXPORT DuplexSpongeMode {
  enum class Type {
    // The sponge is currently absorbing data.
    kAbsorbing,
    // The sponge is currently squeezing data out.
    kSqueezing,
  };

  constexpr static DuplexSpongeMode Absorbing(size_t next_index = 0) {
    return {Type::kAbsorbing, next_index};
  }
  constexpr static DuplexSpongeMode Squeezing(size_t next_index = 0) {
    return {Type::kSqueezing, next_index};
  }

  Type type;
  // When |type| is |kAbsorbing|, it is interpreted as next position of the
  // state to be XOR-ed when absorbing.
  // When |type| is |kSqueezing|, it is interpreted as next position of the
  // state to be outputted when squeezing.
  size_t next_index;

 private:
  constexpr DuplexSpongeMode(Type type, size_t next_index)
      : type(type), next_index(next_index) {}
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_H_
