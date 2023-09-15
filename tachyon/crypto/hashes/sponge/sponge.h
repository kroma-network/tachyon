#ifndef TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_H_

#include <vector>

#include "tachyon/crypto/hashes/bytes_serializable.h"
#include "tachyon/crypto/hashes/prime_field_serializable.h"
#include "tachyon/math/finite_fields/prime_field_forward.h"

namespace tachyon::crypto {

using namespace math;

template <typename Config>
class FieldElementSize;

template <typename SpongeConfig, typename PrimeFieldTy>
class CryptographicSponge;

template <typename SpongeConfig, typename PrimeFieldTy>
class FieldBasedCryptographicSponge;

template <typename SpongeConfig, typename PrimeFieldTy>
ALWAYS_INLINE std::vector<PrimeField<PrimeFieldTy>>
SqueezeFieldElementsWithSizesDefaultImpl(
    CryptographicSponge<SpongeConfig, PrimeField<PrimeFieldTy>>& sponge,
    std::vector<FieldElementSize<PrimeField<PrimeFieldTy>>> sizes) {
  using F = PrimeField<PrimeFieldTy>;

  if (sizes.size() == 0) {
    return std::vector<F>();
  }

  size_t total_bits = 0;
  for (const auto& size : sizes) {
    total_bits += size.GetNumBits();
  }

  std::vector<bool> bits = sponge.SqueezeBits(total_bits);
  std::vector<bool>::iterator bits_window_begin = bits.begin();
  std::vector<bool>::iterator bits_window_end = bits.begin();

  std::vector<F> output;
  output.reserve(sizes.size());
  for (const auto& size : sizes) {
    bits_window_end += size.GetNumBits();
    std::vector<uint8_t> nonnative_bytes;
    for (auto it = bits_window_begin; it < bits_window_end; it += 8) {
      uint8_t byte = 0;
      auto end = std::min(it + 8, bits_window_end);
      for (auto bit_it = it; bit_it != end; ++bit_it) {
        if (*bit_it) {
          byte += (1 << (bit_it - it));
        }
      }
      nonnative_bytes.push_back(byte);
    }
    std::stringstream ss;
    for (uint8_t byte : nonnative_bytes) {
      ss << std::setw(2) << std::setfill('0') << std::hex
         << static_cast<int>(byte);
    }
    std::string hex_str = ss.str();
    output.push_back(F::FromHexString(hex_str));
    // TODO(insun35): Implement FromBytes in PrimeField
  }

  assert(bits_window_begin == bits.end());
  return output;
}

template <typename PrimeFieldTy>
class FieldElementSize<PrimeField<PrimeFieldTy>> {
 public:
  using F = PrimeField<PrimeFieldTy>;
  FieldElementSize() : is_truncated_(false), num_bits_(0) {}
  explicit FieldElementSize(size_t num_bits)
      : is_truncated_(true), num_bits_(num_bits) {}

  size_t GetNumBits() const {
    if (is_truncated_ && num_bits_ > PrimeFieldTy::kModulusBits) {
      throw std::runtime_error(
          "num_bits is greater than the capacity of the field.");
    }
    return PrimeFieldTy::kModulusBits - 1;
  }

  // Calculate the sum of field element sizes in `elements`.
  static size_t Sum(const std::vector<FieldElementSize<F>>& elements) {
    return ByteSerializable<std::vector<FieldElementSize<F>>>::GetSize(
        elements);
  }

 private:
  bool is_truncated_;
  size_t num_bits_;
};

// The interface for a cryptographic sponge.
// A sponge can `absorb` inputs and later `squeeze` bytes or field elements.
template <typename SpongeConfig, typename PrimeFieldTy>
class CryptographicSponge<SpongeConfig, PrimeField<PrimeFieldTy>> {
 public:
  using Config = SpongeConfig;
  using F = PrimeField<PrimeFieldTy>;

  virtual ~CryptographicSponge() = default;

  virtual CryptographicSponge* clone() const = 0;

  // Absorb an input into the sponge.
  template <typename T>
  virtual void Absorb(T& input) = 0;

  // Squeeze `num_bytes` bytes from the sponge.
  virtual std::vector<uint8_t> SqueezeBytes(size_t num_bytes) = 0;

  // Squeeze `num_bits` bits from the sponge.
  virtual std::vector<bool> SqueezeBits(size_t num_bits) = 0;

  // Squeeze `sizes.size()` field elements from the sponge
  // where the `i`-th element has `sizes[i]` bits.
  std::vector<F> SqueezeFieldElementsWithSizes(
      std::vector<FieldElementSize<F>> sizes) {
    return SqueezeFieldElementsWithSizesDefaultImpl<SpongeConfig, PrimeFieldTy>(
        *this, sizes);
  };

  // Squeeze `num_elements` nonnative field elements from the sponge.
  std::vector<F> SqueezeFieldElements(size_t num_elements) {
    std::vector<FieldElementSize<F>> sizes(num_elements, FieldElementSize<F>());
    return SqueezeFieldElementsWithSizes(sizes)
  };

  // Creates a new sponge with applied domain separation.
  virtual CryptographicSponge* fork(const std::vector<std::byte>& domain) {
    CryptographicSponge* new_sponge = this->clone();

    std::vector<std::byte> input =
        Absorb<std::byte>::to_sponge_bytes_as_vec(domain.size());
    input.insert(input.end(), domain.begin(), domain.end());
    new_sponge->absorb(input);

    return new_sponge;
  }
};

// The interface for field-based cryptographic sponge.
// `CF` is the native field used by the cryptographic sponge implementation.
template <typename SpongeConfig, typename PrimeFieldTy>
class FieldBasedCryptographicSponge<SpongeConfig, PrimeField<PrimeFieldTy>>
    : public CryptographicSponge<SpongeConfig, PrimeField<PrimeFieldTy>> {
  using Config = SpongeConfig;
  using CF = PrimeField<PrimeFieldTy>;

  // Squeeze `num_elements` field elements from the sponge.
  virtual std::vector<CF> SqueezeNativeFieldElements(size_t num_elements) = 0;

  // Squeeze `sizes.len()` field elements from the sponge,
  // where the `i`-th element of the output has `sizes[i]`.
  std::vector<CF> SqueezeNativeFieldElementsWithSize(
      std::vector<FieldElementSize<CF>> field_element_sizes) {
    bool all_full_size = true;
    for (auto size : field_element_sizes) {
      if (size.GetNumBits() != PrimeFieldTy::kModulusBits) {
        all_full_size = false;
        break;
      }
    }

    if (all_full_size) {
      this.SqueezeNativeFieldElements(field_element_sizes.size());
    } else {
      SqueezeFieldElementsWithSizesDefaultImpl<SpongeConfig, PrimeFieldTy>(
          *this, field_element_sizes);
    }
  }
};

// An extension for the interface of a cryptographic sponge.
// In addition to operations defined in `CryptographicSponge`,
// `SpongeExt` can convert itself to a state, and instantiate from state.
template <typename SpongeConfig, typename PrimeFieldTy>
class SpongeExt : public CryptographicSponge<SpongeConfig, PrimeFieldTy> {
 public:
  using Config = SpongeConfig;
  using F = PrimeField<PrimeFieldTy>;

  class State {
   public:
    virtual ~State() = default;
    virtual std::unique_ptr<State> clone() const = 0;
  };

  // Instantiate the sponge from a state.
  virtual SpongeExt FromState(std::unique_ptr<State> state,
                              const Config& params) = 0;

  // Convert the sponge to a state.
  virtual std::vector<F> ToState() = 0;
};

enum class DuplexSpongeMode { Absorbing, Squeezing };

struct Absorbing {
  size_t next_absorb_index;
};

struct Squeezing {
  size_t next_squeeze_index;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_H_
