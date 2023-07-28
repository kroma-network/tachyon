#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

namespace tachyon::math {

// From:
// https://github.com/arkworks-rs/gemini/blob/main/src/kzg/msm/variable_base.rs#L20
std::vector<int64_t> MakeDigits(const mpz_class& scalar, size_t w,
                                size_t num_bits) {
  static_assert(GMP_LIMB_BITS == 64, "This code assumes limb bits is 64 bit");
  uint64_t radix = 1 << w;
  uint64_t window_mask = radix - 1;

  uint64_t carry = 0;
  if (num_bits == 0) {
    num_bits = gmp::GetNumBits(scalar);
  }
  size_t digits_count = (num_bits + w - 1) / w;
  std::vector<int64_t> digits =
      base::CreateVector(digits_count, static_cast<int64_t>(0));
  for (size_t i = 0; i < digits.size(); ++i) {
    // Construct a buffer of bits of the scalar, starting at `bit_offset`.
    size_t bit_offset = i * w;
    size_t u64_idx = bit_offset / 64;
    size_t bit_idx = bit_offset % 64;

    size_t limb_size = gmp::GetLimbSize(scalar);
    // Read the bits from the scalar
    uint64_t bit_buf;
    if (limb_size == 0) {
      bit_buf = 0;
    } else if (bit_idx < 64 - w || u64_idx == limb_size - 1) {
      // This window's bits are contained in a single u64,
      // or it's the last u64 anyway.
      bit_buf = gmp::GetLimbConstRef(scalar, u64_idx) >> bit_idx;
    } else {
      // Combine the current u64's bits with the bits from the next u64
      bit_buf = (gmp::GetLimbConstRef(scalar, u64_idx) >> bit_idx) |
                (gmp::GetLimbConstRef(scalar, 1 + u64_idx) << (64 - bit_idx));
    }

    // Read the actual coefficient value from the window
    uint64_t coeff = carry + (bit_buf & window_mask);  // coeff = [0, 2^r)

    // Recenter coefficients from [0,2^w) to [-2^w/2, 2^w/2)
    carry = (coeff + radix / 2) >> w;
    digits[i] = static_cast<int64_t>(coeff) - static_cast<int64_t>(carry << w);
  }

  digits[digits_count - 1] += static_cast<int64_t>(carry << w);

  return digits;
}

}  // namespace tachyon::math
