// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_grain_lfsr.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bls12/bls12_381/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

namespace {

class PoseidonGrainLFSRTest
    : public math::FiniteFieldTest<math::bls12_381::Fr> {
 public:
  void SetUp() override {
    default_config_.is_sbox_an_inverse = false;
    default_config_.prime_num_bits = 255;
    default_config_.state_len = 3;
    default_config_.num_full_rounds = 8;
    default_config_.num_partial_rounds = 31;
  }

 protected:
  PoseidonGrainLFSRConfig default_config_;
};

}  // namespace

TEST_F(PoseidonGrainLFSRTest, GetBits) {
  PoseidonGrainLFSR<math::bls12_381::Fr> lfsr(default_config_);

  std::bitset<255> bits = lfsr.GetBits(255);
  ASSERT_EQ(bits.size(), 255);
  EXPECT_EQ(
      bits,
      std::bitset<255>(
          "00001100110011010111010101110101111010101001010111110100001011100000"
          "11000011001111011111101000111100111100100000111111100000101101011001"
          "11001101001101111001110110111001110000110001000110011111010111000101"
          "110111011110110010011001010101011011110011111101110"));
}

TEST_F(PoseidonGrainLFSRTest, GetFieldElementsModP) {
  PoseidonGrainLFSR<math::bls12_381::Fr> lfsr(default_config_);

  // clang-format off
  EXPECT_EQ(lfsr.GetFieldElementsModP(1)[0],
            *math::bls12_381::Fr::FromDecString(
                "27117311055620256798560880810000042840428971800021819916023577129547249660720"));
  EXPECT_EQ(lfsr.GetFieldElementsModP(1)[0],
            *math::bls12_381::Fr::FromDecString(
                "51641662388546346858987925410984003801092143452466182801674685248597955169158"));
  // clang-format on
}

TEST_F(PoseidonGrainLFSRTest, GetFieldElementsRejectionSampling) {
  PoseidonGrainLFSR<math::bls12_381::Fr> lfsr(default_config_);

  // clang-format off
  EXPECT_EQ(lfsr.GetFieldElementsRejectionSampling(1)[0],
            *math::bls12_381::Fr::FromDecString(
                "27117311055620256798560880810000042840428971800021819916023577129547249660720"));
  EXPECT_EQ(lfsr.GetFieldElementsRejectionSampling(1)[0],
            *math::bls12_381::Fr::FromDecString(
                "51641662388546346858987925410984003801092143452466182801674685248597955169158"));
  // clang-format on
}

TEST_F(PoseidonGrainLFSRTest, GrainLFSRConsistency) {
  PoseidonGrainLFSR<math::bls12_381::Fr> lfsr(default_config_);

  // clang-format off
  EXPECT_EQ(lfsr.GetFieldElementsRejectionSampling(1)[0],
            *math::bls12_381::Fr::FromDecString(
                "27117311055620256798560880810000042840428971800021819916023577129547249660720"));
  EXPECT_EQ(lfsr.GetFieldElementsRejectionSampling(1)[0],
            *math::bls12_381::Fr::FromDecString(
                "51641662388546346858987925410984003801092143452466182801674685248597955169158"));
  EXPECT_EQ(lfsr.GetFieldElementsModP(1)[0],
            *math::bls12_381::Fr::FromDecString(
                "30468495022634911716522728179277518871747767531215914044579216845399211650580"));
  EXPECT_EQ(lfsr.GetFieldElementsModP(1)[0],
            *math::bls12_381::Fr::FromDecString(
                "17250718238509906485015112994867732544602358855445377986727968022920517907825"));
  // clang-format on
}

}  // namespace tachyon::crypto
