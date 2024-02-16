#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"
#include "tachyon/zk/plonk/halo2/poseidon_transcript.h"
#include "tachyon/zk/plonk/halo2/sha256_transcript.h"

namespace tachyon::zk::plonk::halo2 {

template <typename Transcript>
class TranscriptWriterTest : public math::FiniteFieldTest<math::bn254::Fr> {
 public:
  void TearDown() override {
    tachyon_halo2_bn254_transcript_writer_destroy(writer_);
    tachyon_halo2_bn254_transcript_writer_destroy(writer_clone_);
  }

 protected:
  tachyon_halo2_bn254_transcript_writer* writer_ = nullptr;
  tachyon_halo2_bn254_transcript_writer* writer_clone_ = nullptr;
};

using TranscriptTypes =
    testing::Types<Blake2bWriter<math::bn254::G1AffinePoint>,
                   PoseidonWriter<math::bn254::G1AffinePoint>,
                   Sha256Writer<math::bn254::G1AffinePoint>>;
TYPED_TEST_SUITE(TranscriptWriterTest, TranscriptTypes);

TYPED_TEST(TranscriptWriterTest, APIs) {
  using TranscriptWriter = TypeParam;

  base::Uint8VectorBuffer buffer;
  TranscriptWriter cpp_writer(std::move(buffer));

  uint8_t type;
  if constexpr (std::is_same_v<TranscriptWriter,
                               Blake2bWriter<math::bn254::G1AffinePoint>>) {
    type = TACHYON_HALO2_BLAKE2B_TRANSCRIPT;
    // NOLINTNEXTLINE(readability/braces)
  } else if constexpr (std::is_same_v<
                           TranscriptWriter,
                           PoseidonWriter<math::bn254::G1AffinePoint>>) {
    type = TACHYON_HALO2_POSEIDON_TRANSCRIPT;
    // NOLINTNEXTLINE(readability/braces)
  } else if constexpr (std::is_same_v<
                           TranscriptWriter,
                           Sha256Writer<math::bn254::G1AffinePoint>>) {
    type = TACHYON_HALO2_SHA256_TRANSCRIPT;
  }

  this->writer_ = tachyon_halo2_bn254_transcript_writer_create(type);

  EXPECT_EQ(cpp_writer.SqueezeChallenge(),
            reinterpret_cast<TranscriptWriter*>(this->writer_->extra)
                ->SqueezeChallenge());

  size_t expected_state_len =
      reinterpret_cast<TranscriptWriter*>(this->writer_->extra)->GetStateLen();

  size_t state_len;
  tachyon_halo2_bn254_transcript_writer_get_state(this->writer_, nullptr,
                                                  &state_len);
  ASSERT_EQ(state_len, expected_state_len);

  std::vector<uint8_t> state(state_len);
  tachyon_halo2_bn254_transcript_writer_get_state(this->writer_, state.data(),
                                                  &state_len);
  ASSERT_EQ(state_len, expected_state_len);

  this->writer_clone_ = tachyon_halo2_bn254_transcript_writer_create_from_state(
      type, state.data(), state_len);

  EXPECT_EQ(reinterpret_cast<TranscriptWriter*>(this->writer_->extra)
                ->SqueezeChallenge(),
            reinterpret_cast<TranscriptWriter*>(this->writer_clone_->extra)
                ->SqueezeChallenge());
}

}  // namespace tachyon::zk::plonk::halo2
