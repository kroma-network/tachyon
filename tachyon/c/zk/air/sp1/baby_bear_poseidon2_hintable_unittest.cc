#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_hintable.h"

#include <utility>
#include <vector>

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_two_adic_fri_type_traits.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon {

using F = c::zk::air::sp1::baby_bear::F;
using Params = c::zk::air::sp1::baby_bear::Params;
using Poseidon2 = c::zk::air::sp1::baby_bear::Poseidon2;
using Block = c::zk::air::sp1::Block<F>;

constexpr size_t kRate = 4;

class HintableTest : public math::FiniteFieldTest<F> {};

TEST_F(HintableTest, DuplexChallenger) {
  Poseidon2 sponge;
  crypto::DuplexChallenger<Poseidon2, kRate> challenger(std::move(sponge));
  for (size_t i = 0; i < Params::kWidth; ++i) {
    challenger.state_[i] = F(i);
  }
  for (size_t i = 0; i < kRate / 2; ++i) {
    challenger.input_buffer_.push_back(F(i));
  }
  for (size_t i = 0; i < Params::kWidth / 2; ++i) {
    challenger.output_buffer_.push_back(F(i));
  }

  std::vector<std::vector<Block>> actual =
      c::zk::air::sp1::baby_bear::WriteHint(challenger);

  std::vector<std::vector<Block>> expected = {
      {base::CreateVector(Params::kWidth,
                          [](size_t i) { return Block::From(F(i)); })},
      {Block::From(F(kRate / 2))},
      {base::CreateVector(Params::kWidth,
                          [](size_t i) {
                            if (i < kRate / 2) {
                              return Block::From(F(i));
                            } else {
                              return Block();
                            }
                          })},
      {Block::From(F(Params::kWidth / 2))},
      {base::CreateVector(Params::kWidth,
                          [](size_t i) {
                            if (i < Params::kWidth / 2) {
                              return Block::From(F(i));
                            } else {
                              return Block();
                            }
                          })},
  };

  EXPECT_EQ(actual.size(),
            c::zk::air::sp1::baby_bear::EstimateSize(challenger));
  EXPECT_EQ(actual, expected);
}

}  // namespace tachyon
