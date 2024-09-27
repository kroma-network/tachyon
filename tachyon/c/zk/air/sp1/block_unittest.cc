#include "tachyon/c/zk/air/sp1/block.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon {

namespace {

using F = math::BabyBear;

class BlockTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST(BlockTest, Copyable) {
  c::zk::air::sp1::Block<F> expected(
      base::CreateArray<4>([]() { return F::Random(); }));

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  c::zk::air::sp1::Block<F> value;
  ASSERT_TRUE(write_buf.Read(&value));

  EXPECT_EQ(value, expected);
}

}  // namespace tachyon
