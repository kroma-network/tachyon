// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#include "tachyon/crypto/hashes/sponge/duplex_sponge_mode.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"

namespace tachyon::crypto {

TEST(DuplexSpongeModeTest, Copyable) {
  DuplexSpongeMode expected = DuplexSpongeMode::Squeezing(3);

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  DuplexSpongeMode value;
  ASSERT_TRUE(write_buf.Read(&value));

  EXPECT_EQ(value, expected);
}

}  // namespace tachyon::crypto
