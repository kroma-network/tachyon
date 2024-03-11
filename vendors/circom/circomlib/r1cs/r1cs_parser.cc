#include "circomlib/r1cs/r1cs_parser.h"

#include <string.h>

#include <vector>

#include "tachyon/base/buffer/read_only_buffer.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/logging.h"

namespace tachyon::circom {

std::unique_ptr<R1CS> R1CSParser::Parse(const base::FilePath& path) const {
  std::optional<std::vector<uint8_t>> r1cs_data = base::ReadFileToBytes(path);
  if (!r1cs_data.has_value()) {
    LOG(ERROR) << "Failed to read file: " << path.value();
    return nullptr;
  }

  base::ReadOnlyBuffer buffer(r1cs_data->data(), r1cs_data->size());
  buffer.set_endian(base::Endian::kLittle);
  char magic[4];
  uint32_t version;
  if (!buffer.ReadMany(magic, &version)) return nullptr;
  if (memcmp(magic, kR1CSMagic, 4) != 0) {
    LOG(ERROR) << "Invalid magic: " << magic;
    return nullptr;
  }
  std::unique_ptr<R1CS> r1cs;
  if (version == 1) {
    r1cs.reset(new v1::R1CS());
    CHECK(r1cs->ToV1()->Read(buffer));
  } else {
    LOG(ERROR) << "Invalid version: " << version;
    return nullptr;
  }
  return r1cs;
}

}  // namespace tachyon::circom
