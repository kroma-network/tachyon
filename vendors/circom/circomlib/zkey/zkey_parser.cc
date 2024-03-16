#include "circomlib/zkey/zkey_parser.h"

#include <string.h>

#include <vector>

#include "tachyon/base/buffer/read_only_buffer.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/logging.h"

namespace tachyon::circom {

std::unique_ptr<ZKey> ZKeyParser::Parse(const base::FilePath& path) const {
  std::optional<std::vector<uint8_t>> zkey_data = base::ReadFileToBytes(path);
  if (!zkey_data.has_value()) {
    LOG(ERROR) << "Failed to read file: " << path.value();
    return nullptr;
  }

  base::ReadOnlyBuffer buffer(zkey_data->data(), zkey_data->size());
  buffer.set_endian(base::Endian::kLittle);
  char magic[4];
  uint32_t version;
  if (!buffer.ReadMany(magic, &version)) return nullptr;
  if (memcmp(magic, kZkeyMagic, 4) != 0) {
    LOG(ERROR) << "Invalid magic: " << magic;
    return nullptr;
  }
  std::unique_ptr<ZKey> zkey;
  if (version == 1) {
    zkey.reset(new v1::ZKey());
    CHECK(zkey->ToV1()->Read(buffer));
  } else {
    LOG(ERROR) << "Invalid version: " << version;
    return nullptr;
  }
  return zkey;
}

}  // namespace tachyon::circom
