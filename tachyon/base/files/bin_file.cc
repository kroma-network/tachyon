#include "tachyon/base/files/bin_file.h"

#include <optional>
#include <utility>

#include "tachyon/base/files/file_util.h"
#include "tachyon/base/logging.h"

namespace tachyon::base {

bool BinFile::Load(const base::FilePath& path, bool use_mmap) {
  if (use_mmap) {
    base::File file(path, base::File::FLAG_OPEN | base::File::FLAG_READ);
    std::unique_ptr<base::MemoryMappedFile> map(new base::MemoryMappedFile);
    if (!map->Initialize(std::move(file),
                          base::MemoryMappedFile::Region::kWholeFile,
                          base::MemoryMappedFile::Access::READ_WRITE_COPY))
      return false;
    map_ = std::move(map);
    return true;
  } else {
    std::optional<std::vector<uint8_t>> vec = base::ReadFileToBytes(path);
    if (!vec.has_value()) {
      LOG(ERROR) << "Failed to read file: " << path.value();
      return false;
    }
    vec_ = std::move(vec).value();
    return true;
  }
}

}  // namespace tachyon::base
