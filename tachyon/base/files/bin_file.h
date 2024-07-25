#ifndef TACHYON_BASE_FILES_BIN_FILE_H_
#define TACHYON_BASE_FILES_BIN_FILE_H_

#include <memory>
#include <vector>

#include "tachyon/base/buffer/read_only_buffer.h"
#include "tachyon/base/files/memory_mapped_file.h"
#include "tachyon/export.h"

namespace tachyon::base {

class TACHYON_EXPORT BinFile {
 public:
  BinFile() {}
  BinFile(const BinFile& other) = delete;
  BinFile& operator=(const BinFile& other) = delete;
  BinFile(BinFile&& other) = default;
  BinFile& operator=(BinFile&& other) = default;

  const std::vector<uint8_t>& vec() const { return vec_; }
  std::vector<uint8_t>& vec() { return vec_; }

  const MemoryMappedFile* map() const { return map_.get(); }
  MemoryMappedFile* map() { return map_.get(); }

  const uint8_t* GetData() const {
    if (map_) {
      return map_->data();
    } else {
      return vec_.data();
    }
  }

  uint8_t* GetData() {
    if (map_) {
      return map_->data();
    } else {
      return const_cast<uint8_t*>(vec_.data());
    }
  }

  size_t GetDataLength() const {
    if (map_) {
      return map_->length();
    } else {
      return vec_.size();
    }
  }

  ReadOnlyBuffer ToReadOnlyBuffer() const {
    return {GetData(), GetDataLength()};
  }

  bool Load(const FilePath& path, bool use_mmap);

 private:
  std::vector<uint8_t> vec_;
  std::unique_ptr<MemoryMappedFile> map_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_FILES_BIN_FILE_H_
