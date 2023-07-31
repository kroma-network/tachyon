#ifndef TACHYON_BASE_FILES_FILE_PATH_FLAG_H_
#define TACHYON_BASE_FILES_FILE_PATH_FLAG_H_

#include "tachyon/base/files/file_path.h"
#include "tachyon/base/flag/flag.h"

namespace tachyon::base {

template <>
class FlagValueTraits<FilePath> {
 public:
  static bool ParseValue(std::string_view input, FilePath* value,
                         std::string* reason) {
    *value = FilePath(input);
    return true;
  }
};

typedef Flag<FilePath> FilePathFlag;

}  // namespace tachyon::base

#endif  // TACHYON_BASE_FILES_FILE_PATH_FLAG_H_
