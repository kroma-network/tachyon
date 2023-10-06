#ifndef TACHYON_BUILD_WRITER_H_
#define TACHYON_BUILD_WRITER_H_

#include <string>

#include "tachyon/base/files/file_path.h"

namespace tachyon::build {

struct Writer {
  int Write(const std::string& content) const;

  std::string generator;
  base::FilePath out;
};

}  // namespace tachyon::build

#endif  // TACHYON_BUILD_WRITER_H_
