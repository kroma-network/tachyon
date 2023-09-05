#include "tachyon/build/writer.h"

#include "tachyon/base/files/file_util.h"

namespace tachyon::build {

int Writer::Write(const std::string& content) const {
  if (!base::WriteFile(out, content)) return 1;
  return 0;
}

}  // namespace tachyon::build
