#ifndef TACHYON_BUILD_CC_WRITER_H_
#define TACHYON_BUILD_CC_WRITER_H_

#include "tachyon/build/writer.h"

namespace tachyon::build {

struct CcWriter : public Writer {
  base::FilePath GetHdrPath() const;

  int WriteHdr(const std::string& content) const;
  int WriteSrc(const std::string& content) const;
};

}  // namespace tachyon::build

#endif  // TACHYON_BUILD_CC_WRITER_H_
