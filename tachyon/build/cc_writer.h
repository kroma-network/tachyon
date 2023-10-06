#ifndef TACHYON_BUILD_CC_WRITER_H_
#define TACHYON_BUILD_CC_WRITER_H_

#include <string>

#include "tachyon/build/writer.h"

namespace tachyon::build {

struct CcWriter : public Writer {
  base::FilePath GetHdrPath() const;

  // NOTE: You should mark %{extern_c_front} after header inclusion when |c_api|
  // is true.
  int WriteHdr(const std::string& content, bool c_api) const;
  int WriteSrc(const std::string& content) const;
};

}  // namespace tachyon::build

#endif  // TACHYON_BUILD_CC_WRITER_H_
