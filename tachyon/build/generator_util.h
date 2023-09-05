#ifndef TACHYON_BUILD_GENERATOR_UTIL_H_
#define TACHYON_BUILD_GENERATOR_UTIL_H_

#include "tachyon/base/files/file_path.h"

namespace tachyon::build {

base::FilePath BazelOutToHdrPath(const base::FilePath& out);

std::string BazelOutToHdrGuardMacro(const base::FilePath& out);

}  // namespace tachyon::build

#endif  // TACHYON_BUILD_GENERATOR_UTIL_H_
