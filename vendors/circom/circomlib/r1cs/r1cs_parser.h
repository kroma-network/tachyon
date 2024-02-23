#ifndef VENDORS_CIRCOM_CIRCOMLIB_R1CS_R1CS_PARSER_H_
#define VENDORS_CIRCOM_CIRCOMLIB_R1CS_R1CS_PARSER_H_

#include <memory>

#include "circomlib/r1cs/r1cs.h"
#include "tachyon/base/files/file_path.h"

namespace tachyon::circom {

class R1CSParser {
 public:
  // Return nullptr if the parser failed to parse.
  std::unique_ptr<R1CS> Parse(const base::FilePath& path) const;
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_R1CS_R1CS_PARSER_H_
