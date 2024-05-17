#ifndef VENDORS_CIRCOM_CIRCOMLIB_R1CS_R1CS_PARSER_H_
#define VENDORS_CIRCOM_CIRCOMLIB_R1CS_R1CS_PARSER_H_

#include <memory>

#include "circomlib/r1cs/r1cs.h"
#include "tachyon/base/files/file_path.h"

namespace tachyon::circom {

template <typename F>
class R1CSParser {
 public:
  // Return nullptr if the parser failed to parse.
  std::unique_ptr<R1CS<F>> Parse(const base::FilePath& path) const;
};

}  // namespace tachyon::circom

// NOLINTNEXTLINE(build/include)
#include "circomlib/r1cs/r1cs_parser.cc"

#endif  // VENDORS_CIRCOM_CIRCOMLIB_R1CS_R1CS_PARSER_H_
