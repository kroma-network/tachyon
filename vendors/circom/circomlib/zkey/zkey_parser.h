#ifndef VENDORS_CIRCOM_CIRCOMLIB_ZKEY_ZKEY_PARSER_H_
#define VENDORS_CIRCOM_CIRCOMLIB_ZKEY_ZKEY_PARSER_H_

#include <memory>

#include "circomlib/zkey/zkey.h"
#include "tachyon/base/files/file_path.h"

namespace tachyon::circom {

template <typename Curve>
class ZKeyParser {
 public:
  // Return nullptr if the parser failed to parse.
  std::unique_ptr<ZKey<Curve>> Parse(const base::FilePath& path) const;
};

}  // namespace tachyon::circom

// NOLINTNEXTLINE(build/include)
#include "circomlib/zkey/zkey_parser.cc"

#endif  // VENDORS_CIRCOM_CIRCOMLIB_ZKEY_ZKEY_PARSER_H_
