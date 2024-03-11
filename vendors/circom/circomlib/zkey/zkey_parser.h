#ifndef VENDORS_CIRCOM_CIRCOMLIB_ZKEY_ZKEY_PARSER_H_
#define VENDORS_CIRCOM_CIRCOMLIB_ZKEY_ZKEY_PARSER_H_

#include <memory>

#include "circomlib/zkey/zkey.h"
#include "tachyon/base/files/file_path.h"

namespace tachyon::circom {

class ZKeyParser {
 public:
  // Return nullptr if the parser failed to parse.
  std::unique_ptr<ZKey> Parse(const base::FilePath& path) const;
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_ZKEY_ZKEY_PARSER_H_
