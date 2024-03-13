#ifndef TACHYON_BUILD_CC_WRITER_H_
#define TACHYON_BUILD_CC_WRITER_H_

#include <string>
#include <vector>

#include "tachyon/build/writer.h"

namespace tachyon::build {

struct CcWriter : public Writer {
  base::FilePath GetHdrPath() const;

  // NOTE: You should mark %{extern_c_front} after header inclusion when |c_api|
  // is true.
  int WriteHdr(const std::string& content, bool c_api) const;
  int WriteSrc(const std::string& content) const;

  // Remove all lines equals to "%{if |tag|}" or "%{endif |tag|}"
  // if |select_tag_block| is true. Otherwise, remove all lines that are
  // enclosed between "%{if |tag|}" and "%{endif |tag|}".
  static void RemoveOptionalLines(std::vector<std::string>& tpl_lines,
                                  std::string_view tag, bool select_tag_block);

 private:
  // Remove each first line that equals to |start_line| and |end_line| if
  // |select_tag_block| is true. Otherwise, remove lines between the lines
  // (including boundary lines).
  static bool DoRemoveOptionalLines(std::vector<std::string>& tpl_lines,
                                    std::string_view start_line,
                                    std::string_view end_line,
                                    bool select_tag_block);
};

}  // namespace tachyon::build

#endif  // TACHYON_BUILD_CC_WRITER_H_
