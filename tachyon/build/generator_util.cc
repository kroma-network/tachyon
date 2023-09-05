#include "tachyon/build/generator_util.h"

#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"

#include "tachyon/base/strings/string_util.h"

namespace tachyon::build {

base::FilePath BazelOutToHdrPath(const base::FilePath& out) {
  std::vector<std::string> components = out.GetComponents();
  base::FilePath header_path(absl::StrJoin(components.begin() + 3,
                                           components.end() - 1,
                                           base::FilePath::kSeparators));
  header_path =
      header_path.Append(out.BaseName().RemoveExtension().value() + ".h");
  return header_path;
}

std::string BazelOutToHdrGuardMacro(const base::FilePath& out) {
  std::vector<std::string> components = out.GetComponents();
  base::FilePath header_path(absl::StrJoin(components.begin() + 3,
                                           components.end() - 1,
                                           base::FilePath::kSeparators));
  // In case of .cu.h, it removes extension twice.
  base::FilePath basename = out.BaseName().RemoveExtension().RemoveExtension();
  return base::ToUpperASCII(absl::StrCat(
      absl::StrJoin(components.begin() + 3, components.end() - 1, "_"),
      absl::Substitute("_$0_H_", basename.value())));
}

}  // namespace tachyon::build
