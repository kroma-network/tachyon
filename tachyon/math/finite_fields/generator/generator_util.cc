#include "tachyon/math/finite_fields/generator/generator_util.h"

#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"

namespace tachyon::math {

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

// TODO(chokobole): Consider bigendian.
std::string MpzClassToString(const mpz_class& m) {
  size_t limb_size = math::gmp::GetLimbSize(m);
  if (limb_size == 0) {
    return "UINT64_C(0)";
  }

  std::vector<std::string> ret;
  ret.reserve(limb_size);
  for (size_t i = 0; i < limb_size; ++i) {
    ret.push_back(
        absl::Substitute("UINT64_C($0)", math::gmp::GetLimbConstRef(m, i)));
  }
  return absl::StrJoin(ret, ", ");
}

std::string MpzClassToMontString(const mpz_class& v, const mpz_class& m) {
  size_t limb_size = math::gmp::GetLimbSize(m);
  switch (limb_size) {
    case 1:
      return MpzClassToMontString<1>(v, m);
    case 2:
      return MpzClassToMontString<2>(v, m);
    case 3:
      return MpzClassToMontString<3>(v, m);
    case 4:
      return MpzClassToMontString<4>(v, m);
    case 5:
      return MpzClassToMontString<5>(v, m);
    case 6:
      return MpzClassToMontString<6>(v, m);
    case 7:
      return MpzClassToMontString<7>(v, m);
    case 8:
      return MpzClassToMontString<8>(v, m);
    case 9:
      return MpzClassToMontString<9>(v, m);
  }
  NOTREACHED();
  return "";
}

}  // namespace tachyon::math
