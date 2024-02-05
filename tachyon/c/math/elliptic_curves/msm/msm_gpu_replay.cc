#if TACHYON_CUDA
#include <iostream>

#include "absl/strings/str_split.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/time/time.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_prime_field_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/msm_gpu.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon {

using namespace math;

template <typename R>
R ReadField(std::string_view* txt) {
  *txt = absl::StripLeadingAsciiWhitespace(*txt);
  size_t idx = txt->find(',');
  std::string_view elm = txt->substr(0, idx);
  txt->remove_prefix(idx);
  auto bigint = BigInt<4>::FromDecString(std::string(elm));
  return R::FromMontgomery(bigint);
}

std::vector<bn254::G1AffinePoint> ReadAffinePoints(const base::FilePath& path) {
  std::string bases;
  CHECK(base::ReadFileToString(path, &bases));
  std::vector<std::string> lines = absl::StrSplit(bases, "\n");
  lines.pop_back();
  return base::Map(lines, [](const std::string& line) {
    std::string_view txt = line;
    CHECK(base::ConsumePrefix(&txt, "("));
    CHECK(base::ConsumeSuffix(&txt, ")"));
    auto x = ReadField<bn254::Fq>(&txt);
    CHECK(base::ConsumePrefix(&txt, ", "));
    auto y = ReadField<bn254::Fq>(&txt);
    return bn254::G1AffinePoint(x, y, x.IsZero() && y.IsZero());
  });
}

std::vector<bn254::Fr> ReadScalarFields(const base::FilePath& path) {
  std::string scalars;
  CHECK(base::ReadFileToString(path, &scalars));
  std::vector<std::string> lines = absl::StrSplit(scalars, "\n");
  lines.pop_back();
  return base::Map(lines, [](const std::string& line) {
    std::string_view txt = line;
    return ReadField<bn254::Fr>(&txt);
  });
}

int RealMain(int argc, char** argv) {
  if (base::Environment::Has("TACHYON_SAVE_LOCATION")) {
    tachyon_cerr << "If this is set, the log is overwritten" << std::endl;
    return 1;
  }

  base::FlagParser parser;
  std::vector<int> idxes;
  int degree;
  base::FilePath location;
  int algorithm = 0;
  parser.AddFlag<base::Flag<std::vector<int>>>(&idxes)
      .set_long_name("--idx")
      .set_required();
  parser.AddFlag<base::IntFlag>(&degree)
      .set_long_name("--degree")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&location)
      .set_long_name("--location")
      .set_required();
  parser
      .AddFlag<base::IntFlag>(
          [&algorithm](std::string_view arg, std::string* reason) {
            if (arg == "bellman_msm") {
              algorithm = 0;
              return true;
            } else if (arg == "cuzk") {
              algorithm = 1;
              return true;
            }
            *reason = absl::Substitute("Not supported algorithm: $0", arg);
            return false;
          })
      .set_long_name("--algo")
      .set_help(
          "Algorithms to be benchmarked with. (supported algorithms: "
          "bellman_msm, cuzk)");
  {
    std::string error;
    if (!parser.Parse(argc, argv, &error)) {
      tachyon_cerr << error << std::endl;
      return 1;
    }
  }

  tachyon_bn254_g1_init();
  tachyon_bn254_g1_msm_gpu_ptr msm =
      tachyon_bn254_g1_create_msm_gpu(degree, algorithm);

  for (int idx : idxes) {
    base::FilePath bases_txt(absl::Substitute("$0/bases$1.txt", location, idx));
    base::FilePath scalars_txt(
        absl::Substitute("$0/scalars$1.txt", location, idx));
    auto bases = ReadAffinePoints(bases_txt);
    auto scalars = ReadScalarFields(scalars_txt);
    CHECK_EQ(bases.size(), scalars.size());

    base::TimeTicks now = base::TimeTicks::Now();
    std::unique_ptr<tachyon_bn254_g1_jacobian> ret(
        tachyon_bn254_g1_affine_msm_gpu(
            msm, reinterpret_cast<const tachyon_bn254_g1_affine*>(bases.data()),
            reinterpret_cast<const tachyon_bn254_fr*>(scalars.data()),
            scalars.size()));
    std::cout << (base::TimeTicks::Now() - now) << std::endl;
    std::cout << cc::math::ToJacobianPoint(*ret).ToAffine().ToHexString()
              << std::endl;
  }

  tachyon_bn254_g1_destroy_msm_gpu(msm);

  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
#else
#include "tachyon/base/console/iostream.h"

int main(int argc, char **argv) {
  tachyon_cerr << "please build with --config cuda" << std::endl;
  return 1;
}
#endif  // TACHYON_CUDA
