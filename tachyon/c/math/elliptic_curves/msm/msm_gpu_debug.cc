#if TACHYON_CUDA
#include <iostream>

#include "absl/strings/str_split.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/time/time.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_prime_field_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/msm/msm_gpu.h"
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
  base::FlagParser parser;
  std::vector<int> idxes;
  int degree;
  parser.AddFlag<base::Flag<std::vector<int>>>(&idxes)
      .set_long_name("--idx")
      .set_required();
  parser.AddFlag<base::IntFlag>(&degree)
      .set_long_name("--degree")
      .set_required();
  {
    std::string error;
    if (!parser.Parse(argc, argv, &error)) {
      tachyon_cerr << error << std::endl;
      return 1;
    }
  }

  tachyon_init_msm_gpu(degree);
  bn254::G1JacobianPoint::Curve::Init();

  std::string_view save_location_str;
  CHECK(base::Environment::Get("TACHYON_SAVE_LOCATION", &save_location_str));
  for (int idx : idxes) {
    base::FilePath bases_txt(
        absl::Substitute("$0/bases$1.txt", save_location_str, idx));
    base::FilePath scalars_txt(
        absl::Substitute("$0/scalars$1.txt", save_location_str, idx));
    auto bases = ReadAffinePoints(bases_txt);
    auto scalars = ReadScalarFields(scalars_txt);
    CHECK_EQ(bases.size(), scalars.size());

    base::TimeTicks now = base::TimeTicks::Now();
    std::unique_ptr<tachyon_bn254_g1_jacobian> ret(
        tachyon_bn254_g1_affine_msm_gpu(
            reinterpret_cast<const tachyon_bn254_g1_affine*>(bases.data()),
            bases.size(),
            reinterpret_cast<const tachyon_bn254_fr*>(scalars.data()),
            scalars.size()));
    std::cout << (base::TimeTicks::Now() - now) << std::endl;
    std::cout << cc::math::ToJacobianPoint(*ret).ToAffine().ToHexString()
              << std::endl;
  }

  tachyon_release_msm_gpu();

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
