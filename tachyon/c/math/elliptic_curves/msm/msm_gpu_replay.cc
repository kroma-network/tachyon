#if TACHYON_CUDA
#include <iostream>

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/time/time.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/msm_gpu.h"

namespace tachyon {

using namespace math;

std::vector<bn254::G1AffinePoint> ReadAffinePoints(const base::FilePath& path) {
  std::string bases;
  CHECK(base::ReadFileToString(path, &bases));
  base::Buffer buffer(reinterpret_cast<char*>(bases.data()), bases.size());
  std::vector<bn254::G1AffinePoint> ret;
  CHECK(buffer.Read(&ret));
  CHECK(buffer.Done());
  return ret;
}

std::vector<bn254::Fr> ReadScalarFields(const base::FilePath& path) {
  std::string scalars;
  CHECK(base::ReadFileToString(path, &scalars));
  base::Buffer buffer(reinterpret_cast<char*>(scalars.data()), scalars.size());
  std::vector<bn254::Fr> ret;
  CHECK(buffer.Read(&ret));
  CHECK(buffer.Done());
  return ret;
}

int RealMain(int argc, char** argv) {
  if (base::Environment::Has("TACHYON_MSM_GPU_INPUT_DIR")) {
    tachyon_cerr << "If this is set, the log is overwritten" << std::endl;
    return 1;
  }

  base::FlagParser parser;
  std::vector<int> idxes;
  int degree;
  base::FilePath input_dir;
  parser.AddFlag<base::Flag<std::vector<int>>>(&idxes)
      .set_long_name("--idx")
      .set_required();
  parser.AddFlag<base::IntFlag>(&degree)
      .set_long_name("--degree")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&input_dir)
      .set_long_name("--input_dir")
      .set_required();
  {
    std::string error;
    if (!parser.Parse(argc, argv, &error)) {
      tachyon_cerr << error << std::endl;
      return 1;
    }
  }

  tachyon_bn254_g1_init();
  tachyon_bn254_g1_msm_gpu_ptr msm = tachyon_bn254_g1_create_msm_gpu(degree);

  for (int idx : idxes) {
    base::FilePath bases_txt(
        absl::Substitute("$0/bases$1.txt", input_dir.value(), idx));
    base::FilePath scalars_txt(
        absl::Substitute("$0/scalars$1.txt", input_dir.value(), idx));
    auto bases = ReadAffinePoints(bases_txt);
    auto scalars = ReadScalarFields(scalars_txt);
    CHECK_EQ(bases.size(), scalars.size());

    base::TimeTicks now = base::TimeTicks::Now();
    std::unique_ptr<tachyon_bn254_g1_jacobian> ret(
        tachyon_bn254_g1_affine_msm_gpu(msm, c::base::c_cast(bases.data()),
                                        c::base::c_cast(scalars.data()),
                                        scalars.size()));
    std::cout << (base::TimeTicks::Now() - now) << std::endl;
    std::cout << c::base::native_cast(*ret).ToAffine().ToHexString()
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
