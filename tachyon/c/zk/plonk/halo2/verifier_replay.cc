#include <string>

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/logging.h"
#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_pcs.h"
#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_verifier.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key_impl.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/halo2/transcript_type.h"
#include "tachyon/zk/plonk/halo2/verifier.h"

constexpr uint8_t kProof[] = {
    // You need to fill here!
    0,
};

namespace tachyon {
namespace c::zk::plonk::halo2::bn254 {

using ProvingKey = plonk::bn254::ProvingKeyImpl;

using Verifier = tachyon::zk::plonk::halo2::Verifier<PCS>;

using VerifyingKey =
    tachyon::zk::plonk::VerifyingKey<PCS::Field, PCS::Commitment>;

std::vector<std::vector<PCS::Evals>> GetInstanceColumnsVec() {
  // You need to fill here!
  return {{{}}};
}

bool VerifyProof(tachyon_halo2_bn254_shplonk_verifier* c_verifier,
                 const std::vector<uint8_t>& pk_bytes) {
  Verifier* verifier = reinterpret_cast<Verifier*>(c_verifier);
  std::cout << "deserializing proving key" << std::endl;
  ProvingKey pk(absl::MakeConstSpan(pk_bytes), /*read_only_vk=*/true);
  std::cout << "done deserializing proving key" << std::endl;

  uint32_t extended_k = pk.verifying_key().constraint_system().ComputeExtendedK(
      verifier->pcs().K());
  verifier->set_extended_domain(
      PCS::ExtendedDomain::Create(size_t{1} << extended_k));
  pk.SetTranscriptRepr(*verifier);

  return verifier->VerifyProof(pk.verifying_key(), GetInstanceColumnsVec());
}

}  // namespace c::zk::plonk::halo2::bn254

int RunMain(int argc, char** argv) {
  if (base::Environment::Has("TACHYON_PCS_PARAMS_PATH")) {
    tachyon_cerr << "If this is set, the pcs params is overwritten"
                 << std::endl;
    return 1;
  }
  if (base::Environment::Has("TACHYON_PK_LOG_PATH")) {
    tachyon_cerr << "If this is set, the pk log is overwritten" << std::endl;
    return 1;
  }

  zk::plonk::halo2::TranscriptType transcript_type;
  uint32_t k;
  std::string s_hex;
  base::FilePath pcs_params_path;
  base::FilePath pk_path;
  base::FlagParser parser;
  parser.AddFlag<base::Flag<zk::plonk::halo2::TranscriptType>>(&transcript_type)
      .set_long_name("--transcript_type")
      .set_required()
      .set_help("Transcript type");
  parser.AddFlag<base::Uint32Flag>(&k)
      .set_short_name("-k")
      .set_required()
      .set_help("K");
  parser.AddFlag<base::FilePathFlag>(&pcs_params_path)
      .set_long_name("--pcs_params")
      .set_required()
      .set_help("The path to pcs params");
  parser.AddFlag<base::FilePathFlag>(&pk_path)
      .set_long_name("--pk")
      .set_required()
      .set_help("The path to proving key");
  {
    std::string error;
    if (!parser.Parse(argc, argv, &error)) {
      tachyon_cerr << error << std::endl;
      return 1;
    }
  }

  std ::optional<std::vector<uint8_t>> pk_bytes =
      base::ReadFileToBytes(pk_path);
  if (!pk_bytes.has_value()) {
    tachyon_cerr << "Failed to read file: " << pk_path.value() << std::endl;
    return 1;
  }

  std::optional<std::vector<uint8_t>> pcs_params_bytes =
      base::ReadFileToBytes(pcs_params_path);
  if (!pcs_params_bytes.has_value()) {
    tachyon_cerr << "Failed to read file: " << pcs_params_path.value()
                 << std::endl;
    return 1;
  }

  std::cout << "creating verifier" << std::endl;
  tachyon_halo2_bn254_shplonk_verifier* verifier =
      tachyon_halo2_bn254_shplonk_verifier_create_from_params(
          static_cast<uint8_t>(transcript_type), k, pcs_params_bytes->data(),
          pcs_params_bytes->size(), std::data(kProof), std::size(kProof));
  std::cout << "done creating verifier" << std::endl;

  if (!c::zk::plonk::halo2::bn254::VerifyProof(verifier, pk_bytes.value())) {
    tachyon_halo2_bn254_shplonk_verifier_destroy(verifier);
    tachyon_cerr << "Failed to verify proof" << std::endl;
    return 1;
  }

  tachyon_halo2_bn254_shplonk_verifier_destroy(verifier);
  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RunMain(argc, argv); }
