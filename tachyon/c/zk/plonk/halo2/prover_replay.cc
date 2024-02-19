#include <string>

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/logging.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_prime_field_traits.h"
#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_prover.h"
#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_prover_impl.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key_impl.h"
#include "tachyon/cc/math/finite_fields/prime_field_conversions.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/halo2/constants.h"
#include "tachyon/zk/plonk/halo2/transcript_type.h"

namespace tachyon {
namespace c::zk::plonk::halo2::bn254 {

using ArgumentData =
    tachyon::zk::plonk::halo2::ArgumentData<PCS::Poly, PCS::Evals>;

using Prover = SHPlonkProverImpl;

using ProvingKey = plonk::bn254::ProvingKeyImpl;

ArgumentData DeserializeArgumentData(
    const std::vector<uint8_t>& arg_data_bytes) {
  ArgumentData arg_data;
  base::Buffer buffer(const_cast<uint8_t*>(arg_data_bytes.data()),
                      arg_data_bytes.size());
  CHECK(buffer.Read(&arg_data));
  CHECK(buffer.Done());
  return arg_data;
}

void WriteParams(tachyon_halo2_bn254_shplonk_prover* c_prover,
                 const base::FilePath& params_path) {
  Prover* prover = reinterpret_cast<Prover*>(c_prover);
  base::Uint8VectorBuffer buffer;
  CHECK(buffer.Grow(base::EstimateSize(prover->pcs())));
  CHECK(buffer.Write(prover->pcs()));
  CHECK(buffer.Done());
  CHECK(base::WriteLargeFile(params_path,
                             absl::MakeConstSpan(buffer.owned_buffer())));
}

void CreateProof(tachyon_halo2_bn254_shplonk_prover* c_prover,
                 const std::vector<uint8_t>& pk_bytes,
                 const std::vector<uint8_t>& arg_data_bytes,
                 const std::vector<uint8_t>& transcript_state_bytes) {
  Prover* prover = reinterpret_cast<Prover*>(c_prover);
  std::cout << "deserializing proving key" << std::endl;
  ProvingKey pk(absl::MakeConstSpan(pk_bytes), /*read_only_vk=*/false);
  std::cout << "done deserializing proving key" << std::endl;

  uint32_t extended_k = pk.verifying_key().constraint_system().ComputeExtendedK(
      prover->pcs().K());
  prover->set_extended_domain(
      PCS::ExtendedDomain::Create(size_t{1} << extended_k));

  tachyon_halo2_bn254_shplonk_prover_set_transcript_state(
      c_prover, transcript_state_bytes.data(), transcript_state_bytes.size());

  prover->SetRng(std::make_unique<crypto::XORShiftRNG>(
      crypto::XORShiftRNG::FromSeed(tachyon::zk::plonk::halo2::kXORShiftSeed)));

  std::cout << "deserializing argument data" << std::endl;
  ArgumentData arg_data = DeserializeArgumentData(arg_data_bytes);
  std::cout << "done deserializing argument data" << std::endl;
  for (size_t i = 0; i < arg_data.advice_blinds_vec().size(); ++i) {
    const std::vector<tachyon::math::bn254::Fr>& advice_blinds =
        arg_data.advice_blinds_vec()[i];
    for (size_t j = 0; j < advice_blinds.size(); ++j) {
      // Update Rng state
      prover->blinder().Generate();
    }
  }

  prover->blinder().set_blinding_factors(
      pk.verifying_key().constraint_system().ComputeBlindingFactors());
  prover->CreateProof(pk, &arg_data);
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
  if (base::Environment::Has("TACHYON_ARG_DATA_LOG_PATH")) {
    tachyon_cerr << "If this is set, the arg data log is overwritten"
                 << std::endl;
    return 1;
  }
  if (base::Environment::Has("TACHYON_TRANSCRIPT_STATE_LOG_PATH")) {
    tachyon_cerr << "If this is set, the transcript state log is overwritten"
                 << std::endl;
    return 1;
  }

  zk::plonk::halo2::TranscriptType transcript_type;
  uint32_t k;
  std::string s_hex;
  base::FilePath pcs_params_path;
  base::FilePath pk_path;
  base::FilePath arg_data_path;
  base::FilePath transcript_state_path;
  base::FlagParser parser;
  parser.AddFlag<base::Flag<zk::plonk::halo2::TranscriptType>>(&transcript_type)
      .set_long_name("--transcript_type")
      .set_required()
      .set_help("Transcript type");
  parser.AddFlag<base::Uint32Flag>(&k)
      .set_short_name("-k")
      .set_required()
      .set_help("K");
  parser.AddFlag<base::StringFlag>(&s_hex).set_short_name("-s").set_help(
      "s in hex");
  parser.AddFlag<base::FilePathFlag>(&pcs_params_path)
      .set_long_name("--pcs_params")
      .set_help("The path to pcs params");
  parser.AddFlag<base::FilePathFlag>(&pk_path)
      .set_long_name("--pk")
      .set_required()
      .set_help("The path to proving key");
  parser.AddFlag<base::FilePathFlag>(&arg_data_path)
      .set_long_name("--arg_data")
      .set_required()
      .set_help("The path to argument data");
  parser.AddFlag<base::FilePathFlag>(&transcript_state_path)
      .set_long_name("--transcript_state")
      .set_required()
      .set_help("The path to transcript state");
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

  std::optional<std::vector<uint8_t>> arg_data_bytes =
      base::ReadFileToBytes(arg_data_path);
  if (!arg_data_bytes.has_value()) {
    tachyon_cerr << "Failed to read file: " << arg_data_path.value()
                 << std::endl;
    return 1;
  }

  std::optional<std::vector<uint8_t>> transcript_state_bytes =
      base::ReadFileToBytes(transcript_state_path);
  if (!transcript_state_bytes.has_value()) {
    tachyon_cerr << "Failed to read file: " << transcript_state_path.value()
                 << std::endl;
    return 1;
  }

  tachyon_halo2_bn254_shplonk_prover* prover;
  std::optional<std::vector<uint8_t>> pcs_params_bytes;
  if (!pcs_params_path.empty()) {
    pcs_params_bytes = base::ReadFileToBytes(pcs_params_path);
  }

  if (pcs_params_bytes.has_value()) {
    std::cout << "creating prover" << std::endl;
    prover = tachyon_halo2_bn254_shplonk_prover_create_from_params(
        static_cast<uint8_t>(transcript_type), k, pcs_params_bytes->data(),
        pcs_params_bytes->size());
    std::cout << "done creating prover" << std::endl;
  } else {
    if (s_hex.empty()) {
      tachyon_cerr << "s_hex is empty" << std::endl;
      return 1;
    }
    math::bn254::Fr cpp_s = math::bn254::Fr::FromHexString(s_hex);
    const tachyon_bn254_fr& s = cc::math::c_cast(cpp_s);

    std::cout << "creating prover" << std::endl;
    prover = tachyon_halo2_bn254_shplonk_prover_create_from_unsafe_setup(
        static_cast<uint8_t>(transcript_type), k, &s);
    std::cout << "done creating prover" << std::endl;
    if (!pcs_params_path.empty()) {
      c::zk::plonk::halo2::bn254::WriteParams(prover, pcs_params_path);
    }
  }

  c::zk::plonk::halo2::bn254::CreateProof(prover, pk_bytes.value(),
                                          arg_data_bytes.value(),
                                          transcript_state_bytes.value());

  std::vector<uint8_t> proof;
  size_t proof_size;
  tachyon_halo2_bn254_shplonk_prover_get_proof(prover, nullptr, &proof_size);
  proof.resize(proof_size);
  tachyon_halo2_bn254_shplonk_prover_get_proof(prover, proof.data(),
                                               &proof_size);

  std::cout << "proof: [";
  for (size_t i = 0; i < proof_size; ++i) {
    std::cout << uint32_t{proof[i]};
    if (i != proof_size - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;

  tachyon_halo2_bn254_shplonk_prover_destroy(prover);
  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RunMain(argc, argv); }
