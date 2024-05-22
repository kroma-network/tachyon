#ifndef VENDORS_CIRCOM_CIRCOMLIB_JSON_GROTH16_PROOF_H_
#define VENDORS_CIRCOM_CIRCOMLIB_JSON_GROTH16_PROOF_H_

#include "circomlib/json/json_converter_forward.h"
#include "circomlib/json/points.h"
#include "tachyon/zk/r1cs/groth16/proof.h"

namespace tachyon::circom {

template <typename Curve>
class JsonSerializer<zk::r1cs::groth16::Proof<Curve>> {
 public:
  static rapidjson::Document ToJson(
      const zk::r1cs::groth16::Proof<Curve>& proof) {
    rapidjson::Document document;
    rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
    document.SetObject();

    internal::AddMember(document, "pi_a", proof.a());
    internal::AddMember(document, "pi_b", proof.b());
    internal::AddMember(document, "pi_c", proof.c());

    document.AddMember("protocol", "groth16", allocator);

    std::string_view curve_name = Curve::Config::kName;
    rapidjson::Value curve;
    if (curve_name == "tachyon::math::bn254::BN254") {
      curve.SetString("bn128");
    } else if (curve_name == "tachyon::math::bls12_381::BLS12_381") {
      curve.SetString("bls12381");
    } else {
      NOTREACHED();
    }
    document.AddMember("curve", curve, allocator);
    return document;
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_JSON_GROTH16_PROOF_H_
