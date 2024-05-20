#include "benchmark/poseidon2/poseidon2_config.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/flag/flag_parser.h"

namespace tachyon {
namespace base {

template <>
class FlagValueTraits<Poseidon2Config::PrimeField> {
 public:
  static bool ParseValue(std::string_view input,
                         Poseidon2Config::PrimeField* value,
                         std::string* reason) {
    if (input == "bn254_fr") {
      *value = Poseidon2Config::PrimeField::kBn254Fr;
    } else {
      *reason = absl::Substitute("Unknown test set: $0", input);
      return false;
    }
    return true;
  }
};

template <>
class FlagValueTraits<Poseidon2Config::Vendor> {
 public:
  static bool ParseValue(std::string_view input, Poseidon2Config::Vendor* value,
                         std::string* reason) {
    if (input == "horizen") {
      *value = Poseidon2Config::Vendor::kHorizen;
    } else if (input == "plonky3") {
      *value = Poseidon2Config::Vendor::kPlonky3;
    } else {
      *reason = absl::Substitute("Unknown test set: $0", input);
      return false;
    }
    return true;
  }
};

}  // namespace base

// static
std::string Poseidon2Config::VendorToString(Poseidon2Config::Vendor vendor) {
  switch (vendor) {
    case Poseidon2Config::Vendor::kHorizen:
      return "horizen";
    case Poseidon2Config::Vendor::kPlonky3:
      return "plonky3";
  }
  NOTREACHED();
  return "";
}

bool Poseidon2Config::Parse(int argc, char** argv) {
  base::FlagParser parser;

  parser.AddFlag<base::Flag<size_t>>(&repeating_num_)
      .set_short_name("-n")
      .set_help("Specify the number of repetition 'n'. By default, 10.");
  parser.AddFlag<base::Flag<bool>>(&check_results_)
      .set_long_name("--check_results")
      .set_help("Whether checks results generated by each poseidon2 runner.");
  parser.AddFlag<base::Flag<PrimeField>>(&prime_field_)
      .set_short_name("-p")
      .set_long_name("--prime_field")
      .set_help(
          "A prime field to be benchmarked with. (supported prime fields: "
          "bn254_fr)");
  parser.AddFlag<base::Flag<std::vector<Vendor>>>(&vendors_)
      .set_long_name("--vendor")
      .set_help(
          "Vendors to be benchmarked with. (supported vendors: horizen, "
          "plonky3)");

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return false;
  }

  return true;
}

}  // namespace tachyon