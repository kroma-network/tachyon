#include "benchmark/fft/fft_config.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/flag/flag_parser.h"

namespace tachyon {

bool FFTConfig::Parse(int argc, char** argv) {
  base::FlagParser parser;
  // clang-format off
  parser.AddFlag<base::Flag<uint64_t>>(&k_)
      .set_short_name("-k")
      .set_required()
      .set_help("Specify the exponent 'k' where the degree of poly to test is 2ᵏ.");
  // clang-format on
  {
    std::string error;
    if (!parser.Parse(argc, argv, &error)) {
      tachyon_cerr << error << std::endl;
      return false;
    }
  }

  return true;
}

}  // namespace tachyon
