#ifndef VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_WITNESS_LOADER_H_
#define VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_WITNESS_LOADER_H_

#include <memory>
#include <string>
#include <vector>

#include "circomlib/base/fr_element_conversion.h"
#include "circomlib/generated/common/calcwit.hpp"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/files/file_path.h"

namespace tachyon::circom {

template <typename F>
class WitnessLoader {
 public:
  explicit WitnessLoader(const base::FilePath& data)
      : circuit_(loadCircuit(data.value())),
        calc_wit_(new Circom_CalcWit(circuit_.get())) {}

  void Set(std::string_view label, const F& value) {
    witness_[label] = std::vector<FrElement>{ConvertToFrElement(value)};
  }

  void Set(std::string_view label, const std::vector<F>& values) {
    witness_[label] = base::Map(
        values, [](const F& value) { return ConvertToFrElement(value); });
  }

  void Load() { loadWitness(calc_wit_.get(), witness_); }

  F Get(uint32_t i) const {
    FrElement v;
    calc_wit_->getWitness(i, &v);
    return ConvertFromFrElement<F>(v);
  }

 private:
  std::unique_ptr<Circom_Circuit> circuit_;
  std::unique_ptr<Circom_CalcWit> calc_wit_;
  absl::flat_hash_map<std::string, std::vector<FrElement>> witness_;
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_WITNESS_LOADER_H_
