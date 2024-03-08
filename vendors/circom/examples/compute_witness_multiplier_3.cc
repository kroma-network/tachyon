#include <memory>

#include "circomlib/generated/common/calcwit.hpp"

namespace tachyon {

int RealMain(int argc, char **argv) {
  std::unique_ptr<Circom_Circuit> circuit(
      loadCircuit("examples/multiplier_3_cpp/multiplier_3.dat"));
  std::unique_ptr<Circom_CalcWit> ctx(new Circom_CalcWit(circuit.get()));
  absl::flat_hash_map<std::string, std::vector<FrElement>> witness;
  std::vector<FrElement> in(3);
  Fr_str2element(&in[0], "3", 10);
  Fr_str2element(&in[1], "4", 10);
  Fr_str2element(&in[2], "5", 10);
  witness["in"] = std::move(in);
  loadWitness(ctx.get(), witness);
  writeBinWitness(ctx.get(), "multiplier_3.wtns");
  return 0;
}

}  // namespace tachyon

int main(int argc, char **argv) { return tachyon::RealMain(argc, argv); }
