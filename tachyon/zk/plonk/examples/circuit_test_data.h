#ifndef TACHYON_ZK_PLONK_EXAMPLES_CIRCUIT_TEST_DATA_H_
#define TACHYON_ZK_PLONK_EXAMPLES_CIRCUIT_TEST_DATA_H_

#include <string_view>
#include <utility>
#include <vector>

#include "tachyon/base/range.h"
#include "tachyon/zk/plonk/examples/point.h"

namespace tachyon::zk::plonk {

template <typename Circuit, typename PCS, typename LS>
class CircuitTestData {
 public:
  using Evals = typename PCS::Evals;
  using RationalEvals = typename PCS::RationalEvals;

  constexpr static bool kAssemblyFixedColumnsFlag = false;
  constexpr static bool kAssemblyPermutationColumnsFlag = false;
  constexpr static bool kCycleStoreMappingFlag = false;
  constexpr static bool kCycleStoreAuxFlag = false;
  constexpr static bool kCycleStoreSizesFlag = false;
  constexpr static bool kLFirstFlag = false;
  constexpr static bool kLLastFlag = false;
  constexpr static bool kLActiveRowFlag = false;
  constexpr static bool kFixedColumnsFlag = false;
  constexpr static bool kFixedPolysFlag = false;
  constexpr static bool kPermutationsColumnsFlag = false;
  constexpr static bool kPermutationsPolysFlag = false;
  constexpr static bool kAdviceCommitmentsFlag = false;
  constexpr static bool kChallengesFlag = false;
  constexpr static bool kLookupPermutedCommitmentsFlag = false;
  constexpr static bool kLookupMPolyCommitmentsFlag = false;
  constexpr static bool kPermutationProductCommitmentsFlag = false;
  constexpr static bool kLookupProductCommitmentsFlag = false;
  constexpr static bool kLookupSumCommitmentsFlag = false;
  constexpr static bool kShuffleProductCommitmentsFlag = false;
  constexpr static bool kAdviceEvalsFlag = false;
  constexpr static bool kFixedEvalsFlag = false;
  constexpr static bool kCommonPermutationEvalsFlag = false;
  constexpr static bool kPermutationProductEvalsFlag = false;
  constexpr static bool kPermutationProductNextEvalsFlag = false;
  constexpr static bool kPermutationProductLastEvalsFlag = false;
  constexpr static bool kLookupProductEvalsFlag = false;
  constexpr static bool kLookupSumEvalsFlag = false;
  constexpr static bool kLookupProductNextEvalsFlag = false;
  constexpr static bool kLookupSumNextEvalsFlag = false;
  constexpr static bool kLookupPermutedInputEvalsFlag = false;
  constexpr static bool kLookupPermutedInputPrevEvalsFlag = false;
  constexpr static bool kLookupPermutedTableEvalsFlag = false;
  constexpr static bool kLookupMEvalsFlag = false;
  constexpr static bool kShuffleProductEvalsFlag = false;
  constexpr static bool kShuffleProductNextEvalsFlag = false;

  constexpr static base::Range<RowIndex> kUsableRows =
      base::Range<RowIndex>::Until(10);
  constexpr static Point kVanishingRandomPolyCommitment{
      "0x0000000000000000000000000000000000000000000000000000000000000001",
      "0x0000000000000000000000000000000000000000000000000000000000000002"};
  constexpr static std::string_view kVanishingRandomEval =
      "0x0000000000000000000000000000000000000000000000000000000000000001";

  static Circuit GetCircuit() { return Circuit(); }

  static std::vector<Circuit> Get2Circuits() {
    Circuit circuit = GetCircuit();
    return {circuit, std::move(circuit)};
  }

  static std::vector<RationalEvals> GetFixedColumns() { return {}; }

  static std::vector<Evals> GetInstanceColumns() { return {}; }
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_CIRCUIT_TEST_DATA_H_
