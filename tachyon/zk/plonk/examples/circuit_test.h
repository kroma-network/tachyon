#ifndef TACHYON_ZK_PLONK_EXAMPLES_CIRCUIT_TEST_H_
#define TACHYON_ZK_PLONK_EXAMPLES_CIRCUIT_TEST_H_

#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/math/elliptic_curves/bn/bn254/halo2/bn254.h"
#include "tachyon/zk/lookup/pair.h"
#include "tachyon/zk/plonk/examples/circuit_test_type_traits.h"
#include "tachyon/zk/plonk/examples/point.h"
#include "tachyon/zk/plonk/halo2/pinned_constraint_system.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/halo2/prover_test.h"

namespace tachyon::zk::plonk {

template <typename _Circuit, typename _PCS, typename _LS>
struct TestArguments {
  using Circuit = _Circuit;
  using PCS = _PCS;
  using LS = _LS;
};

template <typename TestArguments, typename TestData>
class CircuitTest : public halo2::ProverTest<typename TestArguments::PCS,
                                             typename TestArguments::LS> {
 public:
  using Circuit = typename TestArguments::Circuit;
  using PCS = typename TestArguments::PCS;
  using LS = typename TestArguments::LS;
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;
  using Evals = typename PCS::Evals;
  using Domain = typename PCS::Domain;
  using Commitment = typename PCS::Commitment;
  using RationalEvals = typename PCS::RationalEvals;

  static void SetUpTestSuite() {
    math::bn254::BN254Curve::Init();
    math::halo2::OverrideSubgroupGenerator();
  }

  void ConfigureTest() {
    ConstraintSystem<F> constraint_system;
    constraint_system.set_lookup_type(TestArguments::LS::type);

    auto config = Circuit::Configure(constraint_system);

    TestData::TestConfig(config);

    halo2::PinnedConstraintSystem<F> pinned_constraint_system(
        constraint_system);
    EXPECT_EQ(TestData::kPinnedConstraintSystem,
              base::ToRustDebugString(pinned_constraint_system));

    EXPECT_TRUE(constraint_system.selector_map().empty());
    EXPECT_TRUE(constraint_system.general_column_annotations().empty());
  }

  void SynthesizeTest() {
    CHECK(this->prover_->pcs().UnsafeSetup(TestData::kN, F(2)));
    this->prover_->set_domain(Domain::Create(TestData::kN));
    const Domain* domain = this->prover_->domain();

    ConstraintSystem<F> constraint_system;
    constraint_system.set_lookup_type(TestArguments::LS::type);

    auto config = Circuit::Configure(constraint_system);
    Assembly<RationalEvals> assembly =
        VerifyingKey<F, Commitment>::template CreateAssembly<RationalEvals>(
            domain, constraint_system);

    Circuit circuit = TestData::GetCircuit();
    typename Circuit::FloorPlanner floor_planner;
    floor_planner.Synthesize(&assembly, circuit, std::move(config),
                             constraint_system.constants());

    if constexpr (TestData::kAssemblyFixedColumnsFlag) {
      std::vector<RationalEvals> expected_fixed_columns = CreateRationalColumns(
          base::Array2DToVector2D(TestData::kAssemblyFixedColumns));
      EXPECT_EQ(assembly.fixed_columns(), expected_fixed_columns);
    } else {
      EXPECT_TRUE(assembly.fixed_columns().empty());
    }

    if constexpr (TestData::kAssemblyPermutationColumnsFlag) {
      std::vector<AnyColumnKey> expected_columns =
          base::ArrayToVector(TestData::kAssemblyPermutationColumns);
      EXPECT_EQ(assembly.permutation().columns(), expected_columns);
    } else {
      EXPECT_TRUE(assembly.permutation().columns().empty());
    }

    const CycleStore& cycle_store = assembly.permutation().cycle_store();

    if constexpr (TestData::kCycleStoreMappingFlag) {
      CycleStore::Table<Label> expected_mapping(
          base::Array2DToVector2D(TestData::kCycleStoreMapping));
      EXPECT_EQ(cycle_store.mapping(), expected_mapping);
    } else {
      EXPECT_TRUE(cycle_store.mapping().IsEmpty());
    }

    if constexpr (TestData::kCycleStoreAuxFlag) {
      CycleStore::Table<Label> expected_aux(
          base::Array2DToVector2D(TestData::kCycleStoreAux));
      EXPECT_EQ(cycle_store.aux(), expected_aux);
    } else {
      EXPECT_TRUE(cycle_store.aux().IsEmpty());
    }

    if constexpr (TestData::kCycleStoreSizesFlag) {
      CycleStore::Table<size_t> expected_sizes(
          base::Array2DToVector2D(TestData::kCycleStoreSizes));
      EXPECT_EQ(cycle_store.sizes(), expected_sizes);
    } else {
      EXPECT_TRUE(cycle_store.sizes().IsEmpty());
    }

    std::vector<std::vector<bool>> expected_selectors =
        base::Array2DToVector2D(TestData::kSelectors);
    EXPECT_EQ(assembly.selectors(), expected_selectors);

    EXPECT_EQ(assembly.usable_rows(), TestData::kUsableRows);
  }

  void LoadVerifyingKeyTest() {
    CHECK(this->prover_->pcs().UnsafeSetup(TestData::kN, F(2)));
    this->prover_->set_domain(Domain::Create(TestData::kN));

    Circuit circuit = TestData::GetCircuit();

    VerifyingKey<F, Commitment> vkey;
    ASSERT_TRUE(
        vkey.Load(this->prover_.get(), circuit, TestArguments::LS::type));

    halo2::PinnedVerifyingKey pinned_vkey(this->prover_.get(), vkey);
    EXPECT_EQ(base::ToRustDebugString(pinned_vkey),
              TestData::kPinnedVerifyingKey);

    F expected_transcript_repr = *F::FromHexString(TestData::kTranscriptRepr);
    EXPECT_EQ(vkey.transcript_repr(), expected_transcript_repr);
  }

  void LoadProvingKeyTest() {
    CHECK(this->prover_->pcs().UnsafeSetup(TestData::kN, F(2)));
    this->prover_->set_domain(Domain::Create(TestData::kN));

    Circuit circuit = TestData::GetCircuit();

    for (size_t i = 0; i < 2; ++i) {
      ProvingKey<LS> pkey;
      bool load_verifying_key = i == 0;
      SCOPED_TRACE(
          absl::Substitute("load_verifying_key: $0", load_verifying_key));
      if (load_verifying_key) {
        VerifyingKey<F, Commitment> vkey;
        ASSERT_TRUE(
            vkey.Load(this->prover_.get(), circuit, TestArguments::LS::type));
        ASSERT_TRUE(pkey.LoadWithVerifyingKey(this->prover_.get(), circuit,
                                              std::move(vkey)));
      } else {
        ASSERT_TRUE(pkey.Load(this->prover_.get(), circuit));
      }

      if constexpr (TestData::kLFirstFlag) {
        Poly expected_l_first =
            CreatePoly(base::ArrayToVector(TestData::kLFirst));
        EXPECT_EQ(pkey.l_first(), expected_l_first);
      } else {
        EXPECT_EQ(pkey.l_first().NumElements(), 0);
      }

      if constexpr (TestData::kLLastFlag) {
        Poly expected_l_last =
            CreatePoly(base::ArrayToVector(TestData::kLLast));
        EXPECT_EQ(pkey.l_last(), expected_l_last);
      } else {
        EXPECT_EQ(pkey.l_last().NumElements(), 0);
      }

      if constexpr (TestData::kLActiveRowFlag) {
        Poly expected_l_active_row =
            CreatePoly(base::ArrayToVector(TestData::kLActiveRow));
        EXPECT_EQ(pkey.l_active_row(), expected_l_active_row);
      } else {
        EXPECT_EQ(pkey.l_active_row().NumElements(), 0);
      }

      if constexpr (TestData::kFixedColumnsFlag) {
        std::vector<Evals> expected_fixed_columns =
            CreateColumns(base::Array2DToVector2D(TestData::kFixedColumns));
        EXPECT_EQ(pkey.fixed_columns(), expected_fixed_columns);
      } else {
        EXPECT_TRUE(pkey.fixed_columns().empty());
      }

      if constexpr (TestData::kFixedPolysFlag) {
        std::vector<Poly> expected_fixed_polys =
            CreatePolys(base::Array2DToVector2D(TestData::kFixedPolys));
        EXPECT_EQ(pkey.fixed_polys(), expected_fixed_polys);
      } else {
        EXPECT_TRUE(pkey.fixed_polys().empty());
      }

      if constexpr (TestData::kPermutationsColumnsFlag) {
        std::vector<Evals> expected_permutations_columns = CreateColumns(
            base::Array2DToVector2D(TestData::kPermutationsColumns));
        EXPECT_EQ(pkey.permutation_proving_key().permutations(),
                  expected_permutations_columns);
      } else {
        EXPECT_TRUE(pkey.permutation_proving_key().permutations().empty());
      }

      if constexpr (TestData::kPermutationsPolysFlag) {
        std::vector<Poly> expected_fixed_polys =
            CreatePolys(base::Array2DToVector2D(TestData::kPermutationsPolys));
        EXPECT_EQ(pkey.permutation_proving_key().polys(), expected_fixed_polys);
      } else {
        EXPECT_TRUE(pkey.permutation_proving_key().polys().empty());
      }
    }
  }

  void CreateProofTest() {
    CHECK(this->prover_->pcs().UnsafeSetup(TestData::kN, F(2)));
    this->prover_->set_domain(Domain::Create(TestData::kN));

    std::vector<Circuit> circuits = TestData::Get2Circuits();

    std::vector<Evals> instance_columns = TestData::GetInstanceColumns();
    std::vector<std::vector<Evals>> instance_columns_vec = {
        instance_columns, std::move(instance_columns)};

    ProvingKey<LS> pkey;
    ASSERT_TRUE(pkey.Load(this->prover_.get(), circuits[0]));
    this->prover_->CreateProof(pkey, std::move(instance_columns_vec), circuits);

    std::vector<uint8_t> proof =
        this->prover_->GetWriter()->buffer().owned_buffer();
    EXPECT_EQ(proof, base::ArrayToVector(TestData::kProof));
  }

  void VerifyProofTest() {
    using Proof = typename TestArguments::LS::Proof;
    CHECK(this->prover_->pcs().UnsafeSetup(TestData::kN, F(2)));
    this->prover_->set_domain(Domain::Create(TestData::kN));

    Circuit circuit = TestData::GetCircuit();

    VerifyingKey<F, Commitment> vkey;
    ASSERT_TRUE(
        vkey.Load(this->prover_.get(), circuit, TestArguments::LS::type));

    std::vector<uint8_t> owned_proof = base::ArrayToVector(TestData::kProof);

    halo2::Verifier<PCS, LS> verifier = this->CreateVerifier(
        CreateBufferWithProof(absl::MakeSpan(owned_proof)));

    std::vector<Evals> instance_columns = TestData::GetInstanceColumns();
    std::vector<std::vector<Evals>> instance_columns_vec = {
        instance_columns, std::move(instance_columns)};

    Proof proof;
    F h_eval;
    ASSERT_TRUE(verifier.VerifyProofForTesting(vkey, instance_columns_vec,
                                               &proof, &h_eval));

    if constexpr (TestData::kAdviceCommitmentsFlag) {
      std::vector<std::vector<Commitment>> expected_advice_commitments_vec{
          CreateCommitments(
              base::ArrayToVector(TestData::kAdviceCommitments[0])),
          CreateCommitments(
              base::ArrayToVector(TestData::kAdviceCommitments[1])),
      };
      EXPECT_EQ(proof.advices_commitments_vec, expected_advice_commitments_vec);
    } else {
      EXPECT_TRUE(proof.advices_commitments_vec[0].empty());
    }

    if constexpr (TestData::kChallengesFlag) {
      std::vector<F> expected_challenges =
          CreateEvals(base::ArrayToVector(TestData::kChallenges));
      EXPECT_EQ(proof.challenges, expected_challenges);
    } else {
      EXPECT_TRUE(proof.challenges.empty());
    }

    F expected_theta = *F::FromHexString(TestData::kTheta);
    EXPECT_EQ(proof.theta, expected_theta);

    if constexpr (TestData::kLookupPermutedCommitmentsFlag) {
      std::vector<std::vector<lookup::Pair<Commitment>>>
          expected_lookup_permuted_commitments_vec{
              CreateLookupPermutedCommitments(
                  base::ArrayToVector(
                      TestData::kLookupPermutedCommitmentsInput[0]),
                  base::ArrayToVector(
                      TestData::kLookupPermutedCommitmentsTable[0])),
              CreateLookupPermutedCommitments(
                  base::ArrayToVector(
                      TestData::kLookupPermutedCommitmentsInput[1]),
                  base::ArrayToVector(
                      TestData::kLookupPermutedCommitmentsTable[1])),
          };
      EXPECT_EQ(proof.lookup_permuted_commitments_vec,
                expected_lookup_permuted_commitments_vec);
    } else if constexpr (TestData::kLookupMPolyCommitmentsFlag) {
      std::vector<std::vector<Commitment>>
          expected_lookup_m_poly_commitments_vec{
              CreateCommitments(
                  base::ArrayToVector(TestData::kLookupMPolyCommitments[0])),
              CreateCommitments(
                  base::ArrayToVector(TestData::kLookupMPolyCommitments[1])),
          };
      EXPECT_EQ(proof.lookup_m_poly_commitments_vec,
                expected_lookup_m_poly_commitments_vec);
    }

    F expected_beta = *F::FromHexString(TestData::kBeta);
    EXPECT_EQ(proof.beta, expected_beta);

    F expected_gamma = *F::FromHexString(TestData::kGamma);
    EXPECT_EQ(proof.gamma, expected_gamma);

    if constexpr (TestData::kPermutationProductCommitmentsFlag) {
      std::vector<std::vector<Commitment>>
          expected_permutation_product_commitments_vec{
              CreateCommitments(base::ArrayToVector(
                  TestData::kPermutationProductCommitments[0])),
              CreateCommitments(base::ArrayToVector(
                  TestData::kPermutationProductCommitments[1])),
          };
      EXPECT_EQ(proof.permutation_product_commitments_vec,
                expected_permutation_product_commitments_vec);
    } else {
      EXPECT_TRUE(proof.permutation_product_commitments_vec[0].empty());
    }

    if constexpr (TestData::kLookupProductCommitmentsFlag) {
      std::vector<std::vector<Commitment>>
          expected_lookup_product_commitments_vec{
              CreateCommitments(
                  base::ArrayToVector(TestData::kLookupProductCommitments[0])),
              CreateCommitments(
                  base::ArrayToVector(TestData::kLookupProductCommitments[1])),
          };
      EXPECT_EQ(proof.lookup_product_commitments_vec,
                expected_lookup_product_commitments_vec);
    } else if constexpr (TestData::kLookupSumCommitmentsFlag) {
      std::vector<std::vector<Commitment>>
          expected_lookup_sum_commitments_commitments_vec{
              CreateCommitments(
                  base::ArrayToVector(TestData::kLookupSumCommitments[0])),
              CreateCommitments(
                  base::ArrayToVector(TestData::kLookupSumCommitments[1])),
          };
      EXPECT_EQ(proof.lookup_sum_commitments_vec,
                expected_lookup_sum_commitments_commitments_vec);
    }

    Commitment expected_vanishing_random_poly_commitment =
        CreateCommitment(TestData::kVanishingRandomPolyCommitment);
    EXPECT_EQ(proof.vanishing_random_poly_commitment,
              expected_vanishing_random_poly_commitment);

    F expected_y = *F::FromHexString(TestData::kY);
    EXPECT_EQ(proof.y, expected_y);

    std::vector<Commitment> expected_vanishing_h_poly_commitments =
        CreateCommitments(
            base::ArrayToVector(TestData::kVanishingHPolyCommitments));
    EXPECT_EQ(proof.vanishing_h_poly_commitments,
              expected_vanishing_h_poly_commitments);

    F expected_x = *F::FromHexString(TestData::kX);
    EXPECT_EQ(proof.x, expected_x);

    if constexpr (TestData::kAdviceEvalsFlag) {
      std::vector<std::vector<F>> expected_advice_evals_vec{
          CreateEvals(base::ArrayToVector(TestData::kAdviceEvals[0])),
          CreateEvals(base::ArrayToVector(TestData::kAdviceEvals[1])),
      };
      EXPECT_EQ(proof.advice_evals_vec, expected_advice_evals_vec);
    } else {
      EXPECT_TRUE(proof.advice_evals_vec[0].empty());
    }

    if constexpr (TestData::kFixedEvalsFlag) {
      std::vector<F> expected_fixed_evals =
          CreateEvals(base::ArrayToVector(TestData::kFixedEvals));
      EXPECT_EQ(proof.fixed_evals, expected_fixed_evals);
    } else {
      EXPECT_TRUE(proof.fixed_evals.empty());
    }

    F expected_vanishing_random_eval =
        *F::FromHexString(TestData::kVanishingRandomEval);
    EXPECT_EQ(proof.vanishing_random_eval, expected_vanishing_random_eval);

    if constexpr (TestData::kCommonPermutationEvalsFlag) {
      std::vector<F> expected_common_permutation_evals =
          CreateEvals(base::ArrayToVector(TestData::kCommonPermutationEvals));
      EXPECT_EQ(proof.common_permutation_evals,
                expected_common_permutation_evals);
    } else {
      EXPECT_TRUE(proof.common_permutation_evals.empty());
    }

    if constexpr (TestData::kPermutationProductEvalsFlag) {
      std::vector<std::vector<F>> expected_permutation_product_evals_vec{
          CreateEvals(
              base::ArrayToVector(TestData::kPermutationProductEvals[0])),
          CreateEvals(
              base::ArrayToVector(TestData::kPermutationProductEvals[1])),
      };
      EXPECT_EQ(proof.permutation_product_evals_vec,
                expected_permutation_product_evals_vec);
    } else {
      EXPECT_TRUE(proof.permutation_product_evals_vec[0].empty());
    }

    if constexpr (TestData::kPermutationProductNextEvalsFlag) {
      std::vector<std::vector<F>> expected_permutation_product_next_evals_vec{
          CreateEvals(
              base::ArrayToVector(TestData::kPermutationProductNextEvals[0])),
          CreateEvals(
              base::ArrayToVector(TestData::kPermutationProductNextEvals[1])),
      };
      EXPECT_EQ(proof.permutation_product_next_evals_vec,
                expected_permutation_product_next_evals_vec);
    } else {
      EXPECT_TRUE(proof.permutation_product_next_evals_vec[0].empty());
    }

    if constexpr (TestData::kPermutationProductLastEvalsFlag) {
      std::vector<std::vector<std::optional<F>>>
          expected_permutation_product_last_evals_vec{
              CreateOptionalEvals(base::ArrayToVector(
                  TestData::kPermutationProductLastEvals[0])),
              CreateOptionalEvals(base::ArrayToVector(
                  TestData::kPermutationProductLastEvals[1])),
          };
      EXPECT_EQ(proof.permutation_product_last_evals_vec,
                expected_permutation_product_last_evals_vec);
    } else {
      EXPECT_TRUE(proof.permutation_product_last_evals_vec[0].empty());
    }

    if constexpr (TestData::kLookupProductEvalsFlag) {
      std::vector<std::vector<F>> expected_lookup_product_evals_vec{
          CreateEvals(base::ArrayToVector(TestData::kLookupProductEvals[0])),
          CreateEvals(base::ArrayToVector(TestData::kLookupProductEvals[1])),
      };
      EXPECT_EQ(proof.lookup_product_evals_vec,
                expected_lookup_product_evals_vec);
    } else if constexpr (TestData::kLookupSumEvalsFlag) {
      std::vector<std::vector<F>> expected_lookup_sum_evals_vec{
          CreateEvals(base::ArrayToVector(TestData::kLookupSumEvals[0])),
          CreateEvals(base::ArrayToVector(TestData::kLookupSumEvals[1])),
      };
      EXPECT_EQ(proof.lookup_sum_evals_vec, expected_lookup_sum_evals_vec);
    }

    if constexpr (TestData::kLookupProductNextEvalsFlag) {
      std::vector<std::vector<F>> expected_lookup_product_next_evals_vec{
          CreateEvals(
              base::ArrayToVector(TestData::kLookupProductNextEvals[0])),
          CreateEvals(
              base::ArrayToVector(TestData::kLookupProductNextEvals[1])),
      };
      EXPECT_EQ(proof.lookup_product_next_evals_vec,
                expected_lookup_product_next_evals_vec);
    } else if constexpr (TestData::kLookupSumNextEvalsFlag) {
      std::vector<std::vector<F>> expected_lookup_sum_next_evals_vec{
          CreateEvals(base::ArrayToVector(TestData::kLookupSumNextEvals[0])),
          CreateEvals(base::ArrayToVector(TestData::kLookupSumNextEvals[1])),
      };
      EXPECT_EQ(proof.lookup_sum_next_evals_vec,
                expected_lookup_sum_next_evals_vec);
    }

    if constexpr (TestData::kLookupPermutedInputEvalsFlag) {
      std::vector<std::vector<F>> expected_lookup_permuted_input_evals_vec{
          CreateEvals(
              base::ArrayToVector(TestData::kLookupPermutedInputEvals[0])),
          CreateEvals(
              base::ArrayToVector(TestData::kLookupPermutedInputEvals[1])),
      };
      EXPECT_EQ(proof.lookup_permuted_input_evals_vec,
                expected_lookup_permuted_input_evals_vec);
    }

    if constexpr (TestData::kLookupPermutedInputPrevEvalsFlag) {
      std::vector<std::vector<F>> expected_lookup_permuted_input_prev_evals_vec{
          CreateEvals(
              base::ArrayToVector(TestData::kLookupPermutedInputPrevEvals[0])),
          CreateEvals(
              base::ArrayToVector(TestData::kLookupPermutedInputPrevEvals[1])),
      };
      EXPECT_EQ(proof.lookup_permuted_input_prev_evals_vec,
                expected_lookup_permuted_input_prev_evals_vec);
    }

    if constexpr (TestData::kLookupPermutedTableEvalsFlag) {
      std::vector<std::vector<F>> expected_lookup_permuted_table_evals_vec{
          CreateEvals(
              base::ArrayToVector(TestData::kLookupPermutedTableEvals[0])),
          CreateEvals(
              base::ArrayToVector(TestData::kLookupPermutedTableEvals[1])),
      };
      EXPECT_EQ(proof.lookup_permuted_table_evals_vec,
                expected_lookup_permuted_table_evals_vec);
    }

    if constexpr (TestData::kLookupMEvalsFlag) {
      std::vector<std::vector<F>> expected_lookup_m_evals_vec{
          CreateEvals(base::ArrayToVector(TestData::kLookupMEvals[0])),
          CreateEvals(base::ArrayToVector(TestData::kLookupMEvals[1])),
      };
      EXPECT_EQ(proof.lookup_m_evals_vec, expected_lookup_m_evals_vec);
    }

    // TODO(ashjeong): get |h_eval| for fibonacci tests
    if constexpr (!IsFibonacci<Circuit>) {
      F expected_h_eval = *F::FromHexString(TestData::kHEval);
      EXPECT_EQ(h_eval, expected_h_eval);
    }
  }

 private:
  static Commitment CreateCommitment(const Point& point) {
    using BaseField = typename Commitment::BaseField;
    return Commitment(*BaseField::FromHexString(point.x),
                      *BaseField::FromHexString(point.y));
  }

  static std::vector<Commitment> CreateCommitments(
      const std::vector<Point>& points) {
    return base::Map(points, &CreateCommitment);
  }

  static std::vector<lookup::Pair<Commitment>> CreateLookupPermutedCommitments(
      const std::vector<Point>& input_points,
      const std::vector<Point>& table_points) {
    std::vector<lookup::Pair<Commitment>> lookup_pairs;
    return base::Map(
        input_points, [&table_points](size_t i, const Point& input_point) {
          return lookup::Pair<Commitment>(CreateCommitment(input_point),
                                          CreateCommitment(table_points[i]));
        });
  }

  static Evals CreateColumn(const std::vector<std::string_view>& column) {
    std::vector<F> evaluations = base::Map(column, [](std::string_view coeff) {
      return *F::FromHexString(coeff);
    });
    return Evals(std::move(evaluations));
  }

  static std::vector<Evals> CreateColumns(
      const std::vector<std::vector<std::string_view>>& columns) {
    return base::Map(columns, &CreateColumn);
  }

  static RationalEvals CreateRationalColumn(
      const std::vector<std::string_view>& column) {
    std::vector<math::RationalField<F>> evaluations =
        base::Map(column, [](std::string_view coeff) {
          return math::RationalField<F>(*F::FromHexString(coeff));
        });
    return RationalEvals(std::move(evaluations));
  }

  static std::vector<RationalEvals> CreateRationalColumns(
      const std::vector<std::vector<std::string_view>>& columns) {
    return base::Map(columns, &CreateRationalColumn);
  }

  static Poly CreatePoly(const std::vector<std::string_view>& poly) {
    std::vector<F> coefficients = base::Map(
        poly, [](std::string_view coeff) { return *F::FromHexString(coeff); });
    return Poly(math::UnivariateDenseCoefficients<F, halo2::kMaxDegree>(
        std::move(coefficients), true));
  }

  static std::vector<Poly> CreatePolys(
      const std::vector<std::vector<std::string_view>>& polys) {
    return base::Map(polys, &CreatePoly);
  }

  static std::vector<F> CreateEvals(
      const std::vector<std::string_view>& evals) {
    return base::Map(
        evals, [](std::string_view eval) { return *F::FromHexString(eval); });
  }

  static std::vector<std::optional<F>> CreateOptionalEvals(
      const std::vector<std::string_view>& evals) {
    return base::Map(evals, [](std::string_view eval) {
      if (eval.empty()) return std::optional<F>();
      return F::FromHexString(eval);
    });
  }

  static base::Buffer CreateBufferWithProof(absl::Span<uint8_t> proof) {
    return {proof.data(), proof.size()};
  }
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_CIRCUIT_TEST_H_
