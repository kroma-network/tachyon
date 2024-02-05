#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/examples/fibonacci/fibonacci4_circuit.h"
#include "tachyon/zk/plonk/examples/fibonacci/fibonacci4_circuit_test_data.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/keys/proving_key.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_floor_planner.h"

namespace tachyon::zk::plonk::halo2 {

namespace {

class Fibonacci4V1CircuitTest : public CircuitTest {};

}  // namespace

TEST_F(Fibonacci4V1CircuitTest, CreateProof) {
  size_t n = 2048;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  Fibonacci4Circuit<F, V1FloorPlanner> circuit;
  std::vector<Fibonacci4Circuit<F, V1FloorPlanner>> circuits = {
      circuit, std::move(circuit)};

  F a(1);
  F b(1);
  F out(21);
  std::vector<F> instance_column = {std::move(a), std::move(b), std::move(out)};
  std::vector<Evals> instance_columns = {Evals(std::move(instance_column))};
  std::vector<std::vector<Evals>> instance_columns_vec = {
      instance_columns, std::move(instance_columns)};

  ProvingKey<Poly, Evals, Commitment> pkey;
  ASSERT_TRUE(pkey.Load(prover_.get(), circuit));
  prover_->CreateProof(pkey, std::move(instance_columns_vec), circuits);

  std::vector<uint8_t> proof = prover_->GetWriter()->buffer().owned_buffer();
  std::vector<uint8_t> expected_proof(std::begin(fibonacci4_v1::kExpectedProof),
                                      std::end(fibonacci4_v1::kExpectedProof));
  EXPECT_THAT(proof, testing::ContainerEq(expected_proof));
}

TEST_F(Fibonacci4V1CircuitTest, VerifyProof) {
  size_t n = 2048;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  Fibonacci4Circuit<F, V1FloorPlanner> circuit;

  VerifyingKey<F, Commitment> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  std::vector<uint8_t> owned_proof(std::begin(fibonacci4_v1::kExpectedProof),
                                   std::end(fibonacci4_v1::kExpectedProof));
  Verifier<PCS> verifier =
      CreateVerifier(CreateBufferWithProof(absl::MakeSpan(owned_proof)));

  F a(1);
  F b(1);
  F out(21);
  std::vector<F> instance_column = {std::move(a), std::move(b), std::move(out)};
  std::vector<Evals> instance_columns = {Evals(std::move(instance_column))};
  std::vector<std::vector<Evals>> instance_columns_vec = {
      instance_columns, std::move(instance_columns)};

  Proof<F, Commitment> proof;
  F h_eval;
  ASSERT_TRUE(verifier.VerifyProofForTesting(vkey, instance_columns_vec, &proof,
                                             &h_eval));

  std::vector<std::vector<Commitment>> expected_advice_commitments_vec;
  {
    std::vector<Point> points = {
        {"0x1ff5354a0976453b28608845fba99fc094fddf490129383d4fad68479842ce1c",
         "0x1cd0c20f3fa582fe5370f1b25c8c232cb49cccdd49e4055ae3e86c81cce14cba"},
        {"0x0012c278617f16ed4f5ea375f5867d5be75bc5fa0852db43348f561795daa89c",
         "0x1789d49845e902ce9de8449b57f8db1be2d8fe107ae48122036e711690ffbcf3"},
        {"0x0ec1d574e43d42faf48e77323f753b904a2b5437109eea2df663bad9c74e0381",
         "0x0eff45bd58fb821a22e59cc0ce9316c8a210399539ac47c5e4c6e5a6e46fc8e1"},
    };
    expected_advice_commitments_vec.push_back(CreateCommitments(points));

    points = {
        {"0x1ff5354a0976453b28608845fba99fc094fddf490129383d4fad68479842ce1c",
         "0x1cd0c20f3fa582fe5370f1b25c8c232cb49cccdd49e4055ae3e86c81cce14cba"},
        {"0x0012c278617f16ed4f5ea375f5867d5be75bc5fa0852db43348f561795daa89c",
         "0x1789d49845e902ce9de8449b57f8db1be2d8fe107ae48122036e711690ffbcf3"},
        {"0x0ec1d574e43d42faf48e77323f753b904a2b5437109eea2df663bad9c74e0381",
         "0x0eff45bd58fb821a22e59cc0ce9316c8a210399539ac47c5e4c6e5a6e46fc8e1"},
    };
    expected_advice_commitments_vec.push_back(CreateCommitments(points));
  }
  EXPECT_EQ(proof.advices_commitments_vec, expected_advice_commitments_vec);

  EXPECT_TRUE(proof.challenges.empty());

  F expected_theta = F::FromHexString(
      "0x07305b7429560970f9ea5dc228a32cbbd28214ecec2d48358f8cd2549e25c871");
  EXPECT_EQ(proof.theta, expected_theta);

  std::vector<LookupPairs<Commitment>> expected_lookup_permuted_commitments_vec;
  {
    std::vector<Point> input_points = {
        {"0x074a21a624d487cb87f9be140cbfd5f50164e1b3834251646631cecbdc9a837f",
         "0x03c0831ebd9006fafe37789974fb493f62405643ebfc096b098af30a11c9d6d7"},
    };
    std::vector<Point> table_points = {
        {"0x1b1b2537df4e03eb51d7d96a4ef3163150c5dc67fbf2d1b6fe7ae2a3db4c3605",
         "0x277d23785c91aa865f99c5a2fcc8f521dfd3a41a22fd09b24f5d1543232f08c0"},
    };
    expected_lookup_permuted_commitments_vec.push_back(
        CreateLookupPermutedCommitments(input_points, table_points));

    input_points = {
        {"0x1e21a8e3435c81705a2689934087e0fd254326dd98487b2b5c3de27b91317a1c",
         "0x10378dbcb27185fb5aaaf7b3abd538804fc2e183ca9885d50d9845181a2b17d8"},
    };
    table_points = {
        {"0x25a2261a08705daa6f287cb7fcf995305832a7b8bc7e6b00f39e737b5c5aaa19",
         "0x149a7e7a2b9869bce102999818771ccf99887e9a70b9e1b6901866023a758c3f"},
    };
    expected_lookup_permuted_commitments_vec.push_back(
        CreateLookupPermutedCommitments(input_points, table_points));
  }
  EXPECT_EQ(proof.lookup_permuted_commitments_vec,
            expected_lookup_permuted_commitments_vec);

  F expected_beta = F::FromHexString(
      "0x25b2496d32f58f7038c10819af46c783ed0ee2a8da1e0fe57f0554252c96340f");
  EXPECT_EQ(proof.beta, expected_beta);

  F expected_gamma = F::FromHexString(
      "0x14a438f67727701291871317f2c22a9b28f8e6091b40a5e4a5c9c1e60b9accf8");
  EXPECT_EQ(proof.gamma, expected_gamma);

  std::vector<std::vector<Commitment>>
      expected_permutation_product_commitments_vec;
  {
    std::vector<Point> points = {
        {"0x06f6df522b948a10ee91739d99ee1c7256344177f74bd3a93dd86c318bd155d6",
         "0x28ac08a6e857c0c006e3b68f6e6c7bf78abbc3da2015d3c2eb0269fbaf8e783d"},
        {"0x2bd66857e4cef1a049cf4a5559bce275045c7c7a8d66985076a31323c6057914",
         "0x25d01d039adc46fd77f386ec430773828867b990d33f24eacbebd431e0bf1c56"},
    };
    expected_permutation_product_commitments_vec.push_back(
        CreateCommitments(points));

    points = {
        {"0x10f8cf0256e9982458ea5267bea2af1b7d2940855a65efc69a89ac6b9b626678",
         "0x02104de0aa270a161adf99f37b6079bdf1151531e70c776291052796b878214b"},
        {"0x0f502935663e393e335d27a118ab3d483959384e5e510fd30575c0492d428bbf",
         "0x030a1739da444dd8ee212fab5c0b83adab41bafd8c28787bf11a5133e191f1ae"},
    };
    expected_permutation_product_commitments_vec.push_back(
        CreateCommitments(points));
  }
  EXPECT_EQ(proof.permutation_product_commitments_vec,
            expected_permutation_product_commitments_vec);

  std::vector<std::vector<Commitment>> expected_lookup_product_commitments_vec;
  {
    std::vector<Point> points = {
        {"0x1f7148d0254edbada9857d824cd040c223f4d202fc5ed90eb03313c9e1b25503",
         "0x3047b38f6f7c3199518790bb9887d8f7173ad54a3d6ffaffb28666055012fcd3"},
    };
    expected_lookup_product_commitments_vec.push_back(
        CreateCommitments(points));

    points = {
        {"0x17c443d0a37c3bea40eeb21342f2bd9cbe17b9753e1e8ebe88bdab87bb20dff0",
         "0x11c434536f9eccbc4095309d05855b743ae763049d9d96e1504fd7d661435fdf"},
    };
    expected_lookup_product_commitments_vec.push_back(
        CreateCommitments(points));
  }
  EXPECT_EQ(proof.lookup_product_commitments_vec,
            expected_lookup_product_commitments_vec);

  Commitment expected_vanishing_random_poly_commitment;
  {
    expected_vanishing_random_poly_commitment = CreateCommitment(
        {"0x0000000000000000000000000000000000000000000000000000000000000001",
         "0x0000000000000000000000000000000000000000000000000000000000000002"});
  }
  EXPECT_EQ(proof.vanishing_random_poly_commitment,
            expected_vanishing_random_poly_commitment);

  F expected_y = F::FromHexString(
      "0x2d504abb62d15e24bcfdceb45dd19b590555844a0d274eb3c8750a0639f9cf1a");
  EXPECT_EQ(proof.y, expected_y);

  std::vector<Commitment> expected_vanishing_h_poly_commitments;
  {
    std::vector<Point> points = {
        {"0x271a470d613c69e29ff888ee1ec7b53ffb5544d5a53cbe19a397768382d67e23",
         "0x29dadfd27f2a056957ea9ba683043fd4c1f647adb5d1428ca998cda8db73b2d5"},
        {"0x243947364f30dbc593d8feee68d6f761612d8e3c2ec52fb90de9376564a5708c",
         "0x287d037e2d10bcdfa356091078b2c8217ca66e7244f25d2d2b76a5ec28179b35"},
        {"0x2c92a8244233330881f1d897466363da649e88502decfa24dad465d81ce5629a",
         "0x0647ca052c9d2a2872721db8671014fda2b6e8207ed3cce90dce9e51f9b435de"},
        {"0x241656a160172d39b23143afac968f1c0463e3a3ee9f4798df6fdd0681944167",
         "0x0acb9830613e912ce41be5a65d9a79bb44ec261064c1ba6b8dadb5bd0b68c805"},
    };
    expected_vanishing_h_poly_commitments = CreateCommitments(points);
  }
  EXPECT_EQ(proof.vanishing_h_poly_commitments,
            expected_vanishing_h_poly_commitments);

  F expected_x = F::FromHexString(
      "0x2cdf546a9633ac7ded289492c97a5b24067e1d5898c9e9a2761900f770558809");
  EXPECT_EQ(proof.x, expected_x);

  std::vector<std::vector<F>> expected_advice_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x295364357b717f95bf540b583906a006b9e092b40acb385bdca5ef5b3000f995",
        "0x120c1ce5c6dc93d59a0dfc8f1c953b67132660aeb33ae3d3ef5c62c26bc10568",
        "0x2ab78830433ae0de6f3f5ecaeee772850b008d92fd3bdd079d14604804f4b344",
    };
    expected_advice_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x295364357b717f95bf540b583906a006b9e092b40acb385bdca5ef5b3000f995",
        "0x120c1ce5c6dc93d59a0dfc8f1c953b67132660aeb33ae3d3ef5c62c26bc10568",
        "0x2ab78830433ae0de6f3f5ecaeee772850b008d92fd3bdd079d14604804f4b344",
    };
    expected_advice_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.advice_evals_vec, expected_advice_evals_vec);

  std::vector<F> expected_fixed_evals;
  {
    std::vector<std::string_view> evals = {
        "0x1b73400828389c7b2a6afd13ee343bd120b8794556a06b12a5ff1ce34323fb34",
        "0x034d459a504505fa2af105e7c03c8ccef44576b4fa544c81e5b66b596d244c79",
        "0x11b8758e9a61d457a9ecbcef8f3f303964950fb89b8263a82c2bb86b1dd972db",
        "0x07d806affc2f3336d5b68995cc6bf67cd3cb64e20b2a9047ac966bfa7cbe1432",
        "0x2164646e0fd09aa38244926cf94b2c8ec490f6f918eed00d3aecd40798acb2a8",
    };
    expected_fixed_evals = CreateEvals(evals);
  }
  EXPECT_EQ(proof.fixed_evals, expected_fixed_evals);

  F expected_vanishing_random_eval = F::FromHexString(
      "0x0000000000000000000000000000000000000000000000000000000000000001");
  EXPECT_EQ(proof.vanishing_random_eval, expected_vanishing_random_eval);

  std::vector<F> expected_common_permutation_evals;
  {
    std::vector<std::string_view> evals = {
        "0x1360dc9c7a30e6f1bb390933bdc3d32a083e3dcb45a6cb870070efc7dcd66699",
        "0x1d469ae55f0954b3ea99d180ebdca12695eda7c4ba5f61aa61fe6409db2180a7",
        "0x00f671ecba12fc4d9129d9affa8dbff1dbf5f649bb6240a494cfc7391ad1e7a4",
        "0x060ea8f03a2ead18d5d7e8e2a08c59e67cd9ecc7759550c3fae6b7ba9cdfea3c",
    };
    expected_common_permutation_evals = CreateEvals(evals);
  }
  EXPECT_EQ(proof.common_permutation_evals, expected_common_permutation_evals);

  std::vector<std::vector<F>> expected_permutation_product_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x24488f42350023743fdd269f73c77c176b62a747d5bbafb223fce1ec07829730",
        "0x1d4aede84858db8a189688c5dc53661d08978175e93bd9c004e0c8384372f493",
    };
    expected_permutation_product_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x09b6a9e106a89f46b8e160d4465bbc88bd901cbe05001d09c0fbf69e0c00d497",
        "0x165fd66b962a0e5e8e03f1aea41dfbdb36641bdb844305322801d1f4c03b875a",
    };
    expected_permutation_product_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_evals_vec,
            expected_permutation_product_evals_vec);

  std::vector<std::vector<F>> expected_permutation_product_next_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x1d9909c3178b5378ba9df90952d42fa439512722fc34c2ca5944f8bb48e2c195",
        "0x07434f5c5a8e5ec012e9648066bf1d3a1d03b3634599c935680f1d8311c06de1",
    };
    expected_permutation_product_next_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x0009d1abb9539cc4a43814f248b70106188e39ddaeda92a2cbb2c9b29ca1cddd",
        "0x03f9a997efc0ba820bf54b4a9d8713077da3fecafb3c16f5e318734d5be00b2b",
    };
    expected_permutation_product_next_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_next_evals_vec,
            expected_permutation_product_next_evals_vec);

  std::vector<std::vector<std::optional<F>>>
      expected_permutation_product_last_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x1ea0c4a3ccbec4f0bd0804bdda295e3160f84d86a531a2d534ae59127922613e",
        "",
    };
    expected_permutation_product_last_evals_vec.push_back(
        CreateOptionalEvals(evals));

    evals = {
        "0x276b2718502b192876f73b08d2728186274827af950d0ce0b38a979745ef3a33",
        "",
    };
    expected_permutation_product_last_evals_vec.push_back(
        CreateOptionalEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_last_evals_vec,
            expected_permutation_product_last_evals_vec);

  std::vector<std::vector<F>> expected_lookup_product_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x1cddd43db1a11f2fceb294d030c947cd3a515b0b4695065ff4efa47eae6cfbe0",
    };
    expected_lookup_product_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x0621132b8077350a400179a4a315d1f4c8b5b1806297c26f6b0f0bb04f070666",
    };
    expected_lookup_product_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.lookup_product_evals_vec, expected_lookup_product_evals_vec);

  std::vector<std::vector<F>> expected_lookup_product_next_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x10dc3180c93171626e62ddf41d2b19654157c44ae7e8dcd749c4e21364a8afbe",
    };
    expected_lookup_product_next_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x2f00965d82aba1968f1534ccd78acbc19863ebec7704cff24b30a35dcc6cab9e",
    };
    expected_lookup_product_next_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.lookup_product_next_evals_vec,
            expected_lookup_product_next_evals_vec);

  std::vector<std::vector<F>> expected_lookup_permuted_input_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x109976c6bb5120cac9de8ece29b5e259643e3c50e6db604544c4e3bffe7e3b7d",
    };
    expected_lookup_permuted_input_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x19cd0ef47d7ec8abb2691cda73a52ecc220bc15b9116314f262e0e9538971dc4",
    };
    expected_lookup_permuted_input_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.lookup_permuted_input_evals_vec,
            expected_lookup_permuted_input_evals_vec);

  std::vector<std::vector<F>> expected_lookup_permuted_input_inv_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x01c096f1e653a9a736c65b77eb51b88aab75aa2265eca4506a63ad4b34ebc9fe",
    };
    expected_lookup_permuted_input_inv_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x24367b3acf595c94e93fabbe410acbbf50afca00f12cb0063609ff9adf683a51",
    };
    expected_lookup_permuted_input_inv_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.lookup_permuted_input_inv_evals_vec,
            expected_lookup_permuted_input_inv_evals_vec);

  std::vector<std::vector<F>> expected_lookup_permuted_table_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x16b54aff10ae3c2b3c78fefed86f39d84965ac28cc7662122fda8662d840b97b",
    };
    expected_lookup_permuted_table_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x04e92861d3ed75f7cf69de47edb43e866fd5b7576cd58e8758142a3dae558184",
    };
    expected_lookup_permuted_table_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.lookup_permuted_table_evals_vec,
            expected_lookup_permuted_table_evals_vec);
}

}  // namespace tachyon::zk::plonk::halo2
