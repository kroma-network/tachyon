pragma circom 2.1.0;
include "./constants.circom";
include "./poseidon.circom";
include "./utils.circom";
include "./goldilocks.circom";
include "./goldilocks_ext.circom";

template GetMerkleProofToCap(nLeaf, nProof) {
  signal input leaf[nLeaf];
  signal input proof[nProof][4];
  signal input leaf_index;
  signal output digest[4];
  signal output index;

  component c_digest = HashNoPad_BN(nLeaf, 4);
  for (var i = 0; i < nLeaf; i++) {
      c_digest.in[i] <== leaf[i];
  }
  for (var i = 0; i < 4; i++) {
      c_digest.capacity[i] <== 0;
  }

  component poseidon[nProof];
  component shift[nProof];
  signal cur_digest[nProof + 1][4];

  shift[0] = RShift1();
  shift[0].x <== leaf_index;
  for (var i = 0; i < 4; i++) {
    cur_digest[0][i] <== c_digest.out[i];
  }

  signal in0[nProof][4];
  signal in1[nProof][4];
  for (var i = 0; i < nProof; i++) {
    poseidon[i] = Poseidon_BN(4);

    for (var j = 0; j < 4; j++) {
      in0[i][j] <== (1 - shift[i].bit) * cur_digest[i][j];
      poseidon[i].in[j] <== in0[i][j] + shift[i].bit * proof[i][j];
    }
    for (var j = 0; j < 4; j++) {
      in1[i][j] <== (1 - shift[i].bit) * proof[i][j];
      poseidon[i].in[j + 4] <== in1[i][j] + shift[i].bit * cur_digest[i][j];
    }

    for (var j = 0; j < 4; j++) {
      poseidon[i].capacity[j] <== 0;
    }

    for (var j = 0; j < 4; j++) {
      cur_digest[i + 1][j] <== poseidon[i].out[j];
    }

    if (i < nProof - 1) {
      shift[i + 1] = RShift1();
      shift[i + 1].x <== shift[i].out;
    }
  }

  for (var i = 0; i < 4; i++) {
    digest[i] <== cur_digest[nProof][i];
  }
  index <== shift[nProof - 1].out;
}

template CalBarycentricWeights(arity) {
  signal input points[arity];
  signal output out[arity];
  assert(arity > 1);

  component sub[arity][arity - 1];
  var index;

  for (var i = 0; i < arity; i++) {
    index = 0;
    for (var j = 0; j < arity; j++) {
      if (i != j) {
        sub[i][index] = GlSub();
        sub[i][index].a <== points[i];
        sub[i][index].b <== points[j];
        index++;
      }
    }
  }

  component mul[arity][arity - 2];
  component inv[arity];
  for (var i = 0; i < arity; i++) {
    for (var j = 0; j < arity - 2; j++) {
      mul[i][j] = GlMul();
      if (j == 0) mul[i][j].a <== sub[i][0].out;
      else mul[i][j].a <== mul[i][j - 1].out;
      mul[i][j].b <== sub[i][j + 1].out;
    }
    inv[i] = GlInv();
    inv[i].x <== mul[i][arity - 3].out;
  }

  for (var i = 0; i < arity; i++) {
    out[i] <== inv[i].out;
  }
}

template VerifyFriProof() {
  signal input wires_cap[NUM_WIRES_CAP()][4];
  signal input plonk_zs_partial_products_cap[NUM_PLONK_ZS_PARTIAL_PRODUCTS_CAP()][4];
  signal input quotient_polys_cap[NUM_QUOTIENT_POLYS_CAP()][4];

  signal input openings_constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input openings_plonk_sigmas[NUM_OPENINGS_PLONK_SIGMAS()][2];
  signal input openings_wires[NUM_OPENINGS_WIRES()][2];
  signal input openings_plonk_zs[NUM_OPENINGS_PLONK_ZS()][2];
  signal input openings_plonk_zs_next[NUM_OPENINGS_PLONK_ZS_NEXT()][2];
  signal input openings_partial_products[NUM_OPENINGS_PARTIAL_PRODUCTS()][2];
  signal input openings_quotient_polys[NUM_OPENINGS_QUOTIENT_POLYS()][2];
  signal input fri_constants_sigmas_cap[NUM_SIGMA_CAPS()][4];


  signal input fri_commit_phase_merkle_caps[NUM_FRI_COMMIT_ROUND()][FRI_COMMIT_MERKLE_CAP_HEIGHT()][4];
  signal input fri_query_init_constants_sigmas_v[NUM_FRI_QUERY_ROUND()][NUM_FRI_QUERY_INIT_CONSTANTS_SIGMAS_V()];
  signal input fri_query_init_constants_sigmas_p[NUM_FRI_QUERY_ROUND()][NUM_FRI_QUERY_INIT_CONSTANTS_SIGMAS_P()][4];
  signal input fri_query_init_wires_v[NUM_FRI_QUERY_ROUND()][NUM_FRI_QUERY_INIT_WIRES_V()];
  signal input fri_query_init_wires_p[NUM_FRI_QUERY_ROUND()][NUM_FRI_QUERY_INIT_WIRES_P()][4];
  signal input fri_query_init_zs_partial_v[NUM_FRI_QUERY_ROUND()][NUM_FRI_QUERY_INIT_ZS_PARTIAL_V()];
  signal input fri_query_init_zs_partial_p[NUM_FRI_QUERY_ROUND()][NUM_FRI_QUERY_INIT_ZS_PARTIAL_P()][4];
  signal input fri_query_init_quotient_v[NUM_FRI_QUERY_ROUND()][NUM_FRI_QUERY_INIT_QUOTIENT_V()];
  signal input fri_query_init_quotient_p[NUM_FRI_QUERY_ROUND()][NUM_FRI_QUERY_INIT_QUOTIENT_P()][4];
  signal input fri_query_step0_v[NUM_FRI_QUERY_ROUND()][NUM_FRI_QUERY_STEP0_V()][2];
  signal input fri_query_step0_p[NUM_FRI_QUERY_ROUND()][NUM_FRI_QUERY_STEP0_P()][4];
  signal input fri_query_step1_v[NUM_FRI_QUERY_ROUND()][NUM_FRI_QUERY_STEP1_V()][2];
  signal input fri_query_step1_p[NUM_FRI_QUERY_ROUND()][NUM_FRI_QUERY_STEP1_P()][4];
  signal input fri_final_poly_ext_v[NUM_FRI_FINAL_POLY_EXT_V()][2];

  // Challenges
  signal input plonk_zeta[2];
  signal input fri_alpha[2];
  signal input fri_betas[NUM_FRI_COMMIT_ROUND()][2];
  signal input fri_pow_response;
  signal input fri_query_indices[NUM_FRI_QUERY_ROUND()];

  // fri_verify_proof_of_work
  component check = LessNBits(64 - MIN_FRI_POW_RESPONSE());
  check.x <== fri_pow_response;

  component c_gl_mul[NUM_FRI_QUERY_ROUND()][1];
  component c_gl_exp[NUM_FRI_QUERY_ROUND()][1];
  component c_mul[NUM_FRI_QUERY_ROUND()][1];
  component c_exp[NUM_FRI_QUERY_ROUND()][1];
  component c_add[NUM_FRI_QUERY_ROUND()][1];
  component c_sub[NUM_FRI_QUERY_ROUND()][4];
  component c_div[NUM_FRI_QUERY_ROUND()][2];
  component c_reverse_bits[NUM_FRI_QUERY_ROUND()][1];
  component c_reduce[NUM_FRI_QUERY_ROUND()][5];
  component c_random_access[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()];

  signal precomputed_reduced_evals[2][2];
  component reduce[7];
  reduce[5] = Reduce(NUM_OPENINGS_CONSTANTS());
  reduce[4] = Reduce(NUM_OPENINGS_PLONK_SIGMAS());
  reduce[3] = Reduce(NUM_OPENINGS_WIRES());
  reduce[2] = Reduce(NUM_OPENINGS_PLONK_ZS());
  reduce[1] = Reduce(NUM_OPENINGS_PARTIAL_PRODUCTS());
  reduce[0] = Reduce(NUM_OPENINGS_QUOTIENT_POLYS());
  reduce[6] = Reduce(NUM_OPENINGS_PLONK_ZS_NEXT());

  for (var i = 0; i < 7; i++) {
    reduce[i].alpha[0] <== fri_alpha[0];
    reduce[i].alpha[1] <== fri_alpha[1];
  }

  reduce[0].old_eval[0] <== 0;
  reduce[0].old_eval[1] <== 0;
  reduce[6].old_eval[0] <== 0;
  reduce[6].old_eval[1] <== 0;
  for (var i = 0; i < NUM_OPENINGS_QUOTIENT_POLYS(); i++) {
    reduce[0].in[i][0] <== openings_quotient_polys[i][0];
    reduce[0].in[i][1] <== openings_quotient_polys[i][1];
  }
  for (var i = 0; i < NUM_OPENINGS_PARTIAL_PRODUCTS(); i++) {
    reduce[1].in[i][0] <== openings_partial_products[i][0];
    reduce[1].in[i][1] <== openings_partial_products[i][1];
  }
  for (var i = 0; i < NUM_OPENINGS_PLONK_ZS(); i++) {
    reduce[2].in[i][0] <== openings_plonk_zs[i][0];
    reduce[2].in[i][1] <== openings_plonk_zs[i][1];
  }
  for (var i = 0; i < NUM_OPENINGS_WIRES(); i++) {
    reduce[3].in[i][0] <== openings_wires[i][0];
    reduce[3].in[i][1] <== openings_wires[i][1];
  }
  for (var i = 0; i < NUM_OPENINGS_PLONK_SIGMAS(); i++) {
    reduce[4].in[i][0] <== openings_plonk_sigmas[i][0];
    reduce[4].in[i][1] <== openings_plonk_sigmas[i][1];
  }
  for (var i = 0; i < NUM_OPENINGS_CONSTANTS(); i++) {
    reduce[5].in[i][0] <== openings_constants[i][0];
    reduce[5].in[i][1] <== openings_constants[i][1];
  }
  for (var i = 0; i < NUM_OPENINGS_PLONK_ZS_NEXT(); i++) {
    reduce[6].in[i][0] <== openings_plonk_zs_next[i][0];
    reduce[6].in[i][1] <== openings_plonk_zs_next[i][1];
  }

  for (var i = 1; i < 6; i++) {
    reduce[i].old_eval[0] <== reduce[i - 1].out[0];
    reduce[i].old_eval[1] <== reduce[i - 1].out[1];
  }
  precomputed_reduced_evals[0][0] <== reduce[5].out[0];
  precomputed_reduced_evals[0][1] <== reduce[5].out[1];
  precomputed_reduced_evals[1][0] <== reduce[6].out[0];
  precomputed_reduced_evals[1][1] <== reduce[6].out[1];

  component zeta_next = GlExtMul();
  var g[2] = G_FROM_DEGREE_BITS();
  zeta_next.a[0] <== g[0];
  zeta_next.a[1] <== g[1];
  zeta_next.b[0] <== plonk_zeta[0];
  zeta_next.b[1] <== plonk_zeta[1];

  assert(NUM_REDUCTION_ARITY_BITS() == 2);
  var arity_bits[NUM_REDUCTION_ARITY_BITS()] = REDUCTION_ARITY_BITS();
  var max_arity = 0;
  for (var i = 0; i < NUM_REDUCTION_ARITY_BITS(); i++) {
    max_arity = max_arity < (1 << arity_bits[i]) ? (1 << arity_bits[i]) : max_arity;
  }

  signal subgroup_x[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS() + 1][2];
  signal old_eval[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS() + 1][2];
  signal points[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()][max_arity];
  signal l_x[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()][max_arity][2];

  component coset_index[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()];
  component x_index_within_coset[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()];
  component rev_x_index_within_coset[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()];
  component barycentric_weights[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()];

  component p_mul[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()][max_arity];
  component p_exp[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()];
  component p_exp2[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()];
  component e_sub0[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()];
  component e_mul0[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()];
  component e_sub1[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()][max_arity - 1];
  component e_mul1[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()][max_arity - 1];
  component e_add2[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()][max_arity];
  component e_sub2[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()][max_arity];
  component e_div2[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()][max_arity];
  component e_mul2[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()][max_arity];
  component e_rev[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()][max_arity];
  component e_ra[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()][max_arity];

  component sigma_caps[NUM_FRI_QUERY_ROUND()];
  component merkle_caps[NUM_FRI_QUERY_ROUND()][6];
  component c_wires_cap[NUM_FRI_QUERY_ROUND()];
  component c_plonk_zs_partial_products_cap[NUM_FRI_QUERY_ROUND()];
  component c_quotient_polys_cap[NUM_FRI_QUERY_ROUND()];
  component c_commit_merkle_cap[NUM_FRI_QUERY_ROUND()][NUM_REDUCTION_ARITY_BITS()];
  component c_final_eval_mul[NUM_FRI_QUERY_ROUND()][NUM_FRI_FINAL_POLY_EXT_V()];
  component c_final_eval_add[NUM_FRI_QUERY_ROUND()][NUM_FRI_FINAL_POLY_EXT_V()];

  for (var round = 0; round < NUM_FRI_QUERY_ROUND(); round++) {
    // constants_sigmas
    merkle_caps[round][0] = GetMerkleProofToCap(NUM_FRI_QUERY_INIT_CONSTANTS_SIGMAS_V(),
                                                NUM_FRI_QUERY_INIT_CONSTANTS_SIGMAS_P());
    merkle_caps[round][0].leaf_index <== fri_query_indices[round];
    for (var i = 0; i < NUM_FRI_QUERY_INIT_CONSTANTS_SIGMAS_V(); i++) {
      merkle_caps[round][0].leaf[i] <== fri_query_init_constants_sigmas_v[round][i];
    }
    for (var i = 0; i < NUM_FRI_QUERY_INIT_CONSTANTS_SIGMAS_P(); i++) {
      merkle_caps[round][0].proof[i][0] <== fri_query_init_constants_sigmas_p[round][i][0];
      merkle_caps[round][0].proof[i][1] <== fri_query_init_constants_sigmas_p[round][i][1];
      merkle_caps[round][0].proof[i][2] <== fri_query_init_constants_sigmas_p[round][i][2];
      merkle_caps[round][0].proof[i][3] <== fri_query_init_constants_sigmas_p[round][i][3];
    }
    sigma_caps[round] = RandomAccess2(NUM_SIGMA_CAPS(), 4);
    for (var i = 0; i < NUM_SIGMA_CAPS(); i++) {
      sigma_caps[round].a[i][0] <== fri_constants_sigmas_cap[i][0];
      sigma_caps[round].a[i][1] <== fri_constants_sigmas_cap[i][1];
      sigma_caps[round].a[i][2] <== fri_constants_sigmas_cap[i][2];
      sigma_caps[round].a[i][3] <== fri_constants_sigmas_cap[i][3];
    }
    sigma_caps[round].idx <== merkle_caps[round][0].index;
    merkle_caps[round][0].digest[0] === sigma_caps[round].out[0];
    merkle_caps[round][0].digest[1] === sigma_caps[round].out[1];
    merkle_caps[round][0].digest[2] === sigma_caps[round].out[2];
    merkle_caps[round][0].digest[3] === sigma_caps[round].out[3];

    // wires
    merkle_caps[round][1] = GetMerkleProofToCap(NUM_FRI_QUERY_INIT_WIRES_V(),
                                                NUM_FRI_QUERY_INIT_WIRES_P());
    merkle_caps[round][1].leaf_index <== fri_query_indices[round];
    for (var i = 0; i < NUM_FRI_QUERY_INIT_WIRES_V(); i++) {
      merkle_caps[round][1].leaf[i] <== fri_query_init_wires_v[round][i];
    }
    for (var i = 0; i < NUM_FRI_QUERY_INIT_WIRES_P(); i++) {
      merkle_caps[round][1].proof[i][0] <== fri_query_init_wires_p[round][i][0];
      merkle_caps[round][1].proof[i][1] <== fri_query_init_wires_p[round][i][1];
      merkle_caps[round][1].proof[i][2] <== fri_query_init_wires_p[round][i][2];
      merkle_caps[round][1].proof[i][3] <== fri_query_init_wires_p[round][i][3];
    }
    c_wires_cap[round] = RandomAccess2(NUM_WIRES_CAP(), 4);
    for (var i = 0; i < NUM_WIRES_CAP(); i++) {
      c_wires_cap[round].a[i][0] <== wires_cap[i][0];
      c_wires_cap[round].a[i][1] <== wires_cap[i][1];
      c_wires_cap[round].a[i][2] <== wires_cap[i][2];
      c_wires_cap[round].a[i][3] <== wires_cap[i][3];
    }
    c_wires_cap[round].idx <== merkle_caps[round][1].index;
    merkle_caps[round][1].digest[0] === c_wires_cap[round].out[0];
    merkle_caps[round][1].digest[1] === c_wires_cap[round].out[1];
    merkle_caps[round][1].digest[2] === c_wires_cap[round].out[2];
    merkle_caps[round][1].digest[3] === c_wires_cap[round].out[3];

    // plonk_zs_partial_products
    merkle_caps[round][2] = GetMerkleProofToCap(NUM_FRI_QUERY_INIT_ZS_PARTIAL_V(),
                                                NUM_FRI_QUERY_INIT_ZS_PARTIAL_P());
    merkle_caps[round][2].leaf_index <== fri_query_indices[round];
    for (var i = 0; i < NUM_FRI_QUERY_INIT_ZS_PARTIAL_V(); i++) {
      merkle_caps[round][2].leaf[i] <== fri_query_init_zs_partial_v[round][i];
    }
    for (var i = 0; i < NUM_FRI_QUERY_INIT_ZS_PARTIAL_P(); i++) {
      merkle_caps[round][2].proof[i][0] <== fri_query_init_zs_partial_p[round][i][0];
      merkle_caps[round][2].proof[i][1] <== fri_query_init_zs_partial_p[round][i][1];
      merkle_caps[round][2].proof[i][2] <== fri_query_init_zs_partial_p[round][i][2];
      merkle_caps[round][2].proof[i][3] <== fri_query_init_zs_partial_p[round][i][3];
    }
    c_plonk_zs_partial_products_cap[round] = RandomAccess2(NUM_PLONK_ZS_PARTIAL_PRODUCTS_CAP(), 4);
    for (var i = 0; i < NUM_PLONK_ZS_PARTIAL_PRODUCTS_CAP(); i++) {
      c_plonk_zs_partial_products_cap[round].a[i][0] <== plonk_zs_partial_products_cap[i][0];
      c_plonk_zs_partial_products_cap[round].a[i][1] <== plonk_zs_partial_products_cap[i][1];
      c_plonk_zs_partial_products_cap[round].a[i][2] <== plonk_zs_partial_products_cap[i][2];
      c_plonk_zs_partial_products_cap[round].a[i][3] <== plonk_zs_partial_products_cap[i][3];
    }
    c_plonk_zs_partial_products_cap[round].idx <== merkle_caps[round][2].index;
    merkle_caps[round][2].digest[0] === c_plonk_zs_partial_products_cap[round].out[0];
    merkle_caps[round][2].digest[1] === c_plonk_zs_partial_products_cap[round].out[1];
    merkle_caps[round][2].digest[2] === c_plonk_zs_partial_products_cap[round].out[2];
    merkle_caps[round][2].digest[3] === c_plonk_zs_partial_products_cap[round].out[3];

    // quotient
    merkle_caps[round][3] = GetMerkleProofToCap(NUM_FRI_QUERY_INIT_QUOTIENT_V(),
                                                NUM_FRI_QUERY_INIT_QUOTIENT_P());
    merkle_caps[round][3].leaf_index <== fri_query_indices[round];
    for (var i = 0; i < NUM_FRI_QUERY_INIT_QUOTIENT_V(); i++) {
      merkle_caps[round][3].leaf[i] <== fri_query_init_quotient_v[round][i];
    }
    for (var i = 0; i < NUM_FRI_QUERY_INIT_QUOTIENT_P(); i++) {
      merkle_caps[round][3].proof[i][0] <== fri_query_init_quotient_p[round][i][0];
      merkle_caps[round][3].proof[i][1] <== fri_query_init_quotient_p[round][i][1];
      merkle_caps[round][3].proof[i][2] <== fri_query_init_quotient_p[round][i][2];
      merkle_caps[round][3].proof[i][3] <== fri_query_init_quotient_p[round][i][3];
    }
    c_quotient_polys_cap[round] = RandomAccess2(NUM_QUOTIENT_POLYS_CAP(), 4);
    for (var i = 0; i < NUM_QUOTIENT_POLYS_CAP(); i++) {
      c_quotient_polys_cap[round].a[i][0] <== quotient_polys_cap[i][0];
      c_quotient_polys_cap[round].a[i][1] <== quotient_polys_cap[i][1];
      c_quotient_polys_cap[round].a[i][2] <== quotient_polys_cap[i][2];
      c_quotient_polys_cap[round].a[i][3] <== quotient_polys_cap[i][3];
    }
    c_quotient_polys_cap[round].idx <== merkle_caps[round][3].index;
    merkle_caps[round][3].digest[0] === c_quotient_polys_cap[round].out[0];
    merkle_caps[round][3].digest[1] === c_quotient_polys_cap[round].out[1];
    merkle_caps[round][3].digest[2] === c_quotient_polys_cap[round].out[2];
    merkle_caps[round][3].digest[3] === c_quotient_polys_cap[round].out[3];


    c_reverse_bits[round][0] = ReverseBits(LOG_SIZE_OF_LDE_DOMAIN());
    c_reverse_bits[round][0].x <== fri_query_indices[round];
    c_gl_exp[round][0] = GlExp();
    c_gl_exp[round][0].x <== PRIMITIVE_ROOT_OF_UNITY_LDE();
    c_gl_exp[round][0].n <== c_reverse_bits[round][0].out;
    c_gl_mul[round][0] = GlMul();
    c_gl_mul[round][0].a <== MULTIPLICATIVE_GROUP_GENERATOR();
    c_gl_mul[round][0].b <== c_gl_exp[round][0].out;
    subgroup_x[round][0][0] <== c_gl_mul[round][0].out;
    subgroup_x[round][0][1] <== 0;

//    c_exp[round][1] = GlExtExp();
//    c_exp[round][1].x[0] <== fri_alpha[0];
//    c_exp[round][1].x[1] <== fri_alpha[1];
//    c_exp[round][1].n <== NUM_FRI_QUERY_INIT_CONSTANTS_SIGMAS_V() + NUM_FRI_QUERY_INIT_WIRES_V() +
//                          NUM_FRI_QUERY_INIT_ZS_PARTIAL_V() + NUM_FRI_QUERY_INIT_ZS_PARTIAL_V();
//    c_mul[round][1] = GlExtMul();
//    c_mul[round][1].a[0] <== c_exp[round][1].out[0];
//    c_mul[round][1].a[1] <== c_exp[round][1].out[1];
//    c_mul[round][1].b[0] <== 0;  TODO: bug in Plonky2?
//    c_mul[round][1].b[1] <== 0;

    c_reduce[round][0] = Reduce(NUM_FRI_QUERY_INIT_QUOTIENT_V());
    c_reduce[round][1] = Reduce(NUM_FRI_QUERY_INIT_ZS_PARTIAL_V());
    c_reduce[round][2] = Reduce(NUM_FRI_QUERY_INIT_WIRES_V());
    c_reduce[round][3] = Reduce(NUM_FRI_QUERY_INIT_CONSTANTS_SIGMAS_V());
    c_reduce[round][4] = Reduce(NUM_CHALLENGES());
    for (var i = 0; i < 5; i++) {
      c_reduce[round][i].alpha[0] <== fri_alpha[0];
      c_reduce[round][i].alpha[1] <== fri_alpha[1];
    }
    c_reduce[round][0].old_eval[0] <== 0;
    c_reduce[round][0].old_eval[1] <== 0;
    c_reduce[round][4].old_eval[0] <== 0;
    c_reduce[round][4].old_eval[1] <== 0;
    for (var i = 0; i < NUM_FRI_QUERY_INIT_QUOTIENT_V(); i++) {
      c_reduce[round][0].in[i][0] <== fri_query_init_quotient_v[round][i];
      c_reduce[round][0].in[i][1] <== 0;
    }
    for (var i = 0; i < NUM_FRI_QUERY_INIT_ZS_PARTIAL_V(); i++) {
      c_reduce[round][1].in[i][0] <== fri_query_init_zs_partial_v[round][i];
      c_reduce[round][1].in[i][1] <== 0;
    }
    for (var i = 0; i < NUM_FRI_QUERY_INIT_WIRES_V(); i++) {
      c_reduce[round][2].in[i][0] <== fri_query_init_wires_v[round][i];
      c_reduce[round][2].in[i][1] <== 0;
    }
    for (var i = 0; i < NUM_FRI_QUERY_INIT_CONSTANTS_SIGMAS_V(); i++) {
      c_reduce[round][3].in[i][0] <== fri_query_init_constants_sigmas_v[round][i];
      c_reduce[round][3].in[i][1] <== 0;
    }
    for (var i = 0; i < NUM_CHALLENGES(); i++) {
      c_reduce[round][4].in[i][0] <== fri_query_init_zs_partial_v[round][i];
      c_reduce[round][4].in[i][1] <== 0;
    }
    c_reduce[round][1].old_eval[0] <== c_reduce[round][0].out[0];
    c_reduce[round][1].old_eval[1] <== c_reduce[round][0].out[1];
    c_reduce[round][2].old_eval[0] <== c_reduce[round][1].out[0];
    c_reduce[round][2].old_eval[1] <== c_reduce[round][1].out[1];
    c_reduce[round][3].old_eval[0] <== c_reduce[round][2].out[0];
    c_reduce[round][3].old_eval[1] <== c_reduce[round][2].out[1];

    c_sub[round][0] = GlExtSub();
    c_sub[round][0].a[0] <== c_reduce[round][3].out[0];
    c_sub[round][0].a[1] <== c_reduce[round][3].out[1];
    c_sub[round][0].b[0] <== precomputed_reduced_evals[0][0];
    c_sub[round][0].b[1] <== precomputed_reduced_evals[0][1];
    c_sub[round][1] = GlExtSub();
    c_sub[round][1].a[0] <== subgroup_x[round][0][0];
    c_sub[round][1].a[1] <== subgroup_x[round][0][1];
    c_sub[round][1].b[0] <== plonk_zeta[0];
    c_sub[round][1].b[1] <== plonk_zeta[1];
    c_div[round][0] = GlExtDiv();
    c_div[round][0].a[0] <== c_sub[round][0].out[0];
    c_div[round][0].a[1] <== c_sub[round][0].out[1];
    c_div[round][0].b[0] <== c_sub[round][1].out[0];
    c_div[round][0].b[1] <== c_sub[round][1].out[1];

    c_exp[round][0] = GlExtExp();
    c_exp[round][0].x[0] <== fri_alpha[0];
    c_exp[round][0].x[1] <== fri_alpha[1];
    c_exp[round][0].n <== NUM_CHALLENGES();
    c_mul[round][0] = GlExtMul();
    c_mul[round][0].a[0] <== c_exp[round][0].out[0];
    c_mul[round][0].a[1] <== c_exp[round][0].out[1];
    c_mul[round][0].b[0] <== c_div[round][0].out[0];
    c_mul[round][0].b[1] <== c_div[round][0].out[1];

    c_sub[round][2] = GlExtSub();
    c_sub[round][2].a[0] <== c_reduce[round][4].out[0];
    c_sub[round][2].a[1] <== c_reduce[round][4].out[1];
    c_sub[round][2].b[0] <== precomputed_reduced_evals[1][0];
    c_sub[round][2].b[1] <== precomputed_reduced_evals[1][1];
    c_sub[round][3] = GlExtSub();
    c_sub[round][3].a[0] <== subgroup_x[round][0][0];
    c_sub[round][3].a[1] <== subgroup_x[round][0][1];
    c_sub[round][3].b[0] <== zeta_next.out[0];
    c_sub[round][3].b[1] <== zeta_next.out[1];
    c_div[round][1] = GlExtDiv();
    c_div[round][1].a[0] <== c_sub[round][2].out[0];
    c_div[round][1].a[1] <== c_sub[round][2].out[1];
    c_div[round][1].b[0] <== c_sub[round][3].out[0];
    c_div[round][1].b[1] <== c_sub[round][3].out[1];
    c_add[round][0] = GlExtAdd();
    c_add[round][0].a[0] <== c_mul[round][0].out[0];
    c_add[round][0].a[1] <== c_mul[round][0].out[1];
    c_add[round][0].b[0] <== c_div[round][1].out[0];
    c_add[round][0].b[1] <== c_div[round][1].out[1];

    // c_mul[round][1] = GlExtMul();
    // c_mul[round][1].a[0] <== c_add[round][0].out[0];
    // c_mul[round][1].a[1] <== c_add[round][0].out[1];
    // c_mul[round][1].b[0] <== subgroup_x[round][0][0];
    // c_mul[round][1].b[1] <== subgroup_x[round][0][1];

    old_eval[round][0][0] <== c_add[round][0].out[0];
    old_eval[round][0][1] <== c_add[round][0].out[1];

    for (var i = 0; i < NUM_REDUCTION_ARITY_BITS(); i++) {
      var arity = 1 << arity_bits[i];
      coset_index[round][i] = RShift(arity_bits[i]);
      x_index_within_coset[round][i] = LastNBits(arity_bits[i]);
      if (i == 0) {
        coset_index[round][i].x <== fri_query_indices[round];
        x_index_within_coset[round][i].x <== fri_query_indices[round];
      } else {
        coset_index[round][i].x <== coset_index[round][i - 1].out;
        x_index_within_coset[round][i].x <== coset_index[round][i - 1].out;
      }

      if (i == 0) {
        c_random_access[round][i] = RandomAccess2(NUM_FRI_QUERY_STEP0_V(), 2);
        for (var j = 0; j < NUM_FRI_QUERY_STEP0_V(); j++) {
          c_random_access[round][i].a[j][0] <== fri_query_step0_v[round][j][0];
          c_random_access[round][i].a[j][1] <== fri_query_step0_v[round][j][1];
        }
      }
      if (i == 1) {
        c_random_access[round][i] = RandomAccess2(NUM_FRI_QUERY_STEP1_V(), 2);
        for (var j = 0; j < NUM_FRI_QUERY_STEP1_V(); j++) {
          c_random_access[round][i].a[j][0] <== fri_query_step1_v[round][j][0];
          c_random_access[round][i].a[j][1] <== fri_query_step1_v[round][j][1];
        }
      }
      c_random_access[round][i].idx <== x_index_within_coset[round][i].out;
      old_eval[round][i][0] === c_random_access[round][i].out[0];
      old_eval[round][i][1] === c_random_access[round][i].out[1];

      // get_points
      rev_x_index_within_coset[round][i] = ReverseBits(arity_bits[i]);
      rev_x_index_within_coset[round][i].x <== x_index_within_coset[round][i].out;
      p_exp[round][i] = GlExp();
      p_exp[round][i].x <== G_BY_ARITY_BITS(arity_bits[i]);
      p_exp[round][i].n <== arity - rev_x_index_within_coset[round][i].out;
      p_mul[round][i][0] = GlMul();
      p_mul[round][i][0].a <== subgroup_x[round][i][0];
      p_mul[round][i][0].b <== p_exp[round][i].out;
      points[round][i][0] <== p_mul[round][i][0].out;
      for (var j = 1; j < arity; j++) {
        p_mul[round][i][j] = GlMul();
        p_mul[round][i][j].a <== points[round][i][j - 1];
        p_mul[round][i][j].b <== G_BY_ARITY_BITS(arity_bits[i]);
        points[round][i][j] <== p_mul[round][i][j].out;
      }

      // compute evaluation
      barycentric_weights[round][i] = CalBarycentricWeights(arity);
      for (var j = 0; j < arity; j++) {
        barycentric_weights[round][i].points[j] <== points[round][i][j];
      }
      e_sub0[round][i] = GlExtSub();
      e_sub0[round][i].a[0] <== fri_betas[i][0];
      e_sub0[round][i].a[1] <== fri_betas[i][1];
      e_sub0[round][i].b[0] <== points[round][i][0];
      e_sub0[round][i].b[1] <== 0;
      l_x[round][i][0][0] <== e_sub0[round][i].out[0];
      l_x[round][i][0][1] <== e_sub0[round][i].out[1];

      for (var j = 0; j < arity - 1; j++) {
        e_sub1[round][i][j] = GlExtSub();
        e_mul1[round][i][j] = GlExtMul();
        e_sub1[round][i][j].a[0] <== fri_betas[i][0];
        e_sub1[round][i][j].a[1] <== fri_betas[i][1];
        e_sub1[round][i][j].b[0] <== points[round][i][j + 1];
        e_sub1[round][i][j].b[1] <== 0;
        e_mul1[round][i][j].a[0] <== l_x[round][i][j][0];
        e_mul1[round][i][j].a[1] <== l_x[round][i][j][1];
        e_mul1[round][i][j].b[0] <== e_sub1[round][i][j].out[0];
        e_mul1[round][i][j].b[1] <== e_sub1[round][i][j].out[1];
        l_x[round][i][j + 1][0] <== e_mul1[round][i][j].out[0];
        l_x[round][i][j + 1][1] <== e_mul1[round][i][j].out[1];
      }

      for (var j = 0; j < arity; j++) {
        e_add2[round][i][j] = GlExtAdd();
        e_sub2[round][i][j] = GlExtSub();
        e_mul2[round][i][j] = GlExtMul();
        e_div2[round][i][j] = GlExtDiv();
        e_rev[round][i][j] = ReverseBits(arity_bits[i]);

        e_sub2[round][i][j].a[0] <== fri_betas[i][0];
        e_sub2[round][i][j].a[1] <== fri_betas[i][1];
        e_sub2[round][i][j].b[0] <== points[round][i][j];
        e_sub2[round][i][j].b[1] <== 0;
        e_div2[round][i][j].a[0] <== barycentric_weights[round][i].out[j];
        e_div2[round][i][j].a[1] <== 0;
        e_div2[round][i][j].b[0] <== e_sub2[round][i][j].out[0];
        e_div2[round][i][j].b[1] <== e_sub2[round][i][j].out[1];

        e_rev[round][i][j].x <== j;
        if (i == 0) {
          e_ra[round][i][j] = RandomAccess2(NUM_FRI_QUERY_STEP0_V(), 2);
          e_ra[round][i][j].idx <== e_rev[round][i][j].out;
          for (var k = 0; k < NUM_FRI_QUERY_STEP0_V(); k++) {
            e_ra[round][i][j].a[k][0] <== fri_query_step0_v[round][k][0];
            e_ra[round][i][j].a[k][1] <== fri_query_step0_v[round][k][1];
          }
        }
        if (i == 1) {
          e_ra[round][i][j] = RandomAccess2(NUM_FRI_QUERY_STEP1_V(), 2);
          e_ra[round][i][j].idx <== e_rev[round][i][j].out;
          for (var k = 0; k < NUM_FRI_QUERY_STEP1_V(); k++) {
            e_ra[round][i][j].a[k][0] <== fri_query_step1_v[round][k][0];
            e_ra[round][i][j].a[k][1] <== fri_query_step1_v[round][k][1];
          }
        }
        e_mul2[round][i][j].a[0] <== e_div2[round][i][j].out[0];
        e_mul2[round][i][j].a[1] <== e_div2[round][i][j].out[1];
        e_mul2[round][i][j].b[0] <== e_ra[round][i][j].out[0];
        e_mul2[round][i][j].b[1] <== e_ra[round][i][j].out[1];
        if (j > 0) {
          e_add2[round][i][j].a[0] <== e_add2[round][i][j - 1].out[0];
          e_add2[round][i][j].a[1] <== e_add2[round][i][j - 1].out[1];
        } else {
          e_add2[round][i][j].a[0] <== 0;
          e_add2[round][i][j].a[1] <== 0;
        }
        e_add2[round][i][j].b[0] <== e_mul2[round][i][j].out[0];
        e_add2[round][i][j].b[1] <== e_mul2[round][i][j].out[1];
      }

      // TODO: c witness generator does not allow empty component declaration
      for (var j = arity; j < max_arity; j++) {
        p_mul[round][i][j] = GlMul();
        e_add2[round][i][j] = GlExtAdd();
        e_sub2[round][i][j] = GlExtSub();
        e_div2[round][i][j] = GlExtDiv();
        e_mul2[round][i][j] = GlExtMul();
        e_rev[round][i][j] = ReverseBits(1);
        e_ra[round][i][j] = RandomAccess2(1, 1);
      }
      for (var j = arity; j < max_arity - 1; j++) {
        e_sub1[round][i][j] = GlExtSub();
        e_mul1[round][i][j] = GlExtMul();
      }


      e_mul0[round][i] = GlExtMul();
      e_mul0[round][i].a[0] <== l_x[round][i][arity - 1][0];
      e_mul0[round][i].a[1] <== l_x[round][i][arity - 1][1];
      e_mul0[round][i].b[0] <== e_add2[round][i][arity - 1].out[0];
      e_mul0[round][i].b[1] <== e_add2[round][i][arity - 1].out[1];
      old_eval[round][i + 1][0] <== e_mul0[round][i].out[0];
      old_eval[round][i + 1][1] <== e_mul0[round][i].out[1];

      // step 0
      if (i == 0) {
        merkle_caps[round][4] = GetMerkleProofToCap(NUM_FRI_QUERY_STEP0_V() * 2,
                                                    NUM_FRI_QUERY_STEP0_P());
        merkle_caps[round][4].leaf_index <== coset_index[round][i].out;
        for (var j = 0; j < NUM_FRI_QUERY_STEP0_V(); j++) {
          merkle_caps[round][4].leaf[j * 2] <== fri_query_step0_v[round][j][0];
          merkle_caps[round][4].leaf[j * 2 + 1] <== fri_query_step0_v[round][j][1];
        }
        for (var j = 0; j < NUM_FRI_QUERY_STEP0_P(); j++) {
          merkle_caps[round][4].proof[j][0] <== fri_query_step0_p[round][j][0];
          merkle_caps[round][4].proof[j][1] <== fri_query_step0_p[round][j][1];
          merkle_caps[round][4].proof[j][2] <== fri_query_step0_p[round][j][2];
          merkle_caps[round][4].proof[j][3] <== fri_query_step0_p[round][j][3];
        }
        c_commit_merkle_cap[round][i] = RandomAccess2(FRI_COMMIT_MERKLE_CAP_HEIGHT(), 4);
        for (var j = 0; j < FRI_COMMIT_MERKLE_CAP_HEIGHT(); j++) {
          c_commit_merkle_cap[round][i].a[j][0] <== fri_commit_phase_merkle_caps[i][j][0];
          c_commit_merkle_cap[round][i].a[j][1] <== fri_commit_phase_merkle_caps[i][j][1];
          c_commit_merkle_cap[round][i].a[j][2] <== fri_commit_phase_merkle_caps[i][j][2];
          c_commit_merkle_cap[round][i].a[j][3] <== fri_commit_phase_merkle_caps[i][j][3];
        }
        c_commit_merkle_cap[round][i].idx <== merkle_caps[round][4].index;
        merkle_caps[round][4].digest[0] === c_commit_merkle_cap[round][i].out[0];
        merkle_caps[round][4].digest[1] === c_commit_merkle_cap[round][i].out[1];
        merkle_caps[round][4].digest[2] === c_commit_merkle_cap[round][i].out[2];
        merkle_caps[round][4].digest[3] === c_commit_merkle_cap[round][i].out[3];
      }

      // step 1
      if (i == 1) {
        merkle_caps[round][5] = GetMerkleProofToCap(NUM_FRI_QUERY_STEP1_V() * 2,
                                                    NUM_FRI_QUERY_STEP1_P());
        merkle_caps[round][5].leaf_index <== coset_index[round][i].out;
        for (var j = 0; j < NUM_FRI_QUERY_STEP1_V(); j++) {
          merkle_caps[round][5].leaf[j * 2] <== fri_query_step1_v[round][j][0];
          merkle_caps[round][5].leaf[j * 2 + 1] <== fri_query_step1_v[round][j][1];
        }
        for (var j = 0; j < NUM_FRI_QUERY_STEP1_P(); j++) {
          merkle_caps[round][5].proof[j][0] <== fri_query_step1_p[round][j][0];
          merkle_caps[round][5].proof[j][1] <== fri_query_step1_p[round][j][1];
          merkle_caps[round][5].proof[j][2] <== fri_query_step1_p[round][j][2];
          merkle_caps[round][5].proof[j][3] <== fri_query_step1_p[round][j][3];
        }
        c_commit_merkle_cap[round][i] = RandomAccess2(FRI_COMMIT_MERKLE_CAP_HEIGHT(), 4);
        for (var j = 0; j < FRI_COMMIT_MERKLE_CAP_HEIGHT(); j++) {
          c_commit_merkle_cap[round][i].a[j][0] <== fri_commit_phase_merkle_caps[i][j][0];
          c_commit_merkle_cap[round][i].a[j][1] <== fri_commit_phase_merkle_caps[i][j][1];
          c_commit_merkle_cap[round][i].a[j][2] <== fri_commit_phase_merkle_caps[i][j][2];
          c_commit_merkle_cap[round][i].a[j][3] <== fri_commit_phase_merkle_caps[i][j][3];
        }
        c_commit_merkle_cap[round][i].idx <== merkle_caps[round][5].index;
        merkle_caps[round][5].digest[0] === c_commit_merkle_cap[round][i].out[0];
        merkle_caps[round][5].digest[1] === c_commit_merkle_cap[round][i].out[1];
        merkle_caps[round][5].digest[2] === c_commit_merkle_cap[round][i].out[2];
        merkle_caps[round][5].digest[3] === c_commit_merkle_cap[round][i].out[3];
      }

      p_exp2[round][i] = GlExpPowerOf2(arity_bits[i]);
      p_exp2[round][i].x <== subgroup_x[round][i][0];
      subgroup_x[round][i + 1][0] <== p_exp2[round][i].out;
      subgroup_x[round][i + 1][1] <== subgroup_x[round][i][1];
    }

    for (var i = NUM_FRI_FINAL_POLY_EXT_V(); i > 0; i--) {
      c_final_eval_mul[round][i - 1] = GlExtMul();
      c_final_eval_add[round][i - 1] = GlExtAdd();
      if (i == NUM_FRI_FINAL_POLY_EXT_V()) {
        c_final_eval_mul[round][i - 1].a[0] <== 0;
        c_final_eval_mul[round][i - 1].a[1] <== 0;
      } else {
        c_final_eval_mul[round][i - 1].a[0] <== c_final_eval_add[round][i].out[0];
        c_final_eval_mul[round][i - 1].a[1] <== c_final_eval_add[round][i].out[1];
      }
      c_final_eval_mul[round][i - 1].b[0] <== subgroup_x[round][NUM_REDUCTION_ARITY_BITS()][0];
      c_final_eval_mul[round][i - 1].b[1] <== subgroup_x[round][NUM_REDUCTION_ARITY_BITS()][1];
      c_final_eval_add[round][i - 1].a[0] <== fri_final_poly_ext_v[i - 1][0];
      c_final_eval_add[round][i - 1].a[1] <== fri_final_poly_ext_v[i - 1][1];
      c_final_eval_add[round][i - 1].b[0] <== c_final_eval_mul[round][i - 1].out[0];
      c_final_eval_add[round][i - 1].b[1] <== c_final_eval_mul[round][i - 1].out[1];
    }
    old_eval[round][NUM_REDUCTION_ARITY_BITS()][0] === c_final_eval_add[round][0].out[0];
    old_eval[round][NUM_REDUCTION_ARITY_BITS()][1] === c_final_eval_add[round][0].out[1];
  }
}
