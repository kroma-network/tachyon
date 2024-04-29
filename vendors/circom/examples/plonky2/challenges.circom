pragma circom 2.1.0;
include "./constants.circom";
include "./poseidon.circom";

template GetChallenges() {
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

  signal input fri_commit_phase_merkle_caps[NUM_FRI_COMMIT_ROUND()][FRI_COMMIT_MERKLE_CAP_HEIGHT()][4];
  signal input fri_final_poly_ext_v[NUM_FRI_FINAL_POLY_EXT_V()][2];
  signal input fri_pow_witness;
  signal input public_input_hash[4];
  signal input circuit_digest[4];

  signal output plonk_betas[NUM_CHALLENGES()];
  signal output plonk_gammas[NUM_CHALLENGES()];
  signal output plonk_alphas[NUM_CHALLENGES()];
  signal output plonk_zeta[2];
  signal output fri_alpha[2];
  signal output fri_betas[NUM_FRI_COMMIT_ROUND()][2];
  signal output fri_pow_response;
  signal output fri_query_indices[NUM_FRI_QUERY_ROUND()];

  assert(NUM_CHALLENGES() == 2);

  /// batch 0
  var num_inputs_batch_0 = /* circuit digest */ 4 + /* public input */ 4 + NUM_WIRES_CAP() * 4;

  component observe_batch_0 = HashNoPad_BN(num_inputs_batch_0, SPONGE_WIDTH());
  for (var i = 0; i < 4; i++) {
    observe_batch_0.in[i] <== circuit_digest[i];
  }
  for (var i = 0; i < 4; i++) {
    observe_batch_0.in[i + 4] <== public_input_hash[i];
  }
  for (var i = 0; i < NUM_WIRES_CAP(); i++) {
    for (var j = 0; j < 4; j ++) {
      observe_batch_0.in[i * 4 + j + 8] <== wires_cap[i][j];
    }
  }
  for (var i = 0; i < 4; i++) {
      observe_batch_0.capacity[i] <== 0;
  }
  for (var i = 0; i < NUM_CHALLENGES(); i++) {
    plonk_betas[i] <== observe_batch_0.out[SPONGE_RATE() - 1 - i];
    // log(plonk_betas[i]);
  }
  for (var i = 0; i < NUM_CHALLENGES(); i++) {
    plonk_gammas[i] <== observe_batch_0.out[SPONGE_RATE() - 1 - NUM_CHALLENGES() - i];
    // log(plonk_gammas[i]);
  }

  /// batch 1
  var num_inputs_batch_1 = NUM_PLONK_ZS_PARTIAL_PRODUCTS_CAP() * 4;
  component observe_batch_1 = HashNoPad_BN(num_inputs_batch_1 < SPONGE_RATE() ? SPONGE_RATE() : num_inputs_batch_1,
                                           SPONGE_WIDTH());
  for (var i = 0; i < NUM_PLONK_ZS_PARTIAL_PRODUCTS_CAP(); i++) {
    for (var j = 0; j < 4; j ++) {
      observe_batch_1.in[i * 4 + j] <== plonk_zs_partial_products_cap[i][j];
    }
  }
  if (num_inputs_batch_1 < SPONGE_RATE()) {
    for (var i = num_inputs_batch_1; i < SPONGE_RATE(); i++) {
      observe_batch_1.in[i] <== observe_batch_0.out[i];
    }
  }
  for (var i = 0; i < 4; i++) {
      observe_batch_1.capacity[i] <== observe_batch_0.out[SPONGE_RATE() + i];
  }
  for (var i = 0; i < NUM_CHALLENGES(); i++) {
    plonk_alphas[i] <== observe_batch_1.out[SPONGE_RATE() - 1 - i];
    // log(plonk_alphas[i]);
  }

  /// batch 2
  var num_inputs_batch_2 = NUM_QUOTIENT_POLYS_CAP() * 4;
  component observe_batch_2 = HashNoPad_BN(num_inputs_batch_2 < SPONGE_RATE() ? SPONGE_RATE() : num_inputs_batch_2,
                                           SPONGE_WIDTH());
  for (var i = 0; i < NUM_QUOTIENT_POLYS_CAP(); i++) {
    for (var j = 0; j < 4; j ++) {
      observe_batch_2.in[i * 4 + j] <== quotient_polys_cap[i][j];
    }
  }
  if (num_inputs_batch_2 < SPONGE_RATE()) {
    for (var i = num_inputs_batch_2; i < SPONGE_RATE(); i++) {
      observe_batch_2.in[i] <== observe_batch_1.out[i];
    }
  }
  for (var i = 0; i < 4; i++) {
      observe_batch_2.capacity[i] <== observe_batch_1.out[SPONGE_RATE() + i];
  }
  for (var i = 0; i < NUM_CHALLENGES(); i++) {
    plonk_zeta[i] <== observe_batch_2.out[SPONGE_RATE() - 1 - i];
    // log(plonk_zeta[i]);
  }

  /// batch 3
  var num_inputs_batch_3 = (NUM_OPENINGS_CONSTANTS() + NUM_OPENINGS_PLONK_SIGMAS() + NUM_OPENINGS_WIRES()
                           + NUM_OPENINGS_PLONK_ZS() + NUM_OPENINGS_PARTIAL_PRODUCTS() + NUM_OPENINGS_QUOTIENT_POLYS()
                           + NUM_OPENINGS_PLONK_ZS_NEXT()) * 2;
  component observe_batch_3 = HashNoPad_BN(num_inputs_batch_3, SPONGE_WIDTH());
  var idx = 0;
  for (var i = 0; i < NUM_OPENINGS_CONSTANTS(); i++) {
    observe_batch_3.in[idx] <== openings_constants[i][0];
    observe_batch_3.in[idx + 1] <== openings_constants[i][1];
    idx += 2;
  }
  for (var i = 0; i < NUM_OPENINGS_PLONK_SIGMAS(); i++) {
    observe_batch_3.in[idx] <== openings_plonk_sigmas[i][0];
    observe_batch_3.in[idx + 1] <== openings_plonk_sigmas[i][1];
    idx += 2;
  }
  for (var i = 0; i < NUM_OPENINGS_WIRES(); i++) {
    observe_batch_3.in[idx] <== openings_wires[i][0];
    observe_batch_3.in[idx + 1] <== openings_wires[i][1];
    idx += 2;
  }
  for (var i = 0; i < NUM_OPENINGS_PLONK_ZS(); i++) {
    observe_batch_3.in[idx] <== openings_plonk_zs[i][0];
    observe_batch_3.in[idx + 1] <== openings_plonk_zs[i][1];
    idx += 2;
  }
  for (var i = 0; i < NUM_OPENINGS_PARTIAL_PRODUCTS(); i++) {
    observe_batch_3.in[idx] <== openings_partial_products[i][0];
    observe_batch_3.in[idx + 1] <== openings_partial_products[i][1];
    idx += 2;
  }
  for (var i = 0; i < NUM_OPENINGS_QUOTIENT_POLYS(); i++) {
    observe_batch_3.in[idx] <== openings_quotient_polys[i][0];
    observe_batch_3.in[idx + 1] <== openings_quotient_polys[i][1];
    idx += 2;
  }
  for (var i = 0; i < NUM_OPENINGS_PLONK_ZS_NEXT(); i++) {
    observe_batch_3.in[idx] <== openings_plonk_zs_next[i][0];
    observe_batch_3.in[idx + 1] <== openings_plonk_zs_next[i][1];
    idx += 2;
  }
  for (var i = 0; i < 4; i++) {
      observe_batch_3.capacity[i] <== observe_batch_2.out[SPONGE_RATE() + i];
  }
  for (var i = 0; i < 2; i++) {
    fri_alpha[i] <== observe_batch_3.out[SPONGE_RATE() - 1 - i];
    // log(fri_alpha[i]);
  }

  /// batch 4
  component observe_batch_4[NUM_FRI_COMMIT_ROUND()];
  for (var round = 0; round < NUM_FRI_COMMIT_ROUND(); round++) {
    var num_inputs = FRI_COMMIT_MERKLE_CAP_HEIGHT() * 4;
    observe_batch_4[round] = HashNoPad_BN(num_inputs < SPONGE_RATE() ? SPONGE_RATE() : num_inputs, SPONGE_WIDTH());
    for (var i = 0; i < FRI_COMMIT_MERKLE_CAP_HEIGHT(); i++) {
      for (var j = 0; j < 4; j ++) {
        observe_batch_4[round].in[i * 4 + j] <== fri_commit_phase_merkle_caps[round][i][j];
      }
    }
    if (num_inputs < SPONGE_RATE()) {
      for (var i = num_inputs; i < SPONGE_RATE(); i++) {
        if (round == 0) {
          observe_batch_4[round].in[i] <== observe_batch_3.out[i];
        } else {
          observe_batch_4[round].in[i] <== observe_batch_4[round - 1].out[i];
        }
      }
    }
    for (var i = 0; i < 4; i++) {
      if (round == 0) {
        observe_batch_4[round].capacity[i] <== observe_batch_3.out[SPONGE_RATE() + i];
      } else {
        observe_batch_4[round].capacity[i] <== observe_batch_4[round - 1].out[SPONGE_RATE() + i];
      }
    }
    for (var i = 0; i < 2; i++) {
      fri_betas[round][i] <== observe_batch_4[round].out[SPONGE_RATE() - 1 - i];
      // log(fri_betas[round][i]);
    }
  }

  /// batch 5
  var num_inputs_batch_5 = NUM_FRI_FINAL_POLY_EXT_V() * 2 + 1;
  component observe_batch_5 = HashNoPad_BN(num_inputs_batch_5, SPONGE_WIDTH());
  for (var i = 0; i < NUM_FRI_FINAL_POLY_EXT_V(); i++) {
    observe_batch_5.in[i * 2] <== fri_final_poly_ext_v[i][0];
    observe_batch_5.in[i * 2 + 1] <== fri_final_poly_ext_v[i][1];
  }
  observe_batch_5.in[num_inputs_batch_5 - 1] <== fri_pow_witness;
  for (var i = 0; i < 4; i++) {
    observe_batch_5.capacity[i] <== observe_batch_4[NUM_FRI_COMMIT_ROUND() - 1].out[SPONGE_RATE() + i];
  }
  fri_pow_response <== observe_batch_5.out[SPONGE_RATE() - 1];
  // log(fri_pow_response);

  /// Get fri_query_indices
  // First SPONGE_RATE() - 1 = 7 indices
  component mod_lde_size[NUM_FRI_QUERY_ROUND()];
  for (var i = 0; i < NUM_FRI_QUERY_ROUND(); i++) {
    mod_lde_size[i] = LastNBits(DEGREE_BITS() + FRI_RATE_BITS());
  }
  for (var i = 1; i < SPONGE_RATE(); i++) {
    mod_lde_size[i - 1].x <== observe_batch_5.out[SPONGE_RATE() - 1 - i];
    fri_query_indices[i - 1] <== mod_lde_size[i - 1].out;
    // log(fri_query_indices[i - 1]);
  }

  assert(NUM_FRI_QUERY_ROUND() <= 7 + 3 * SPONGE_RATE());
  component observe_batch_6 = HashNoPad_BN(SPONGE_RATE(), SPONGE_WIDTH());
  for (var i = 0; i < SPONGE_RATE(); i++) {
    observe_batch_6.in[i] <== observe_batch_5.out[i];
  }
  for (var i = 0; i < 4; i++) {
    observe_batch_6.capacity[i] <== observe_batch_5.out[SPONGE_RATE() + i];
  }
  for (var i = 7; i < NUM_FRI_QUERY_ROUND() && i < 7 + SPONGE_RATE(); i++) {
    mod_lde_size[i].x <== observe_batch_6.out[SPONGE_RATE() - 1 - (i - 7)];
    fri_query_indices[i] <== mod_lde_size[i].out;
    // log(fri_query_indices[i]);
  }

  component observe_batch_7 = HashNoPad_BN(SPONGE_RATE(), SPONGE_WIDTH());
  if (NUM_FRI_QUERY_ROUND() - 7 > SPONGE_RATE()) {
    for (var i = 0; i < SPONGE_RATE(); i++) {
      observe_batch_7.in[i] <== observe_batch_6.out[i];
    }
    for (var i = 0; i < 4; i++) {
      observe_batch_7.capacity[i] <== observe_batch_6.out[SPONGE_RATE() + i];
    }
    for (var i = 7 + SPONGE_RATE(); i < NUM_FRI_QUERY_ROUND() && i < 7 + 2 * SPONGE_RATE(); i++) {
      mod_lde_size[i].x <== observe_batch_7.out[SPONGE_RATE() - 1 - (i - 7 - SPONGE_RATE())];
      fri_query_indices[i] <== mod_lde_size[i].out;
      // log(fri_query_indices[i]);
    }
  }

  component observe_batch_8 = HashNoPad_BN(SPONGE_RATE(), SPONGE_WIDTH());
  if (NUM_FRI_QUERY_ROUND() - 7 > 2 * SPONGE_RATE()) {
    for (var i = 0; i < SPONGE_RATE(); i++) {
      observe_batch_8.in[i] <== observe_batch_7.out[i];
    }
    for (var i = 0; i < 4; i++) {
      observe_batch_8.capacity[i] <== observe_batch_7.out[SPONGE_RATE() + i];
    }
    for (var i = 7 + 2 * SPONGE_RATE(); i < NUM_FRI_QUERY_ROUND() && i < 7 + 3 * SPONGE_RATE(); i++) {
      mod_lde_size[i].x <== observe_batch_8.out[SPONGE_RATE() - 1 - (i - 7 - 2 * SPONGE_RATE())];
      fri_query_indices[i] <== mod_lde_size[i].out;
      // log(fri_query_indices[i]);
    }
  }
}
