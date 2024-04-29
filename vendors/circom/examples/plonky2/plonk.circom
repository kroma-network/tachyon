pragma circom 2.1.0;
include "./goldilocks_ext.circom";
include "./utils.circom";
include "./gates.circom";

template EvalL1() {
  signal input n;
  signal input x[2];
  signal output out[2];

  signal x_sub_one[2];
  x_sub_one[0] <== x[0] - 1;
  x_sub_one[1] <== x[1];

  component cem = GlExtExp();
  cem.x[0] <== x[0];
  cem.x[1] <== x[1];
  cem.n <== n;

  component xn0 = GlReduce(64);
  xn0.x <== x_sub_one[0] * n;
  component xn1 = GlReduce(64);
  xn1.x <== x_sub_one[1] * n;
  component ced = GlExtDiv();
  ced.a[0] <== cem.out[0] - 1;
  ced.a[1] <== cem.out[1];
  ced.b[0] <== xn0.out;
  ced.b[1] <== xn1.out;

  out[0] <== ced.out[0];
  out[1] <== ced.out[1];
}

template EvalVanishingPoly() {
  signal input plonk_betas[NUM_CHALLENGES()];
  signal input plonk_zeta[2];
  signal input plonk_gammas[NUM_CHALLENGES()];
  signal input openings_constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input openings_wires[NUM_OPENINGS_WIRES()][2];
  signal input openings_plonk_zs[NUM_OPENINGS_PLONK_ZS()][2];
  signal input openings_plonk_sigmas[NUM_OPENINGS_PLONK_SIGMAS()][2];
  signal input openings_plonk_zs_next[NUM_OPENINGS_PLONK_ZS_NEXT()][2];
  signal input openings_partial_products[NUM_OPENINGS_PARTIAL_PRODUCTS()][2];
  signal input public_input_hash[4];

  signal output constraint_terms[NUM_GATE_CONSTRAINTS()][2];
  signal output vanishing_partial_products_terms[NUM_PARTIAL_PRODUCTS_TERMS() * NUM_CHALLENGES()][2];
  signal output vanishing_z_1_terms[NUM_CHALLENGES()][2];

  signal constraints[NUM_GATE_CONSTRAINTS()][2];
  for (var i = 0; i < NUM_GATE_CONSTRAINTS(); i++) {
    constraints[i][0] <== 0;
    constraints[i][1] <== 0;
  }

  component c_eval_gates = EvalGateConstraints();
  c_eval_gates.constants <== openings_constants;
  c_eval_gates.wires <== openings_wires;
  c_eval_gates.public_input_hash <== public_input_hash;
  c_eval_gates.constraints <== constraints;
  constraint_terms <== c_eval_gates.out;

  signal l1_x[2] <== EvalL1()((1 << DEGREE_BITS()), plonk_zeta);
  signal one[2];
  one[0] <== 1;
  one[1] <== 0;

  signal numerator_values[NUM_CHALLENGES()][NUM_OPENINGS_PLONK_SIGMAS()][2][2];
  signal denominator_values[NUM_CHALLENGES()][NUM_OPENINGS_PLONK_SIGMAS()][2][2];
  signal accs[NUM_CHALLENGES()][NUM_PARTIAL_PRODUCTS_TERMS() + 1][2];
  signal numerator_prod[NUM_CHALLENGES()][NUM_PARTIAL_PRODUCTS_TERMS()][QUOTIENT_DEGREE_FACTOR()][2];
  signal denominator_prod[NUM_CHALLENGES()][NUM_PARTIAL_PRODUCTS_TERMS()][QUOTIENT_DEGREE_FACTOR()][2];

  assert(NUM_PARTIAL_PRODUCTS_TERMS() == NUM_OPENINGS_PARTIAL_PRODUCTS() \ NUM_CHALLENGES() + 1);
  for (var i = 0; i < NUM_CHALLENGES(); i++) {
    vanishing_z_1_terms[i] <== GlExtMul()(l1_x, GlExtSub()(openings_plonk_zs[i], one));
    for (var j = 0; j < NUM_OPENINGS_PLONK_SIGMAS(); j ++) {
      numerator_values[i][j][0] <== GlExtAdd()(openings_wires[j], GlExtScalarMul()(GlExtScalarMul()(plonk_zeta, K_IS(j)), plonk_betas[i]));
      numerator_values[i][j][1][0] <== GlAdd()(numerator_values[i][j][0][0], plonk_gammas[i]);
      numerator_values[i][j][1][1] <== numerator_values[i][j][0][1];

      denominator_values[i][j][0] <== GlExtAdd()(openings_wires[j], GlExtScalarMul()(openings_plonk_sigmas[j], plonk_betas[i]));
      denominator_values[i][j][1][0] <== GlAdd()(denominator_values[i][j][0][0], plonk_gammas[i]);
      denominator_values[i][j][1][1] <== denominator_values[i][j][0][1];
    }
    accs[i][0] <== openings_plonk_zs[i];
    accs[i][NUM_PARTIAL_PRODUCTS_TERMS()] <== openings_plonk_zs_next[i];
    for (var j = 1; j < NUM_OPENINGS_PARTIAL_PRODUCTS() \ NUM_CHALLENGES() + 1; j++) {
      accs[i][j] <== openings_partial_products[i * (NUM_OPENINGS_PARTIAL_PRODUCTS() \ NUM_CHALLENGES()) + j - 1];
    }
    var pos = 0;
    for (var j = 0; j < NUM_PARTIAL_PRODUCTS_TERMS(); j++) {
      numerator_prod[i][j][0] <== numerator_values[i][pos][1];
      denominator_prod[i][j][0] <== denominator_values[i][pos][1];
      pos++;
      var last_k = 0;
      for (var k = 1; k < QUOTIENT_DEGREE_FACTOR() && pos < NUM_OPENINGS_PLONK_SIGMAS(); k++) {
        numerator_prod[i][j][k] <== GlExtMul()(numerator_prod[i][j][k - 1], numerator_values[i][pos][1]);
        denominator_prod[i][j][k] <== GlExtMul()(denominator_prod[i][j][k - 1], denominator_values[i][pos][1]);
        last_k = k;
        pos++;
      }
      // Avoid uninitialized signals.
      for (var k = last_k + 1; k < QUOTIENT_DEGREE_FACTOR(); k++) {
        numerator_prod[i][j][k][0] <== 0;
        numerator_prod[i][j][k][1] <== 0;
        denominator_prod[i][j][k][0] <== 0;
        denominator_prod[i][j][k][1] <== 0;
      }
      vanishing_partial_products_terms[NUM_PARTIAL_PRODUCTS_TERMS() * i + j] <== GlExtSub()(
          GlExtMul()(accs[i][j], numerator_prod[i][j][last_k]),
          GlExtMul()(accs[i][j + 1], denominator_prod[i][j][last_k])
        );
    }
  }
}

template CheckZeta() {
  signal input openings_quotient_polys[NUM_OPENINGS_QUOTIENT_POLYS()][2];
  signal input plonk_alphas[NUM_CHALLENGES()];
  signal input plonk_zeta[2];
  signal input constraint_terms[NUM_GATE_CONSTRAINTS()][2];
  signal input vanishing_partial_products_terms[NUM_PARTIAL_PRODUCTS_TERMS() * NUM_CHALLENGES()][2];
  signal input vanishing_z_1_terms[NUM_CHALLENGES()][2];

  component c_reduce[NUM_CHALLENGES()][3];
  for (var i = 0; i < NUM_CHALLENGES(); i++) {
    c_reduce[i][0] = Reduce(NUM_GATE_CONSTRAINTS());
    for (var j = 0; j < NUM_GATE_CONSTRAINTS(); j++) {
      c_reduce[i][0].in[j][0] <== constraint_terms[j][0];
      c_reduce[i][0].in[j][1] <== constraint_terms[j][1];
    }
    c_reduce[i][0].alpha[0] <== plonk_alphas[i];
    c_reduce[i][0].alpha[1] <== 0;
    c_reduce[i][0].old_eval[0] <== 0;
    c_reduce[i][0].old_eval[1] <== 0;

    c_reduce[i][1] = Reduce(NUM_PARTIAL_PRODUCTS_TERMS() * NUM_CHALLENGES());
    for (var j = 0; j < NUM_PARTIAL_PRODUCTS_TERMS() * NUM_CHALLENGES(); j++) {
      // Circom is very buggy, got error with:
      // c_reduce[i][1].in[j] <== vanishing_partial_products_terms[j];
      c_reduce[i][1].in[j][0] <== vanishing_partial_products_terms[j][0];
      c_reduce[i][1].in[j][1] <== vanishing_partial_products_terms[j][1];
    }
    c_reduce[i][1].alpha[0] <== plonk_alphas[i];
    c_reduce[i][1].alpha[1] <== 0;
    c_reduce[i][1].old_eval <== c_reduce[i][0].out;

    c_reduce[i][2] = Reduce(NUM_CHALLENGES());
    for (var j = 0; j < NUM_CHALLENGES(); j++) {
      c_reduce[i][2].in[j][0] <== vanishing_z_1_terms[j][0];
      c_reduce[i][2].in[j][1] <== vanishing_z_1_terms[j][1];
    }
    c_reduce[i][2].alpha[0] <== plonk_alphas[i];
    c_reduce[i][2].alpha[1] <== 0;
    c_reduce[i][2].old_eval <== c_reduce[i][1].out;
  }

  signal zeta_pow_deg[2] <== GlExtExpPowerOf2(DEGREE_BITS())(plonk_zeta);
  signal one[2];
  one[0] <== 1;
  one[1] <== 0;
  signal z_h_zeta[2] <== GlExtSub()(zeta_pow_deg, one);
  signal zeta[NUM_CHALLENGES()][2];
  component c_reduce_with_powers[NUM_CHALLENGES()];
  for (var i = 0; i < NUM_CHALLENGES(); i++) {
    c_reduce_with_powers[i] = Reduce(QUOTIENT_DEGREE_FACTOR());
    c_reduce_with_powers[i].old_eval[0] <== 0;
    c_reduce_with_powers[i].old_eval[1] <== 0;
    c_reduce_with_powers[i].alpha <== zeta_pow_deg;
    for (var j = 0; j < QUOTIENT_DEGREE_FACTOR(); j++) {
      c_reduce_with_powers[i].in[j][0] <== openings_quotient_polys[i * QUOTIENT_DEGREE_FACTOR() + j][0];
      c_reduce_with_powers[i].in[j][1] <== openings_quotient_polys[i * QUOTIENT_DEGREE_FACTOR() + j][1];
    }
    zeta[i] <== GlExtMul()(z_h_zeta, c_reduce_with_powers[i].out);

    c_reduce[i][2].out === zeta[i];
  }
}
