// TODO: check all inputs are 64 bits

pragma circom 2.1.0;
include "./challenges.circom";
include "./plonk.circom";
include "./fri.circom";

template VerifyPlonky2Proof() {
  signal input circuit_digest[4];
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
  signal input fri_pow_witness;
  signal input public_inputs[NUM_PUBLIC_INPUTS()];

  component public_input_hasher = HashNoPad_GL(NUM_PUBLIC_INPUTS(), 4);
  public_input_hasher.in <== public_inputs;
  public_input_hasher.capacity[0] <== 0;
  public_input_hasher.capacity[1] <== 0;
  public_input_hasher.capacity[2] <== 0;
  public_input_hasher.capacity[3] <== 0;

  component get_challenges = GetChallenges();

  get_challenges.wires_cap <== wires_cap;
  get_challenges.plonk_zs_partial_products_cap <== plonk_zs_partial_products_cap;
  get_challenges.quotient_polys_cap <== quotient_polys_cap;

  get_challenges.openings_constants <== openings_constants;
  get_challenges.openings_plonk_sigmas <== openings_plonk_sigmas;
  get_challenges.openings_wires <== openings_wires;
  get_challenges.openings_plonk_zs <== openings_plonk_zs;
  get_challenges.openings_plonk_zs_next <== openings_plonk_zs_next;
  get_challenges.openings_partial_products <== openings_partial_products;
  get_challenges.openings_quotient_polys <== openings_quotient_polys;

  get_challenges.fri_commit_phase_merkle_caps <== fri_commit_phase_merkle_caps;
  get_challenges.fri_final_poly_ext_v <== fri_final_poly_ext_v;
  get_challenges.fri_pow_witness <== fri_pow_witness;
  get_challenges.public_input_hash <== public_input_hasher.out;
  get_challenges.circuit_digest  <== circuit_digest;

  component eval_vanishing_poly = EvalVanishingPoly();

  eval_vanishing_poly.plonk_betas <== get_challenges.plonk_betas;
  eval_vanishing_poly.plonk_zeta <== get_challenges.plonk_zeta;
  eval_vanishing_poly.plonk_gammas <== get_challenges.plonk_gammas;
  eval_vanishing_poly.openings_constants <== openings_constants;
  eval_vanishing_poly.openings_wires <== openings_wires;
  eval_vanishing_poly.openings_plonk_zs <== openings_plonk_zs;
  eval_vanishing_poly.openings_plonk_sigmas <== openings_plonk_sigmas;
  eval_vanishing_poly.openings_plonk_zs_next <== openings_plonk_zs_next;
  eval_vanishing_poly.openings_partial_products <== openings_partial_products;
  eval_vanishing_poly.public_input_hash <== public_input_hasher.out;

  component check_zeta = CheckZeta();

  check_zeta.openings_quotient_polys <== openings_quotient_polys;
  check_zeta.plonk_alphas <== get_challenges.plonk_alphas;
  check_zeta.plonk_zeta <== get_challenges.plonk_zeta;
  check_zeta.constraint_terms <== eval_vanishing_poly.constraint_terms;
  check_zeta.vanishing_partial_products_terms <== eval_vanishing_poly.vanishing_partial_products_terms;
  check_zeta.vanishing_z_1_terms <== eval_vanishing_poly.vanishing_z_1_terms;

  component verify_fri_proof = VerifyFriProof();

  verify_fri_proof.wires_cap <== wires_cap;
  verify_fri_proof.plonk_zs_partial_products_cap <== plonk_zs_partial_products_cap;
  verify_fri_proof.quotient_polys_cap <== quotient_polys_cap;

  verify_fri_proof.openings_constants <== openings_constants;
  verify_fri_proof.openings_plonk_sigmas <== openings_plonk_sigmas;
  verify_fri_proof.openings_wires <== openings_wires;
  verify_fri_proof.openings_plonk_zs <== openings_plonk_zs;
  verify_fri_proof.openings_plonk_zs_next <== openings_plonk_zs_next;
  verify_fri_proof.openings_partial_products <== openings_partial_products;
  verify_fri_proof.openings_quotient_polys <== openings_quotient_polys;
  verify_fri_proof.fri_constants_sigmas_cap <== fri_constants_sigmas_cap;

  verify_fri_proof.fri_commit_phase_merkle_caps <== fri_commit_phase_merkle_caps;
  verify_fri_proof.fri_query_init_constants_sigmas_v <== fri_query_init_constants_sigmas_v;
  verify_fri_proof.fri_query_init_constants_sigmas_p <== fri_query_init_constants_sigmas_p;
  verify_fri_proof.fri_query_init_wires_v <== fri_query_init_wires_v;
  verify_fri_proof.fri_query_init_wires_p <== fri_query_init_wires_p;
  verify_fri_proof.fri_query_init_zs_partial_v <== fri_query_init_zs_partial_v;
  verify_fri_proof.fri_query_init_zs_partial_p <== fri_query_init_zs_partial_p;
  verify_fri_proof.fri_query_init_quotient_v <== fri_query_init_quotient_v;
  verify_fri_proof.fri_query_init_quotient_p <== fri_query_init_quotient_p;
  verify_fri_proof.fri_query_step0_v <== fri_query_step0_v;
  verify_fri_proof.fri_query_step0_p <== fri_query_step0_p;
  verify_fri_proof.fri_query_step1_v <== fri_query_step1_v;
  verify_fri_proof.fri_query_step1_p <== fri_query_step1_p;
  verify_fri_proof.fri_final_poly_ext_v <== fri_final_poly_ext_v;

  // Challenges
  verify_fri_proof.plonk_zeta <== get_challenges.plonk_zeta;
  verify_fri_proof.fri_alpha <== get_challenges.fri_alpha;
  verify_fri_proof.fri_betas <== get_challenges.fri_betas;
  verify_fri_proof.fri_pow_response <== get_challenges.fri_pow_response;
  verify_fri_proof.fri_query_indices <== get_challenges.fri_query_indices;
}

component main {public [public_inputs]} = VerifyPlonky2Proof();
