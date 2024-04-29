pragma circom 2.1.0;
include "./goldilocks_ext.circom";
include "./utils.circom";
include "./poseidon.circom";

template AlgebraMul() {
        signal input l_0[2];
        signal input l_1[2];
        signal input r_0[2];
        signal input r_1[2];
        signal output out_0[2];
        signal output out_1[2];
        out_0 <== GlExtAdd()(GlExtMul()(l_0, r_0), GlExtMul()(GlExtMul()(GlExt(7, 0)(), l_1), r_1));
        out_1 <== GlExtAdd()(GlExtMul()(l_0, r_1), GlExtMul()(l_1, r_0));
}

template WiresAlgebraMul(l, r) {
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal output out[2][2];
  out[0] <== GlExtAdd()(GlExtMul()(wires[l], wires[r]), GlExtMul()(GlExtMul()(GlExt(7, 0)(), wires[l + 1]), wires[r + 1]));
  out[1] <== GlExtAdd()(GlExtMul()(wires[l], wires[r + 1]), GlExtMul()(wires[l + 1], wires[r]));
}

template ConstraintPush() {
  signal input constraint[2];
  signal input filter[2];
  signal input value[2];

  signal output out[2];
  out <== GlExtAdd()(constraint, GlExtMul()(value, filter));
}

template EvalGateConstraints() {
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  // ConstantGate { num_consts: 2 }
  component c_Constant2 = Constant2();
  c_Constant2.constants <== constants;
  c_Constant2.wires <== wires;
  c_Constant2.public_input_hash <== public_input_hash;
  c_Constant2.constraints <== constraints;

  // PoseidonMdsGate(PhantomData<plonky2_field::goldilocks_field::GoldilocksField>)<WIDTH=12>
  component c_PoseidonMdsGate12 = PoseidonMdsGate12();
  c_PoseidonMdsGate12.constants <== constants;
  c_PoseidonMdsGate12.wires <== wires;
  c_PoseidonMdsGate12.public_input_hash <== public_input_hash;
  c_PoseidonMdsGate12.constraints <== c_Constant2.out;

  // PublicInputGate
  component c_PublicInputGateLib = PublicInputGateLib();
  c_PublicInputGateLib.constants <== constants;
  c_PublicInputGateLib.wires <== wires;
  c_PublicInputGateLib.public_input_hash <== public_input_hash;
  c_PublicInputGateLib.constraints <== c_PoseidonMdsGate12.out;

  // BaseSumGate { num_limbs: 63 } + Base: 2
  component c_BaseSum63 = BaseSum63();
  c_BaseSum63.constants <== constants;
  c_BaseSum63.wires <== wires;
  c_BaseSum63.public_input_hash <== public_input_hash;
  c_BaseSum63.constraints <== c_PublicInputGateLib.out;

  // ReducingExtensionGate { num_coeffs: 32 }
  component c_ReducingExtension32 = ReducingExtension32();
  c_ReducingExtension32.constants <== constants;
  c_ReducingExtension32.wires <== wires;
  c_ReducingExtension32.public_input_hash <== public_input_hash;
  c_ReducingExtension32.constraints <== c_BaseSum63.out;

  // ReducingGate { num_coeffs: 43 }
  component c_Reducing43 = Reducing43();
  c_Reducing43.constants <== constants;
  c_Reducing43.wires <== wires;
  c_Reducing43.public_input_hash <== public_input_hash;
  c_Reducing43.constraints <== c_ReducingExtension32.out;

  // ArithmeticExtensionGate { num_ops: 10 }
  component c_ArithmeticExtension10 = ArithmeticExtension10();
  c_ArithmeticExtension10.constants <== constants;
  c_ArithmeticExtension10.wires <== wires;
  c_ArithmeticExtension10.public_input_hash <== public_input_hash;
  c_ArithmeticExtension10.constraints <== c_Reducing43.out;

  // ArithmeticGate { num_ops: 20 }
  component c_Arithmetic20 = Arithmetic20();
  c_Arithmetic20.constants <== constants;
  c_Arithmetic20.wires <== wires;
  c_Arithmetic20.public_input_hash <== public_input_hash;
  c_Arithmetic20.constraints <== c_ArithmeticExtension10.out;

  // MulExtensionGate { num_ops: 13 }
  component c_MultiplicationExtension13 = MultiplicationExtension13();
  c_MultiplicationExtension13.constants <== constants;
  c_MultiplicationExtension13.wires <== wires;
  c_MultiplicationExtension13.public_input_hash <== public_input_hash;
  c_MultiplicationExtension13.constraints <== c_Arithmetic20.out;

  // RandomAccessGate { bits: 4, num_copies: 4, num_extra_constants: 2, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }<D=2>
  component c_RandomAccessB4C4E2 = RandomAccessB4C4E2();
  c_RandomAccessB4C4E2.constants <== constants;
  c_RandomAccessB4C4E2.wires <== wires;
  c_RandomAccessB4C4E2.public_input_hash <== public_input_hash;
  c_RandomAccessB4C4E2.constraints <== c_MultiplicationExtension13.out;

  // CosetInterpolationGate { subgroup_bits: 4, degree: 6, barycentric_weights: [17293822565076172801, 18374686475376656385, 18446744069413535745, 281474976645120, 17592186044416, 256, 18446744000695107601, 18446744065119617025, 1152921504338411520, 72057594037927936, 1048576, 18446462594437939201, 18446726477228539905, 18446744069414584065, 68719476720, 4294967296], _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }<D=2>
  component c_CosetInterpolation4 = CosetInterpolation4();
  c_CosetInterpolation4.constants <== constants;
  c_CosetInterpolation4.wires <== wires;
  c_CosetInterpolation4.public_input_hash <== public_input_hash;
  c_CosetInterpolation4.constraints <== c_RandomAccessB4C4E2.out;

  // PoseidonGate(PhantomData<plonky2_field::goldilocks_field::GoldilocksField>)<WIDTH=12>
  component c_Poseidon12 = Poseidon12();
  c_Poseidon12.constants <== constants;
  c_Poseidon12.wires <== wires;
  c_Poseidon12.public_input_hash <== public_input_hash;
  c_Poseidon12.constraints <== c_CosetInterpolation4.out;
  out <== c_Poseidon12.out;
}
template Constant2() {
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  filter <== GlExtMul()(GlExtSub()(GlExt(0, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(2, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(3, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(4, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(5, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(6, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(4294967295, 0)(), constants[0]), GlExt(1, 0)())))))));

  for (var i = 0; i < 2; i++) {
    out[i] <== ConstraintPush()(constraints[i], filter, GlExtSub()(constants[3 + i], wires[i]));
  }
  for (var i = 2; i < NUM_GATE_CONSTRAINTS(); i++) {
    out[i] <== constraints[i];
  }
}
template PoseidonMdsGate12() {
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  filter <== GlExtMul()(GlExtSub()(GlExt(0, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(1, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(3, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(4, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(5, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(6, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(4294967295, 0)(), constants[0]), GlExt(1, 0)())))))));

  signal state[13][12][2][2];
  for (var r = 0; r < 12; r++) {
    for (var i = 0; i < 12; i++) {
      var j = i + r >= 12 ? i + r - 12 : i + r;
      if (i == 0) {
        state[i][r][0] <== GlExtScalarMul()(wires[j * 2], MDS_MATRIX_CIRC(i));
        state[i][r][1] <== GlExtScalarMul()(wires[j * 2 + 1], MDS_MATRIX_CIRC(i));
      } else {
        state[i][r][0] <== GlExtAdd()(state[i - 1][r][0], GlExtScalarMul()(wires[j * 2], MDS_MATRIX_CIRC(i)));
        state[i][r][1] <== GlExtAdd()(state[i - 1][r][1], GlExtScalarMul()(wires[j * 2 + 1], MDS_MATRIX_CIRC(i)));
      }
    }
    state[12][r][0] <== GlExtAdd()(state[11][r][0], GlExtScalarMul()(wires[r * 2], MDS_MATRIX_DIAG(r)));
    state[12][r][1] <== GlExtAdd()(state[11][r][1], GlExtScalarMul()(wires[r * 2 + 1], MDS_MATRIX_DIAG(r)));
  }

  for (var r = 0; r < 12; r ++) {
    out[r * 2] <== ConstraintPush()(constraints[r * 2], filter, GlExtSub()(wires[(12 + r) * 2], state[12][r][0]));
    out[r * 2 + 1] <== ConstraintPush()(constraints[r * 2 + 1], filter, GlExtSub()(wires[(12 + r) * 2 + 1], state[12][r][1]));
  }

  for (var i = 24; i < NUM_GATE_CONSTRAINTS(); i++) {
    out[i] <== constraints[i];
  }
}
template PublicInputGateLib() {
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  filter <== GlExtMul()(GlExtSub()(GlExt(0, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(1, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(2, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(4, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(5, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(6, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(4294967295, 0)(), constants[0]), GlExt(1, 0)())))))));

  signal hashes[4][2];
  for (var i = 0; i < 4; i++) {
    hashes[i][0] <== public_input_hash[i];
    hashes[i][1] <== 0;
    out[i] <== ConstraintPush()(constraints[i], filter, GlExtSub()(wires[i], hashes[i]));
  }
  for (var i = 4; i < NUM_GATE_CONSTRAINTS(); i++) {
    out[i] <== constraints[i];
  }
}
template BaseSum63() {
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  filter <== GlExtMul()(GlExtSub()(GlExt(0, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(1, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(2, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(3, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(5, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(6, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(4294967295, 0)(), constants[0]), GlExt(1, 0)())))))));

  component reduce = Reduce(63);
  reduce.alpha <== GlExt(2, 0)();
  reduce.old_eval <== GlExt(0, 0)();
  for (var i = 1; i < 63 + 1; i++) {
    reduce.in[i - 1] <== wires[i];
  }
  out[0] <== ConstraintPush()(constraints[0], filter, GlExtSub()(reduce.out, wires[0]));
  component product[63][2 - 1];
  for (var i = 0; i < 63; i++) {
    for (var j = 0; j < 2 - 1; j++) {
      product[i][j] = GlExtMul();
      if (j == 0) product[i][j].a <== wires[i + 1];
      else product[i][j].a <== product[i][j - 1].out;
      product[i][j].b <== GlExtSub()(wires[i + 1], GlExt(j + 1, 0)());
    }
    out[i + 1] <== ConstraintPush()(constraints[i + 1], filter, product[i][2 - 2].out);
  }
  for (var i = 63 + 1; i < NUM_GATE_CONSTRAINTS(); i++) {
    out[i] <== constraints[i];
  }
}
template ReducingExtension32() {
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  filter <== GlExtMul()(GlExtSub()(GlExt(0, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(1, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(2, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(3, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(4, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(6, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(4294967295, 0)(), constants[0]), GlExt(1, 0)())))))));

  var acc_start = 2 * 2;
  signal m[32][2][2];
  for (var i = 0; i < 32; i++) {
    m[i] <== WiresAlgebraMul(acc_start, 2)(wires);
    for (var j = 0; j < 2; j++) {
      out[i * 2 + j] <== ConstraintPush()(constraints[i * 2 + j], filter, GlExtAdd()(m[i][j], GlExtSub()(wires[(3 + i) * 2 + j], wires[re_wires_accs_start(i, 32) + j])));
    }
    acc_start = re_wires_accs_start(i, 32);
  }

  for (var i = 32 * 2; i < NUM_GATE_CONSTRAINTS(); i++) {
    out[i] <== constraints[i];
  }
}
function re_wires_accs_start(i, num_coeffs) {
  if (i == num_coeffs - 1) return 0;
  else return (3 + i + num_coeffs) * 2;
}
template Reducing43() {
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  filter <== GlExtMul()(GlExtSub()(GlExt(0, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(1, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(2, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(3, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(4, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(5, 0)(), constants[0]), GlExtMul()(GlExtSub()(GlExt(4294967295, 0)(), constants[0]), GlExt(1, 0)())))))));

  var acc_start = 2 * 2;
  signal m[43][2][2];
  for (var i = 0; i < 43; i++) {
    m[i] <== WiresAlgebraMul(acc_start, 2)(wires);
    out[i * 2] <== ConstraintPush()(constraints[i * 2], filter, GlExtAdd()(m[i][0], GlExtSub()(wires[3 * 2 + i], wires[r_wires_accs_start(i, 43)])));
    for (var j = 1; j < 2; j++) {
      out[i * 2 + j] <== ConstraintPush()(constraints[i * 2 + j], filter, GlExtSub()(m[i][j], wires[r_wires_accs_start(i, 43) + j]));
    }
    acc_start = r_wires_accs_start(i, 43);
  }

  for (var i = 43 * 2; i < NUM_GATE_CONSTRAINTS(); i++) {
    out[i] <== constraints[i];
  }
}
function r_wires_accs_start(i, num_coeffs) {
  if (i == num_coeffs - 1) return 0;
  else return (3 + i) * 2 + num_coeffs;
}
template ArithmeticExtension10() {
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  filter <== GlExtMul()(GlExtSub()(GlExt(8, 0)(), constants[1]), GlExtMul()(GlExtSub()(GlExt(9, 0)(), constants[1]), GlExtMul()(GlExtSub()(GlExt(10, 0)(), constants[1]), GlExtMul()(GlExtSub()(GlExt(4294967295, 0)(), constants[1]), GlExt(1, 0)()))));

  signal m[10][2][2];
  for (var i = 0; i < 10; i++) {
    m[i] <== WiresAlgebraMul(4 * 2 * i, 4 * 2 * i + 2)(wires);
    for (var j = 0; j < 2; j++) {
      out[i * 2 + j] <== ConstraintPush()(constraints[i * 2 + j], filter, GlExtSub()(wires[4 * 2 * i + 3 * 2 + j], GlExtAdd()(GlExtMul()(m[i][j], constants[3]), GlExtMul()(wires[4 * 2 * i + 2 * 2 + j], constants[3 + 1]))));
    }
  }

  for (var i = 10 * 2; i < NUM_GATE_CONSTRAINTS(); i++) {
    out[i] <== constraints[i];
  }
}
template Arithmetic20() {
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  filter <== GlExtMul()(GlExtSub()(GlExt(7, 0)(), constants[1]), GlExtMul()(GlExtSub()(GlExt(9, 0)(), constants[1]), GlExtMul()(GlExtSub()(GlExt(10, 0)(), constants[1]), GlExtMul()(GlExtSub()(GlExt(4294967295, 0)(), constants[1]), GlExt(1, 0)()))));

  for (var i = 0; i < 20; i++) {
    out[i] <== ConstraintPush()(constraints[i], filter, GlExtSub()(wires[4 * i + 3], GlExtAdd()(GlExtMul()(GlExtMul()(wires[4 * i], wires[4 * i + 1]), constants[3 + 0]), GlExtMul()(wires[4 * i + 2], constants[3 + 1]))));
  }

  for (var i = 20; i < NUM_GATE_CONSTRAINTS(); i++) {
    out[i] <== constraints[i];
  }
}
template MultiplicationExtension13() {
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  filter <== GlExtMul()(GlExtSub()(GlExt(7, 0)(), constants[1]), GlExtMul()(GlExtSub()(GlExt(8, 0)(), constants[1]), GlExtMul()(GlExtSub()(GlExt(10, 0)(), constants[1]), GlExtMul()(GlExtSub()(GlExt(4294967295, 0)(), constants[1]), GlExt(1, 0)()))));

  signal m[13][2][2];
  for (var i = 0; i < 13; i++) {
    m[i] <== WiresAlgebraMul(3 * 2 * i, 3 * 2 * i + 2)(wires);
    for (var j = 0; j < 2; j++) {
      out[i * 2 + j] <== ConstraintPush()(constraints[i * 2 + j], filter, GlExtSub()(wires[3 * 2 * i + 2 * 2 + j], GlExtMul()(m[i][j], constants[3])));
    }
  }
  for (var i = 13 * 2; i < NUM_GATE_CONSTRAINTS(); i++) {
    out[i] <== constraints[i];
  }
}
template RandomAccessB4C4E2() {
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  filter <== GlExtMul()(GlExtSub()(GlExt(7, 0)(), constants[1]), GlExtMul()(GlExtSub()(GlExt(8, 0)(), constants[1]), GlExtMul()(GlExtSub()(GlExt(9, 0)(), constants[1]), GlExtMul()(GlExtSub()(GlExt(4294967295, 0)(), constants[1]), GlExt(1, 0)()))));

  var index = 0;
  signal acc[4][4][2];
  signal list_items[4][4 + 1][16][2];
  for (var copy = 0; copy < 4; copy++) {
    for (var i = 0; i < 4; i++) {
      out[index] <== ConstraintPush()(constraints[index], filter,
        GlExtMul()(wires[ra_wire_bit(i, copy)], GlExtSub()(wires[ra_wire_bit(i, copy)], GlExt(1, 0)())));
      index++;
    }
    for (var i = 4; i > 0; i--) {
      if(i == 4) {
        acc[copy][i - 1] <== wires[ra_wire_bit(i - 1, copy)];
      } else {
        acc[copy][i - 1] <== GlExtAdd()(GlExtAdd()(acc[copy][i], acc[copy][i]), wires[ra_wire_bit(i - 1, copy)]);
      }
    }
    out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(acc[copy][0], wires[(2 + 16) * copy]));
    index++;
    for (var i = 0; i < 16; i++) {
      list_items[copy][0][i] <== wires[(2 + 16) * copy + 2 + i];
    }
    for (var i = 0; i < 4; i++) {
      for (var j = 0; j < (16 >> i); j = j + 2) {
        list_items[copy][i + 1][j \ 2] <== GlExtAdd()(list_items[copy][i][j], GlExtMul()(wires[ra_wire_bit(i, copy)], GlExtSub()(list_items[copy][i][j + 1], list_items[copy][i][j])));
      }
    }
    out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(list_items[copy][4][0], wires[(2 + 16) * copy + 1]));
    index++;
  }
  for (var i = 0; i < 2; i++) {
    out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(constants[3 + i], wires[(2 + 16) * 4 + i]));
    index++;
  }

  for (var i = index; i < NUM_GATE_CONSTRAINTS(); i++) {
    out[i] <== constraints[i];
  }
}
function ra_wire_bit(i, copy) {
  return 74 + copy * 4 + i;
}
template CosetInterpolation4() {
   signal input constants[NUM_OPENINGS_CONSTANTS()][2];
   signal input wires[NUM_OPENINGS_WIRES()][2];
   signal input public_input_hash[4];
   signal input constraints[NUM_GATE_CONSTRAINTS()][2];
   signal output out[NUM_GATE_CONSTRAINTS()][2];

   signal filter[2];
   filter <== GlExtMul()(GlExtSub()(GlExt(12, 0)(), constants[2]), GlExtMul()(GlExtSub()(GlExt(4294967295, 0)(), constants[2]), GlExt(1, 0)()));

   var index = 0;
   var start_values = 1;
   var start_evaluation_point = start_values+ get_point_num() * get_d();
   var start_evaluation_value = start_evaluation_point + get_d();

   var start_intermediates = start_evaluation_value + get_d();
   var wires_shifted_evaluation_point = start_intermediates + get_d()*2*get_intermediates_num();

   signal shift[2];
   signal evaluation_point[2][2];
   signal shifted_evaluation_point[2][2];

   shift <== wires[0];

   for (var i = 0; i < get_d(); i++) {
        evaluation_point[i] <== wires[start_evaluation_point+i];
   }

   for (var i = 0; i < get_d(); i++) {
     shifted_evaluation_point[i] <== wires[wires_shifted_evaluation_point+i];
   }

   for (var i = 0; i < get_d(); i++) {
      out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()( evaluation_point[i] , GlExtMul()(shifted_evaluation_point[i], shift)));
      index++;
    }


     var num_intermediates = 0;

     signal start_eval[2][2] <== [[0,0], [0,0]];
     signal start_prod[2][2] <== [[1,0], [0,0]];

    signal eval[(get_intermediates_num()+1)*2][2];
    signal partial_prod[(get_intermediates_num()+1)*2][2];

    (eval[0], eval[1], partial_prod[0], partial_prod[1]) <== partial_interpolate(0, get_degree())(wires, shifted_evaluation_point, start_eval[0], start_eval[1],start_prod[0],  start_prod[1]);
    signal intermediate_eval[get_intermediates_num()*2][2];
    signal intermediate_prod[get_intermediates_num()*2][2];

    for (var i = 0; i < get_intermediates_num(); i++){
        for (var j = 0; j < get_d(); j++) {
            intermediate_eval[2*i+j] <== wires[start_intermediates+2*i+j];
            intermediate_prod[2*i+j]<== wires[start_intermediates+2*(get_intermediates_num()+i)+j];
        }

        for (var j = 0; j < get_d(); j++) {
            out[index] <==  ConstraintPush()(constraints[index], filter, GlExtSub()(intermediate_eval[2*i+j], eval[2*i+j]));
            index++;
        }

        for (var j = 0; j < get_d(); j++) {
            out[index] <==  ConstraintPush()(constraints[index], filter, GlExtSub()(intermediate_prod[2*i+j], partial_prod[2*i+j]));
            index++;
        }
        var start =  1 + (get_degree() - 1) * (i + 1);
        var end = (start + get_degree() - 1) > get_point_num() ? get_point_num() : (start + get_degree() - 1);

        (eval[(i+1)*2], eval[(i+1)*2+1], partial_prod[(i+1)*2], partial_prod[(i+1)*2 + 1]) <== partial_interpolate(start, end)(wires, shifted_evaluation_point, intermediate_eval[2*i], intermediate_eval[2*i+1], intermediate_prod[2*i], intermediate_prod[2*i+1]);

    }

    signal evaluation_value[2][2];
    for (var i = 0; i < get_d(); i++) {
        evaluation_value[i] <== wires[start_evaluation_value+i];
        out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(evaluation_value[i], eval[get_intermediates_num()*2+i]));
        index++;
    }
    for (var i = index; i < NUM_GATE_CONSTRAINTS(); i++) {
      out[i] <== constraints[i];
    }
}

template partial_interpolate(start, end) {
    signal input wires[NUM_OPENINGS_WIRES()][2];
    signal input shifted_evaluation_point[2][2];
    signal input initial_eval_0[2];
    signal input initial_eval_1[2];
    signal input initial_partial_prod_0[2];
    signal input initial_partial_prod_1[2];

    signal weighted_values[(end-start)*2][2];
    signal term[(end-start)*2][2];
    signal ext_barycentric_weights[(end-start)][2];

    signal eval[(end-start+1)*2][2];
    signal partial_prod[(end-start+1)*2][2];

    signal eval_term[(end-start)*2][2];
    signal eval_weight[(end-start)*2][2];

    signal output eval_out_0[2];
    signal output eval_out_1[2];
    signal output partial_prod_out_0[2];
    signal output partial_prod_out_1[2];

    var start_values = 1;

    eval[0] <== initial_eval_0;
    eval[1] <== initial_eval_1;

    partial_prod[0] <== initial_partial_prod_0;
    partial_prod[1] <== initial_partial_prod_1;

    for  (var i = start; i < end ; i++) {
        ext_barycentric_weights[(i-start)][0] <== barycentric_weights(i);
        ext_barycentric_weights[(i-start)][1] <== 0;
        var pt_pos = start_values+ get_d()*i;
        for (var j = 0; j < get_d(); j++) {
            weighted_values[2*(i-start)+j] <== GlExtMul()(wires[pt_pos+j], ext_barycentric_weights[(i-start)]);
        }

        term[2*(i-start)][0] <== GlSub()(shifted_evaluation_point[0][0], two_adic_subgroup(i));
        term[2*(i-start)][1] <== shifted_evaluation_point[0][1];
        term[2*(i-start)+1] <== shifted_evaluation_point[1];

        (eval_term[2*(i-start)], eval_term[2*(i-start)+1]) <== AlgebraMul()(eval[(i-start)*2], eval[(i-start)*2+1], term[2*(i-start)], term[2*(i-start) +1]);
        (eval_weight[2*(i-start)], eval_weight[2*(i-start)+1] )<== AlgebraMul()(partial_prod[(i-start)*2], partial_prod[(i-start)*2+1], weighted_values[2*(i-start)], weighted_values[2*(i-start)+1]);

        for (var j = 0; j < get_d(); j++) {
            eval[(i-start+1)*2+j] <== GlExtAdd()(eval_term[2*(i-start)+j], eval_weight[2*(i-start)+j]);
        }

        (partial_prod[2*(i-start+1)], partial_prod[2*(i-start+1)+1]) <== AlgebraMul()(partial_prod[(i-start)*2], partial_prod[(i-start)*2+1], term[2*(i-start)], term[2*(i-start) +1]);

    }
    eval_out_0 <== eval[(end-start)*2];
    eval_out_1 <== eval[(end-start)*2 + 1];

    partial_prod_out_0 <== partial_prod[(end-start)*2];
    partial_prod_out_1 <== partial_prod[(end-start)*2 + 1];
}

function get_degree() {
    return 6;
}

function get_d() {
    return 2;
}

function get_point_num() {
    return 16;
}

function get_intermediates_num() {
    return 2;
}

function barycentric_weights(i) {
  var barycentric_weights[16];
    barycentric_weights[0] = 17293822565076172801;
  barycentric_weights[1] = 18374686475376656385;
  barycentric_weights[2] = 18446744069413535745;
  barycentric_weights[3] = 281474976645120;
  barycentric_weights[4] = 17592186044416;
  barycentric_weights[5] = 256;
  barycentric_weights[6] = 18446744000695107601;
  barycentric_weights[7] = 18446744065119617025;
  barycentric_weights[8] = 1152921504338411520;
  barycentric_weights[9] = 72057594037927936;
  barycentric_weights[10] = 1048576;
  barycentric_weights[11] = 18446462594437939201;
  barycentric_weights[12] = 18446726477228539905;
  barycentric_weights[13] = 18446744069414584065;
  barycentric_weights[14] = 68719476720;
  barycentric_weights[15] = 4294967296;
  return barycentric_weights[i];
}
function two_adic_subgroup(i) {
  var subgroup[16];
  subgroup[0] = 1;
  subgroup[1] = 17293822564807737345;
  subgroup[2] = 18446744069397807105;
  subgroup[3] = 4503599626321920;
  subgroup[4] = 281474976710656;
  subgroup[5] = 4096;
  subgroup[6] = 18446742969902956801;
  subgroup[7] = 18446744000695107585;
  subgroup[8] = 18446744069414584320;
  subgroup[9] = 1152921504606846976;
  subgroup[10] = 16777216;
  subgroup[11] = 18442240469788262401;
  subgroup[12] = 18446462594437873665;
  subgroup[13] = 18446744069414580225;
  subgroup[14] = 1099511627520;
  subgroup[15] = 68719476736;
  return subgroup[i];
}
template Poseidon12() {
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  filter <== GlExtMul()(GlExtSub()(GlExt(11, 0)(), constants[2]), GlExtMul()(GlExtSub()(GlExt(4294967295, 0)(), constants[2]), GlExt(1, 0)()));

  var index = 0;
  out[index] <== ConstraintPush()(constraints[index], filter, GlExtMul()(wires[24], GlExtSub()(wires[24], GlExt(1, 0)())));
  index++;

  for (var i = 0; i < 4; i++) {
    out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(GlExtMul()(wires[24], GlExtSub()(wires[i + 4], wires[i])), wires[25 + i]));
    index++;
  }

  // SPONGE_RATE = 8
  // SPONGE_CAPACITY = 4
  // SPONGE_WIDTH = 12
  signal state[12][4 * 8 + 2 + 22 * 2][2];
  var state_round = 0;
  for (var i = 0; i < 4; i++) {
    state[i][state_round] <== GlExtAdd()(wires[i], wires[25 + i]);
    state[i + 4][state_round] <== GlExtSub()(wires[i + 4], wires[25 + i]);
  }

  for (var i = 8; i < 12; i++) {
    state[i][state_round] <== wires[i];
  }
  state_round++;

  var round_ctr = 0;
  // First set of full rounds.
  signal mds_row_shf_field[4][12][13][2];
  for (var r = 0; r < 4; r ++) {
    for (var i = 0; i < 12; i++) {
      state[i][state_round] <== GlExtAdd()(state[i][state_round - 1], GlExt(GL_CONST(i + 12 * round_ctr), 0)());
    }
    state_round++;
    if (r != 0 ) {
      for (var i = 0; i < 12; i++) {
        state[i][state_round] <== wires[25 + 4 + 12 * (r - 1) + i];
        out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(state[i][state_round - 1], state[i][state_round]));
        index++;
      }
      state_round++;
    }
    for (var i = 0; i < 12; i++) {
      state[i][state_round] <== GlExtExpN(3)(state[i][state_round - 1], 7);
    }
    state_round++;
    for (var i = 0; i < 12; i++) { // for r
      mds_row_shf_field[r][i][0][0] <== 0;
      mds_row_shf_field[r][i][0][1] <== 0;
      for (var j = 0; j < 12; j++) { // for i,
        mds_row_shf_field[r][i][j + 1] <== GlExtAdd()(mds_row_shf_field[r][i][j], GlExtMul()(state[(i + j) < 12 ? (i + j) : (i + j - 12)][state_round - 1], GlExt(MDS_MATRIX_CIRC(j), 0)()));
      }
      state[i][state_round] <== GlExtAdd()(mds_row_shf_field[r][i][12], GlExtMul()(state[i][state_round - 1], GlExt(MDS_MATRIX_DIAG(i), 0)()));
    }
    state_round++;
    round_ctr++;
  }

  // Partial rounds.
  for (var i = 0; i < 12; i++) {
    state[i][state_round] <== GlExtAdd()(state[i][state_round - 1], GlExt(FAST_PARTIAL_FIRST_ROUND_CONSTANT(i), 0)());
  }
  state_round++;
  component partial_res[11][11];
  state[0][state_round] <== state[0][state_round - 1];
  for (var r = 0; r < 11; r++) {
    for (var c = 0; c < 11; c++) {
      partial_res[r][c] = GlExtAdd();
      if (r == 0) {
        partial_res[r][c].a <== GlExt(0, 0)();
      } else {
        partial_res[r][c].a <== partial_res[r - 1][c].out;
      }
      partial_res[r][c].b <== GlExtMul()(state[r + 1][state_round - 1], GlExt(FAST_PARTIAL_ROUND_INITIAL_MATRIX(r, c), 0)());
    }
  }
  for (var i = 1; i < 12; i++) {
    state[i][state_round] <== partial_res[10][i - 1].out;
  }
  state_round++;

  signal partial_d[12][22][2];
  for (var r = 0; r < 22; r++) {
    out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(state[0][state_round - 1], wires[65 + r]));
    index++;
    if (r == 22 - 1) {
      state[0][state_round] <== GlExtExpN(3)(wires[65 + r], 7);
    } else {
      state[0][state_round] <== GlExtAdd()(GlExt(FAST_PARTIAL_ROUND_CONSTANTS(r), 0)(), GlExtExpN(3)(wires[65 + r], 7));
    }
    for (var i = 1; i < 12; i++) {
      state[i][state_round] <== state[i][state_round - 1];
    }
    partial_d[0][r] <== GlExtMul()(state[0][state_round], GlExt(MDS_MATRIX_CIRC(0) + MDS_MATRIX_DIAG(0), 0)());
    for (var i = 1; i < 12; i++) {
      partial_d[i][r] <== GlExtAdd()(partial_d[i - 1][r], GlExtMul()(state[i][state_round], GlExt(FAST_PARTIAL_ROUND_W_HATS(r, i - 1), 0)()));
    }
    state_round++;
    state[0][state_round] <== partial_d[11][r];
    for (var i = 1; i < 12; i++) {
      state[i][state_round] <== GlExtAdd()(state[i][state_round - 1], GlExtMul()(state[0][state_round - 1], GlExt(FAST_PARTIAL_ROUND_VS(r, i - 1), 0)()));
    }
    state_round++;
  }
  round_ctr += 22;

  // Second set of full rounds.
  signal mds_row_shf_field2[4][12][13][2];
  for (var r = 0; r < 4; r ++) {
    for (var i = 0; i < 12; i++) {
      state[i][state_round] <== GlExtAdd()(state[i][state_round - 1], GlExt(GL_CONST(i + 12 * round_ctr), 0)());
    }
    state_round++;
    for (var i = 0; i < 12; i++) {
      state[i][state_round] <== wires[87 + 12 * r + i];
      out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(state[i][state_round - 1], state[i][state_round]));
      index++;
    }
    state_round++;
    for (var i = 0; i < 12; i++) {
      state[i][state_round] <== GlExtExpN(3)(state[i][state_round - 1], 7);
    }
    state_round++;
    for (var i = 0; i < 12; i++) { // for r
      mds_row_shf_field2[r][i][0][0] <== 0;
      mds_row_shf_field2[r][i][0][1] <== 0;
      for (var j = 0; j < 12; j++) { // for i,
        mds_row_shf_field2[r][i][j + 1] <== GlExtAdd()(mds_row_shf_field2[r][i][j], GlExtMul()(state[(i + j) < 12 ? (i + j) : (i + j - 12)][state_round - 1], GlExt(MDS_MATRIX_CIRC(j), 0)()));
      }
      state[i][state_round] <== GlExtAdd()(mds_row_shf_field2[r][i][12], GlExtMul()(state[i][state_round - 1], GlExt(MDS_MATRIX_DIAG(i), 0)()));
    }
    state_round++;
    round_ctr++;
  }

  for (var i = 0; i < 12; i++) {
    out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(state[i][state_round - 1], wires[12 + i]));
    index++;
  }

  for (var i = index + 1; i < NUM_GATE_CONSTRAINTS(); i++) {
    out[i] <== constraints[i];
  }
}
function FAST_PARTIAL_ROUND_W_HATS(i, j) {
  var value[22][11];
  value[0][0] = 4438751076270498736;
  value[0][1] = 9317528645525775657;
  value[0][2] = 2603614750616077704;
  value[0][3] = 9834445229934519080;
  value[0][4] = 11955300617986087719;
  value[0][5] = 13674383287779636394;
  value[0][6] = 7242667852302110551;
  value[0][7] = 703710881370165964;
  value[0][8] = 5061939192123688976;
  value[0][9] = 14416184509556335938;
  value[0][10] = 304868360577598380;
  value[1][0] = 7437226027186543243;
  value[1][1] = 15353050892319980048;
  value[1][2] = 3199984117275729523;
  value[1][3] = 11990763268329609629;
  value[1][4] = 5577680852675862792;
  value[1][5] = 17892201254274048377;
  value[1][6] = 4681998189446302081;
  value[1][7] = 6822112447852802370;
  value[1][8] = 7318824523402736059;
  value[1][9] = 63486289239724471;
  value[1][10] = 9953444262837494154;
  value[2][0] = 2317103059171007623;
  value[2][1] = 16480286982765085951;
  value[2][2] = 13705213611198486247;
  value[2][3] = 10236515677047503770;
  value[2][4] = 6341681382391377123;
  value[2][5] = 6362787076607341484;
  value[2][6] = 10057473295910894055;
  value[2][7] = 12586789805515730111;
  value[2][8] = 4352300357074435274;
  value[2][9] = 15739906440350539774;
  value[2][10] = 16786966705537008710;
  value[3][0] = 14247238213840877673;
  value[3][1] = 4982197628621364471;
  value[3][2] = 1650209613801527344;
  value[3][3] = 16334009413005742380;
  value[3][4] = 320004518447392347;
  value[3][5] = 7777559975827687149;
  value[3][6] = 1266186313330142639;
  value[3][7] = 12735743610080455214;
  value[3][8] = 9621059894918028247;
  value[3][9] = 4350447204024668858;
  value[3][10] = 11420240845800225374;
  value[4][0] = 1701204778899409548;
  value[4][1] = 12463216732586668885;
  value[4][2] = 7392209094895994703;
  value[4][3] = 15680934805691729401;
  value[4][4] = 14004357016008534075;
  value[4][5] = 14936251243935649556;
  value[4][6] = 1522896783411827638;
  value[4][7] = 13858466054557097275;
  value[4][8] = 3172936841377972450;
  value[4][9] = 1068421630679369146;
  value[4][10] = 14424837255543781072;
  value[5][0] = 10714170731680699852;
  value[5][1] = 5765613494791770423;
  value[5][2] = 9663820292401160995;
  value[5][3] = 397172480378586284;
  value[5][4] = 4280709209124899452;
  value[5][5] = 1203358955785565947;
  value[5][6] = 11202700275482992172;
  value[5][7] = 13685583713509618195;
  value[5][8] = 3469864161577330170;
  value[5][9] = 8734130268423889220;
  value[5][10] = 16917450195693745928;
  value[6][0] = 8180410513952497551;
  value[6][1] = 7071292797447000945;
  value[6][2] = 14180677607572215618;
  value[6][3] = 6192821375005245090;
  value[6][4] = 11618722403488968531;
  value[6][5] = 16359132914868028498;
  value[6][6] = 629739239384523563;
  value[6][7] = 14807849520380455651;
  value[6][8] = 9453790714124186574;
  value[6][9] = 13094671554168529902;
  value[6][10] = 7712187332553607807;
  value[7][0] = 17023513964361815961;
  value[7][1] = 4047391151444874101;
  value[7][2] = 4322167285472126322;
  value[7][3] = 5857702128726293638;
  value[7][4] = 5139199894843344198;
  value[7][5] = 1693515656102034708;
  value[7][6] = 12470471516364544231;
  value[7][7] = 8323866952084077697;
  value[7][8] = 12651873977826689095;
  value[7][9] = 5067670011142229746;
  value[7][10] = 396279522907796927;
  value[8][0] = 16390401751368131934;
  value[8][1] = 7418420403566340092;
  value[8][2] = 8653653352406274042;
  value[8][3] = 4118931406823846491;
  value[8][4] = 82975984786450442;
  value[8][5] = 18222397316657226499;
  value[8][6] = 2002174628128864983;
  value[8][7] = 9634468324007960767;
  value[8][8] = 3259584970126823840;
  value[8][9] = 581370729274350312;
  value[8][10] = 17755967144133734705;
  value[9][0] = 9071247654034188589;
  value[9][1] = 6594541173975452315;
  value[9][2] = 17782188089785283344;
  value[9][3] = 3595742487221932055;
  value[9][4] = 9841642201692265487;
  value[9][5] = 1029671011456985627;
  value[9][6] = 13457875495926821529;
  value[9][7] = 6870405007338730846;
  value[9][8] = 12744130097658441846;
  value[9][9] = 6788288399186088634;
  value[9][10] = 357912856529587295;
  value[10][0] = 5607434777391338218;
  value[10][1] = 15814876086124552425;
  value[10][2] = 10566177234457318078;
  value[10][3] = 15354864780205183334;
  value[10][4] = 15216311397122257089;
  value[10][5] = 2674093911898978557;
  value[10][6] = 16268280753066444837;
  value[10][7] = 3675451000502615243;
  value[10][8] = 701273502091366776;
  value[10][9] = 15854278682598134666;
  value[10][10] = 6924615965242507246;
  value[11][0] = 1637471090675303584;
  value[11][1] = 4375318637115686030;
  value[11][2] = 12136810621975340177;
  value[11][3] = 105995675382122926;
  value[11][4] = 5987457663538146171;
  value[11][5] = 15717760330284389791;
  value[11][6] = 14670439359715404205;
  value[11][7] = 5464349733274908045;
  value[11][8] = 8636933789572244554;
  value[11][9] = 9769580318971544573;
  value[11][10] = 9102363839782539970;
  value[12][0] = 13571765139831017037;
  value[12][1] = 818883284762741475;
  value[12][2] = 11800681286871024320;
  value[12][3] = 4228007315495729552;
  value[12][4] = 9681067057645014410;
  value[12][5] = 10160317193366865607;
  value[12][6] = 7974952474492003064;
  value[12][7] = 311630947502800583;
  value[12][8] = 16977972518193735910;
  value[12][9] = 615971843838204966;
  value[12][10] = 17678304266887460895;
  value[13][0] = 12163901532241384359;
  value[13][1] = 5826724299253731684;
  value[13][2] = 17423022063725297026;
  value[13][3] = 18082834829462388363;
  value[13][4] = 10626880031407069622;
  value[13][5] = 1952478840402025861;
  value[13][6] = 9036125440908740987;
  value[13][7] = 1042941967034175129;
  value[13][8] = 13710136024884221835;
  value[13][9] = 3995229588248274477;
  value[13][10] = 11993482789377134210;
  value[14][0] = 12697151891341221277;
  value[14][1] = 13408757364964309332;
  value[14][2] = 14636730641620356003;
  value[14][3] = 2917199062768996165;
  value[14][4] = 11768157571822112934;
  value[14][5] = 15407074889369976729;
  value[14][6] = 3320959039775894817;
  value[14][7] = 16277817307991958146;
  value[14][8] = 7362033657200491320;
  value[14][9] = 9990801137147894185;
  value[14][10] = 14676096006818979429;
  value[15][0] = 17204396082766500862;
  value[15][1] = 14458712079049372979;
  value[15][2] = 17287567422807715153;
  value[15][3] = 13337198174858709409;
  value[15][4] = 7624105753184612060;
  value[15][5] = 17074874386857691157;
  value[15][6] = 2909991590741947335;
  value[15][7] = 14770785872198722410;
  value[15][8] = 17719065353010659993;
  value[15][9] = 14898159957685527729;
  value[15][10] = 12135206555549668255;
  value[16][0] = 15626888021543284549;
  value[16][1] = 12464927884746769804;
  value[16][2] = 1471467344747928256;
  value[16][3] = 11413582290460358915;
  value[16][4] = 9282109700482247280;
  value[16][5] = 17976144115670124039;
  value[16][6] = 16456828278798000758;
  value[16][7] = 1008181782916845414;
  value[16][8] = 17610348098917415827;
  value[16][9] = 204173067177706516;
  value[16][10] = 15964669298669259045;
  value[17][0] = 13932676290161493411;
  value[17][1] = 14699132604785301972;
  value[17][2] = 3744215611852980773;
  value[17][3] = 2709414263278899107;
  value[17][4] = 806263865491310800;
  value[17][5] = 7317365142041602481;
  value[17][6] = 16776386564962992796;
  value[17][7] = 11652640766067723448;
  value[17][8] = 1016370456237928832;
  value[17][9] = 961864172302955643;
  value[17][10] = 11539305592151691719;
  value[18][0] = 5260886902259565990;
  value[18][1] = 16171862215293778203;
  value[18][2] = 771114262717812991;
  value[18][3] = 10575516421403467499;
  value[18][4] = 13137658605724015568;
  value[18][5] = 4324696043571725046;
  value[18][6] = 17177140657993423090;
  value[18][7] = 11675287481120654357;
  value[18][8] = 215782959819461329;
  value[18][9] = 16817340479494209298;
  value[18][10] = 2305466969888960689;
  value[19][0] = 9354449820649144563;
  value[19][1] = 17638200638691477463;
  value[19][2] = 17096907883840532417;
  value[19][3] = 795566415402858691;
  value[19][4] = 12763188014703795610;
  value[19][5] = 2111548358776179736;
  value[19][6] = 7338420082729848069;
  value[19][7] = 11736253547470159946;
  value[19][8] = 11882449274483722406;
  value[19][9] = 13880779032198735515;
  value[19][10] = 12012886003476663648;
  value[20][0] = 9561079619973624339;
  value[20][1] = 3427032003991111411;
  value[20][2] = 16026109245305520857;
  value[20][3] = 842178779993054962;
  value[20][4] = 6620069080479782436;
  value[20][5] = 520632651104976912;
  value[20][6] = 5977708219320356796;
  value[20][7] = 14677035874152442976;
  value[20][8] = 12438555763140714832;
  value[20][9] = 10308634069667372976;
  value[20][10] = 1889137300031443018;
  value[21][0] = 4233023069765094533;
  value[21][1] = 11320301090717319475;
  value[21][2] = 529847152638273925;
  value[21][3] = 11362416581384070759;
  value[21][4] = 3913471784331119128;
  value[21][5] = 5817936720856651185;
  value[21][6] = 17448019282603275260;
  value[21][7] = 3425091249974323865;
  value[21][8] = 13157846471433414730;
  value[21][9] = 673370378535461536;
  value[21][10] = 846766219905577371;
  return value[i][j];
}
function FAST_PARTIAL_ROUND_VS(i, j) {
  var value[22][11];
  value[0][0] = 10702656082108580291;
  value[0][1] = 14323272843908492221;
  value[0][2] = 15449530374849795087;
  value[0][3] = 839422581341380592;
  value[0][4] = 11044529172588201887;
  value[0][5] = 9218907426627144627;
  value[0][6] = 16863852725141286670;
  value[0][7] = 12378944184369265821;
  value[0][8] = 4291107264489923137;
  value[0][9] = 18105902022777689401;
  value[0][10] = 4532874245444204412;
  value[1][0] = 783331064993138470;
  value[1][1] = 11780280264626300249;
  value[1][2] = 14317347280917240576;
  value[1][3] = 7639896796391275580;
  value[1][4] = 5524721098652169327;
  value[1][5] = 4647621086109661393;
  value[1][6] = 551557749415629519;
  value[1][7] = 4774730083352601242;
  value[1][8] = 9878226461889807280;
  value[1][9] = 2796688701546052437;
  value[1][10] = 3152254583822593203;
  value[2][0] = 5195684422952000615;
  value[2][1] = 16386310079584461432;
  value[2][2] = 8354845848262314988;
  value[2][3] = 6700373425673846218;
  value[2][4] = 14613275276996917774;
  value[2][5] = 15810393896142816349;
  value[2][6] = 8919907675614209581;
  value[2][7] = 4378937399360000942;
  value[2][8] = 3921314266986613083;
  value[2][9] = 3157453341478075556;
  value[2][10] = 12056705871081879759;
  value[3][0] = 12838957912943317144;
  value[3][1] = 11392036161259909092;
  value[3][2] = 5420611346845318460;
  value[3][3] = 11418874531271499277;
  value[3][4] = 14582096517505941837;
  value[3][5] = 877280106856758747;
  value[3][6] = 11091271673331452926;
  value[3][7] = 9617340340155417663;
  value[3][8] = 9043411348035541157;
  value[3][9] = 16964047224456307403;
  value[3][10] = 10338102439110648229;
  value[4][0] = 1277502887239453738;
  value[4][1] = 11492475458589769996;
  value[4][2] = 12115111105137538533;
  value[4][3] = 6007394463725400498;
  value[4][4] = 4633777909023327008;
  value[4][5] = 12045217224929432404;
  value[4][6] = 5600645681481758769;
  value[4][7] = 13058511211226185597;
  value[4][8] = 10831228388201534917;
  value[4][9] = 10765285645335338967;
  value[4][10] = 12314041551985486068;
  value[5][0] = 4032097614937144430;
  value[5][1] = 5682426829072761065;
  value[5][2] = 14144004233890775432;
  value[5][3] = 11476034762570105656;
  value[5][4] = 11441392943423295273;
  value[5][5] = 14245661866930276468;
  value[5][6] = 11536287954985758398;
  value[5][7] = 6483617259986966714;
  value[5][8] = 10087111781120039554;
  value[5][9] = 13728844829744097141;
  value[5][10] = 14679689325173586623;
  value[6][0] = 6304928008866363842;
  value[6][1] = 9855321538770560945;
  value[6][2] = 9435164398075715846;
  value[6][3] = 9404592978128123150;
  value[6][4] = 11002422368171462947;
  value[6][5] = 8486311906590791617;
  value[6][6] = 18361824531704888434;
  value[6][7] = 2798920999004265189;
  value[6][8] = 17909793464802401204;
  value[6][9] = 5756303597132403312;
  value[6][10] = 5858421860645672190;
  value[7][0] = 17305709116193116427;
  value[7][1] = 735829306202841815;
  value[7][2] = 14847743950994388316;
  value[7][3] = 11139080626411756670;
  value[7][4] = 7092455469264931963;
  value[7][5] = 11583767394161657005;
  value[7][6] = 15774934118411863340;
  value[7][7] = 4416857554682544229;
  value[7][8] = 9159855784268361426;
  value[7][9] = 8216101670692368083;
  value[7][10] = 16367782717227750410;
  value[8][0] = 12329937970340684597;
  value[8][1] = 10602297383654186753;
  value[8][2] = 5891764497626072293;
  value[8][3] = 10671154149112267313;
  value[8][4] = 18234822653119242373;
  value[8][5] = 15287378323692558105;
  value[8][6] = 9967103142034849899;
  value[8][7] = 15861939895842675328;
  value[8][8] = 11730063476303470848;
  value[8][9] = 1586390848658847158;
  value[8][10] = 1015360682565850373;
  value[9][0] = 4417656488067463062;
  value[9][1] = 14987770745080868386;
  value[9][2] = 4702825855063868377;
  value[9][3] = 2465246157933796197;
  value[9][4] = 8034369030882576822;
  value[9][5] = 15698764330557579947;
  value[9][6] = 11839103375501390181;
  value[9][7] = 4595990697051972631;
  value[9][8] = 14148213542088135280;
  value[9][9] = 14849248616009699298;
  value[9][10] = 15807262764748562013;
  value[10][0] = 1262098398535043837;
  value[10][1] = 2436065499532941641;
  value[10][2] = 1138970283407778564;
  value[10][3] = 1825502889302643134;
  value[10][4] = 5500855066099563465;
  value[10][5] = 11666892062115297604;
  value[10][6] = 13463068267332421729;
  value[10][7] = 17516970128403465337;
  value[10][8] = 11088428730628824449;
  value[10][9] = 4615288675764694853;
  value[10][10] = 16220123440754855385;
  value[11][0] = 9570691013274316785;
  value[11][1] = 15613851939195720118;
  value[11][2] = 3699802456427549428;
  value[11][3] = 14363933592354809237;
  value[11][4] = 13863573127618181752;
  value[11][5] = 11428524752427198786;
  value[11][6] = 1512236798846210343;
  value[11][7] = 15492557605200192531;
  value[11][8] = 4471766256042329601;
  value[11][9] = 12055723375080267479;
  value[11][10] = 16720313860519281958;
  value[12][0] = 2561042796132833389;
  value[12][1] = 10464014529858294964;
  value[12][2] = 14401165907148431066;
  value[12][3] = 2413453332765052361;
  value[12][4] = 14620959153325857181;
  value[12][5] = 16368665425253279930;
  value[12][6] = 8913590094823920770;
  value[12][7] = 4357291993877750483;
  value[12][8] = 18315259589408480902;
  value[12][9] = 7040130461852977952;
  value[12][10] = 16913088801316332783;
  value[13][0] = 15483762529902925134;
  value[13][1] = 17034733783218795199;
  value[13][2] = 18136305076967260316;
  value[13][3] = 15896912869485945382;
  value[13][4] = 475392759889361288;
  value[13][5] = 1823867867187688822;
  value[13][6] = 8817375076608676110;
  value[13][7] = 8857453095514132937;
  value[13][8] = 17995601973761478278;
  value[13][9] = 18042919419769033432;
  value[13][10] = 17356815683605755783;
  value[14][0] = 853567178463642200;
  value[14][1] = 781481719657018312;
  value[14][2] = 864881582238738022;
  value[14][3] = 776585443674182031;
  value[14][4] = 868289454518583667;
  value[14][5] = 873991676947315745;
  value[14][6] = 825112067366636056;
  value[14][7] = 904067466148006484;
  value[14][8] = 864277137123579536;
  value[14][9] = 785755357347442049;
  value[14][10] = 861609966041484849;
  value[15][0] = 3644417860664408;
  value[15][1] = 3335591043919560;
  value[15][2] = 3691922388548390;
  value[15][3] = 3315658209334511;
  value[15][4] = 3706319247139923;
  value[15][5] = 3730913850857153;
  value[15][6] = 3522914930316824;
  value[15][7] = 3859199185371348;
  value[15][8] = 3689373458353040;
  value[15][9] = 3354664939836449;
  value[15][10] = 3677753419960785;
  value[16][0] = 15551163980504;
  value[16][1] = 14240130616264;
  value[16][2] = 15771333781862;
  value[16][3] = 14149230256207;
  value[16][4] = 15820017123763;
  value[16][5] = 15936503968609;
  value[16][6] = 15031975505304;
  value[16][7] = 16471548413268;
  value[16][8] = 15760188783376;
  value[16][9] = 14317015483073;
  value[16][10] = 15696239618801;
  value[17][0] = 66326084760;
  value[17][1] = 60935297352;
  value[17][2] = 67215299046;
  value[17][3] = 60348857903;
  value[17][4] = 67671686739;
  value[17][5] = 67914356993;
  value[17][6] = 64112320984;
  value[17][7] = 70469953364;
  value[17][8] = 67111186256;
  value[17][9] = 61118430945;
  value[17][10] = 67182327505;
  value[18][0] = 286463800;
  value[18][1] = 257349000;
  value[18][2] = 285544326;
  value[18][3] = 260345679;
  value[18][4] = 286599123;
  value[18][5] = 289630625;
  value[18][6] = 275722040;
  value[18][7] = 300075668;
  value[18][8] = 285878768;
  value[18][9] = 262796737;
  value[18][10] = 284566993;
  value[19][0] = 1177368;
  value[19][1] = 1095368;
  value[19][2] = 1264278;
  value[19][3] = 1101695;
  value[19][4] = 1199363;
  value[19][5] = 1308833;
  value[19][6] = 1145944;
  value[19][7] = 1256596;
  value[19][8] = 1265600;
  value[19][9] = 1089681;
  value[19][10] = 1214817;
  value[20][0] = 4864;
  value[20][1] = 5968;
  value[20][2] = 4430;
  value[20][3] = 4895;
  value[20][4] = 5755;
  value[20][5] = 4977;
  value[20][6] = 4656;
  value[20][7] = 6188;
  value[20][8] = 4968;
  value[20][9] = 3889;
  value[20][10] = 5577;
  value[21][0] = 20;
  value[21][1] = 34;
  value[21][2] = 18;
  value[21][3] = 39;
  value[21][4] = 13;
  value[21][5] = 13;
  value[21][6] = 28;
  value[21][7] = 2;
  value[21][8] = 16;
  value[21][9] = 41;
  value[21][10] = 15;
  return value[i][j];
}
function FAST_PARTIAL_ROUND_INITIAL_MATRIX(i, j) {
  var value[11][11];
  value[0][0] = 9256917872013944843;
  value[0][1] = 15893897022228540664;
  value[0][2] = 13949760578536372653;
  value[0][3] = 10441609312974976515;
  value[0][4] = 4189528951266599854;
  value[0][5] = 45832257923618046;
  value[0][6] = 8607345711887993138;
  value[0][7] = 10398036555777403988;
  value[0][8] = 13806692727776539476;
  value[0][9] = 4187764176355919243;
  value[0][10] = 4771889745340348367;
  value[1][0] = 16687757000829461707;
  value[1][1] = 12764541860482007578;
  value[1][2] = 1073506034073544330;
  value[1][3] = 12178624353196374758;
  value[1][4] = 9093834777404014814;
  value[1][5] = 12470775297641857694;
  value[1][6] = 14365012582629183475;
  value[1][7] = 17322896464470575084;
  value[1][8] = 12929063850085080619;
  value[1][9] = 8008291477586393637;
  value[1][10] = 4187764176355919243;
  value[2][0] = 15919568759443364026;
  value[2][1] = 1487496629277845135;
  value[2][2] = 5122203447763166523;
  value[2][3] = 2200314810679404686;
  value[2][4] = 13521131922395904812;
  value[2][5] = 16674096007358536750;
  value[2][6] = 12650089191056401741;
  value[2][7] = 15914053419498374975;
  value[2][8] = 14774060794419120357;
  value[2][9] = 12929063850085080619;
  value[2][10] = 13806692727776539476;
  value[3][0] = 17628276356247382281;
  value[3][1] = 14211060579632108547;
  value[3][2] = 9180588347636943785;
  value[3][3] = 11858291964101661402;
  value[3][4] = 3422342838493228737;
  value[3][5] = 16717315056857949245;
  value[3][6] = 4874593437852546498;
  value[3][7] = 14575430061120165237;
  value[3][8] = 15914053419498374975;
  value[3][9] = 17322896464470575084;
  value[3][10] = 10398036555777403988;
  value[4][0] = 17976887162229714000;
  value[4][1] = 6791692987299703477;
  value[4][2] = 6455531853563710059;
  value[4][3] = 506729933833272474;
  value[4][4] = 12479288794463684010;
  value[4][5] = 12357738834545821552;
  value[4][6] = 14664271473160014313;
  value[4][7] = 4874593437852546498;
  value[4][8] = 12650089191056401741;
  value[4][9] = 14365012582629183475;
  value[4][10] = 8607345711887993138;
  value[5][0] = 9191356322801962495;
  value[5][1] = 5412105005886646653;
  value[5][2] = 7077135177323540712;
  value[5][3] = 13768926573657667599;
  value[5][4] = 14009018616032686342;
  value[5][5] = 8447498431838444578;
  value[5][6] = 12357738834545821552;
  value[5][7] = 16717315056857949245;
  value[5][8] = 16674096007358536750;
  value[5][9] = 12470775297641857694;
  value[5][10] = 45832257923618046;
  value[6][0] = 8244675934684975988;
  value[6][1] = 2125569474183208192;
  value[6][2] = 1761883289931249101;
  value[6][3] = 9202082607097456696;
  value[6][4] = 9665676628089346926;
  value[6][5] = 14009018616032686342;
  value[6][6] = 12479288794463684010;
  value[6][7] = 3422342838493228737;
  value[6][8] = 13521131922395904812;
  value[6][9] = 9093834777404014814;
  value[6][10] = 4189528951266599854;
  value[7][0] = 7268127472833019981;
  value[7][1] = 5600686741053600354;
  value[7][2] = 13703919263985638019;
  value[7][3] = 155673126466762010;
  value[7][4] = 9202082607097456696;
  value[7][5] = 13768926573657667599;
  value[7][6] = 506729933833272474;
  value[7][7] = 11858291964101661402;
  value[7][8] = 2200314810679404686;
  value[7][9] = 12178624353196374758;
  value[7][10] = 10441609312974976515;
  value[8][0] = 9602108300053878928;
  value[8][1] = 15610298188943525805;
  value[8][2] = 13828402413953013890;
  value[8][3] = 13703919263985638019;
  value[8][4] = 1761883289931249101;
  value[8][5] = 7077135177323540712;
  value[8][6] = 6455531853563710059;
  value[8][7] = 9180588347636943785;
  value[8][8] = 5122203447763166523;
  value[8][9] = 1073506034073544330;
  value[8][10] = 13949760578536372653;
  value[9][0] = 1540311261654516052;
  value[9][1] = 10517970165082627573;
  value[9][2] = 15610298188943525805;
  value[9][3] = 5600686741053600354;
  value[9][4] = 2125569474183208192;
  value[9][5] = 5412105005886646653;
  value[9][6] = 6791692987299703477;
  value[9][7] = 14211060579632108547;
  value[9][8] = 1487496629277845135;
  value[9][9] = 12764541860482007578;
  value[9][10] = 15893897022228540664;
  value[10][0] = 15582992301522062240;
  value[10][1] = 1540311261654516052;
  value[10][2] = 9602108300053878928;
  value[10][3] = 7268127472833019981;
  value[10][4] = 8244675934684975988;
  value[10][5] = 9191356322801962495;
  value[10][6] = 17976887162229714000;
  value[10][7] = 17628276356247382281;
  value[10][8] = 15919568759443364026;
  value[10][9] = 16687757000829461707;
  value[10][10] = 9256917872013944843;
  return value[i][j];
}
function FAST_PARTIAL_ROUND_CONSTANTS(i) {
  var value[22];
  value[0] = 8415871462856204715;
  value[1] = 15156192896528938595;
  value[2] = 7115538620563575164;
  value[3] = 15396535437187948468;
  value[4] = 13402196712199986140;
  value[5] = 16375052485106733288;
  value[6] = 1054611198573910171;
  value[7] = 14596485233396387590;
  value[8] = 13680159589485875108;
  value[9] = 9441690674504273278;
  value[10] = 7281057872237841107;
  value[11] = 8581622869689923244;
  value[12] = 12649521141086658944;
  value[13] = 13316298133620363637;
  value[14] = 10757436128916982213;
  value[15] = 16047932205709436219;
  value[16] = 17301616663694082334;
  value[17] = 11667617191967502297;
  value[18] = 9658934864843380542;
  value[19] = 3498090033303964622;
  value[20] = 1930488375833774198;
  value[21] = 0;
  return value[i];
}
function FAST_PARTIAL_FIRST_ROUND_CONSTANT(i) {
  var value[12];
  value[0] = 4378616569090929672;
  value[1] = 16831074976302798833;
  value[2] = 17474843094576853935;
  value[3] = 15154628183104001226;
  value[4] = 14219868664549115443;
  value[5] = 10509321604391016962;
  value[6] = 17545903601470498427;
  value[7] = 3273629310481947241;
  value[8] = 8362887214150162593;
  value[9] = 7587761356207546181;
  value[10] = 6959023468757315912;
  value[11] = 14065947794859331340;
  return value[i];
}
function MDS_MATRIX_CIRC(i) {
  var mds[12];
  mds[0] = 17;
  mds[1] = 15;
  mds[2] = 41;
  mds[3] = 16;
  mds[4] = 2;
  mds[5] = 28;
  mds[6] = 13;
  mds[7] = 13;
  mds[8] = 39;
  mds[9] = 18;
  mds[10] = 34;
  mds[11] = 20;
  return mds[i];
}
function MDS_MATRIX_DIAG(i) {
  var mds[12];
  mds[0] = 8;
  mds[1] = 0;
  mds[2] = 0;
  mds[3] = 0;
  mds[4] = 0;
  mds[5] = 0;
  mds[6] = 0;
  mds[7] = 0;
  mds[8] = 0;
  mds[9] = 0;
  mds[10] = 0;
  mds[11] = 0;
  return mds[i];
}
