pragma circom 2.1.0;
include "./constants.circom";
include "./goldilocks.circom";
include "external/kroma_network_circomlib/circuits/bitify.circom";

// GlExt: Goldilocks's quadratic extension
template GlExtAdd() {
  signal input a[2];
  signal input b[2];
  signal output out[2];

  component add[2];
  for (var i = 0; i < 2; i++) {
    add[i] = GlAdd();
    add[i].a <== a[i];
    add[i].b <== b[i];
    out[i] <== add[i].out;
  }
}

template GlExtSub() {
  signal input a[2];
  signal input b[2];
  signal output out[2];

  component cadd[2];
  for (var i = 0; i < 2; i++) {
    cadd[i] = GlAdd();
    cadd[i].a <== a[i];
    cadd[i].b <== Order() - b[i];
    out[i] <== cadd[i].out;
  }
}

// out[0] = a[0] * b[0] + W * a[1] * b[1]
// out[1] = a[0] * b[1] + a[1] * b[0]
template GlExtMul() {
  signal input a[2];
  signal input b[2];
  signal output out[2];

  component a0_mul_b0 = GlMul();
  component a1_mul_W = GlMul();
  component a1_mul_W_mul_b1 = GlMul();
  component a0_mul_b1 = GlMul();
  component a1_mul_b0 = GlMul();
  component cadd0 = GlAdd();
  component cadd1 = GlAdd();

  a0_mul_b0.a <== a[0];
  a0_mul_b0.b <== b[0];
  a1_mul_W.a <== a[1];
  a1_mul_W.b <== W();
  a1_mul_W_mul_b1.a <== a1_mul_W.out;
  a1_mul_W_mul_b1.b <== b[1];
  a0_mul_b1.a <== a[0];
  a0_mul_b1.b <== b[1];
  a1_mul_b0.a <== a[1];
  a1_mul_b0.b <== b[0];

  cadd0.a <== a0_mul_b0.out;
  cadd0.b <== a1_mul_W_mul_b1.out;
  cadd1.a <== a0_mul_b1.out;
  cadd1.b <== a1_mul_b0.out;

  out[0] <== cadd0.out;
  out[1] <== cadd1.out;
}

// out[0] = a[0] * a[0] + W * a[1] * a[1]
// out[1] = 2 * a[0] * a[1]
template GlExtSquare() {
  signal input a[2];
  signal output out[2];

  component a0_square = GlMul();
  component a1_square = GlMul();
  component W_a1_square = GlMul();
  component a0_mul_a1 = GlMul();
  component a0_mul_a1_mul_2 = GlMul();
  component a0_square_add_W_a1_square = GlAdd();

  a0_square.a <== a[0];
  a0_square.b <== a[0];
  a1_square.a <== a[1];
  a1_square.b <== a[1];
  W_a1_square.a <== W();
  W_a1_square.b <== a1_square.out;
  a0_mul_a1.a <== a[0];
  a0_mul_a1.b <== a[1];
  a0_mul_a1_mul_2.a <== a0_mul_a1.out;
  a0_mul_a1_mul_2.b <== 2;

  a0_square_add_W_a1_square.a <== a0_square.out;
  a0_square_add_W_a1_square.b <== W_a1_square.out;

  out[0] <== a0_square_add_W_a1_square.out;
  out[1] <== a0_mul_a1_mul_2.out;
}

template GlExtDiv() {
  signal input a[2];
  signal input b[2];
  signal output out[2];

  component cextmul0 = GlExtMul();
  cextmul0.a[0] <== b[0];
  component b1_mul_d = GlMul();
  b1_mul_d.a <== b[1];
  b1_mul_d.b <== DTH_ROOT();
  cextmul0.a[1] <== b1_mul_d.out;
  cextmul0.b[0] <== b[0];
  cextmul0.b[1] <== b[1];

  component cinv = GlInv();
  cinv.x <== cextmul0.out[0];
  signal inv_out0 <== cinv.out;

  component cextmul1 = GlExtMul();
  cextmul1.a[0] <== a[0];
  cextmul1.a[1] <== a[1];

  component b0_mul_inv = GlMul();
  component b1_mul_d_mul_inv = GlMul();
  b0_mul_inv.a <== b[0];
  b0_mul_inv.b <== inv_out0;
  b1_mul_d_mul_inv.a <== b1_mul_d.out;
  b1_mul_d_mul_inv.b <== inv_out0;
  cextmul1.b[0] <== b0_mul_inv.out;
  cextmul1.b[1] <== b1_mul_d_mul_inv.out;

  out[0] <== cextmul1.out[0];
  out[1] <== cextmul1.out[1];
}

// n < 1 << N
template GlExtExpN(N) {
  signal input x[2];
  signal input n;
  signal output out[2];

  component cbits = Num2Bits(N);
  cbits.in <== n;
  signal e2[N][2];
  component cextmul[N];
  component cdouble[N];
  for (var i = 0; i < N; i++) {
    cextmul[i] = GlExtMul();
    cdouble[i] = GlExtSquare();
  }
  cextmul[0].a[0] <== 1;
  cextmul[0].a[1] <== 0;
  e2[0][0] <== x[0];
  e2[0][1] <== x[1];

  for (var i = 0; i < N - 1; i++) {
    cextmul[i].b[0] <== e2[i][0] * cbits.out[i] + 1 - cbits.out[i];
    cextmul[i].b[1] <== e2[i][1] * cbits.out[i];

    cextmul[i + 1].a[0] <== cextmul[i].out[0];
    cextmul[i + 1].a[1] <== cextmul[i].out[1];

    cdouble[i].a[0] <== e2[i][0];
    cdouble[i].a[1] <== e2[i][1];
    e2[i + 1][0] <== cdouble[i].out[0];
    e2[i + 1][1] <== cdouble[i].out[1];
  }

  cextmul[N - 1].b[0] <== e2[N - 1][0] * cbits.out[N - 1] + 1 - cbits.out[N - 1];
  cextmul[N - 1].b[1] <== e2[N - 1][1] * cbits.out[N - 1];

  cdouble[N-1].a[0] <== 0;
  cdouble[N-1].a[1] <== 0;

  out[0] <== cextmul[N - 1].out[0];
  out[1] <== cextmul[N - 1].out[1];
}

template GlExtExp() {
  signal input x[2];
  signal input n;
  signal output out[2];
  out <== GlExtExpN(64)(x, n);
}

template GlExtExpPowerOf2(N) {
  signal input x[2];
  signal output out[2];
  component mul[N];
  mul[0] = GlExtMul();
  mul[0].a <== x;
  mul[0].b <== x;
  for (var i = 1; i < N; i++) {
    mul[i] = GlExtMul();
    mul[i].a <== mul[i - 1].out;
    mul[i].b <== mul[i - 1].out;
  }
  out <== mul[N - 1].out;
}

template GlExtScalarMul() {
  signal input x[2];
  signal input a;
  signal output out[2];
  out[0] <== GlMul()(x[0], a);
  out[1] <== GlMul()(x[1], a);
}

template GlExt(x, y) {
  signal output out[2];
  out[0] <== x;
  out[1] <== y;
}
