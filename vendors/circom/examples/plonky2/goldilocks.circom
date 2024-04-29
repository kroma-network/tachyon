pragma circom 2.1.0;
include "./constants.circom";

// Range check
// Verifies x < 1 << N
template LessNBits(N) {
  signal input x;
  var e2 = 1;
  signal tmp1[N];
  signal tmp2[N + 1];
  tmp2[0] <== 0;
  for (var i = 0; i < N; i++) {
    tmp1[i] <-- (x >> i) & 1;
    tmp1[i] * (tmp1[i] - 1) === 0;
    tmp2[i + 1] <== tmp1[i] * e2 + tmp2[i];
    e2 = e2 + e2;
  }
  x === tmp2[N];
}

// Gl: Goldilocks
// range check d < 1 << N
template GlReduce(N) {
  signal input x;
  signal output out;

  var r = x % Order();
  var d = (x - r) \ Order();
  out <-- r;
  signal tmp0 <-- d;
  tmp0 * Order() + out === x;

  component c0 = LessNBits(N);
  c0.x <== tmp0;
  component c1 = LessNBits(64);
  c1.x <== out;
}

template GlAdd() {
  signal input a;
  signal input b;
  signal output out;

  component cr = GlReduce(1);
  cr.x <== a + b;
  out <== cr.out;
}

template GlSub() {
  signal input a;
  signal input b;
  signal output out;

  component cr = GlReduce(1);
  cr.x <== a + Order() - b;
  out <== cr.out;
}

template GlMul() {
  signal input a;
  signal input b;
  signal output out;

  component cr = GlReduce(64);
  cr.x <== a * b;
  out <== cr.out;
}

function gl_inverse(x) {
  assert(x != 0);
  var m = Order() - 2;
  var e2 = x;
  var res = 1;
  for (var i = 0; i < 64; i++) {
    if ((m >> i) & 1 == 1) {
      res *= e2;
      res %= Order();
    }
    e2 *= e2;
    e2 %= Order();
  }
  return res;
}

template GlInv() {
  signal input x;
  signal output out;
  out <-- gl_inverse(x);
  component check = GlMul();
  check.a <== x;
  check.b <== out;
}

template GlDiv() {
  signal input a;
  signal input b;
  signal output out;

  component inv_b = GlInv();
  inv_b.x <== b;
  component a_mul_inv_b = GlMul();
  a_mul_inv_b.a <== a;
  a_mul_inv_b.b <== inv_b.out;
  out <== a_mul_inv_b.out;
}

// bit = x & 1
// out = x >> 1
template RShift1() {
  signal input x;
  signal output out;
  signal output bit;

  var o = x >> 1;
  out <-- o;
  bit <== x - out * 2;
  bit * (1 - bit) === 0;
}

// out = x >> n
// where n < N
template RShift(N) {
  signal input x;
  signal output out;
  assert(N < 255);

  out <-- x >> N;
  signal y <-- out << N;
  signal r <== x - y;
  out * 2 ** N === y;

  component c = LessNBits(N);
  c.x <== r;
}

template GlExp() {
  signal input x;
  signal input n;
  signal output out;

  signal e2[65];
  signal mul[65];
  component rshift1[64];
  component cmul[64][2];
  e2[0] <== x;
  mul[0] <== 1;
  rshift1[0] = RShift1();
  rshift1[0].x <== n;
  for (var i = 0; i < 64; i++) {
    if (i > 0) {
      rshift1[i] = RShift1();
      rshift1[i].x <== rshift1[i - 1].out;
    }

    cmul[i][0] = GlMul();
    cmul[i][1] = GlMul();
    cmul[i][0].a <== mul[i];
    cmul[i][0].b <== e2[i] * rshift1[i].bit + 1 - rshift1[i].bit;
    cmul[i][1].a <== e2[i];
    cmul[i][1].b <== e2[i];

    mul[i + 1] <== cmul[i][0].out;
    e2[i + 1] <== cmul[i][1].out;
  }

  out <== mul[64];
}

template GlExpPowerOf2(N) {
  signal input x;
  signal output out;
  component mul[N];
  mul[0] = GlMul();
  mul[0].a <== x;
  mul[0].b <== x;
  for (var i = 1; i < N; i++) {
    mul[i] = GlMul();
    mul[i].a <== mul[i - 1].out;
    mul[i].b <== mul[i - 1].out;
  }
  out <== mul[N - 1].out;
}

// input: 10011 (N = 5)
// output: 11001
template ReverseBits(N) {
  signal input x;
  signal output out;
  component rshift[N];
  signal tmp[N];

  for (var i = 0; i < N; i++) {
    rshift[i] = RShift1();
    if (i == 0) {
      rshift[0].x <== x;
    } else {
      rshift[i].x <== rshift[i - 1].out;
    }

    if (i == 0) {
      tmp[i] <== rshift[i].bit;
    } else {
      tmp[i] <== rshift[i].bit + 2 * tmp[i - 1];
    }
  }
  out <== tmp[N - 1];
}

// input: 10011 (N = 3)
// output: 011
template LastNBits(N) {
  signal input x;
  signal output out;
  component rshift[N];
  signal tmp[N];

  for (var i = 0; i < N; i++) {
    rshift[i] = RShift1();
    if (i == 0) {
      rshift[0].x <== x;
    } else {
      rshift[i].x <== rshift[i - 1].out;
    }

    if (i == 0) {
      tmp[i] <== rshift[i].bit;
    } else {
      tmp[i] <== 2**i * rshift[i].bit + tmp[i - 1];
    }
  }
  out <== tmp[N - 1];
}
