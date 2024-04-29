pragma circom 2.1.0;
include "external/kroma_network_circomlib/circuits/comparators.circom";

template Reduce(N) {
  signal input in[N][2];
  signal input alpha[2];
  signal input old_eval[2];
  signal output out[2];

  component add[N];
  component mul[N];

  for (var i = N; i > 0; i--) {
    add[i - 1] = GlExtAdd();
    mul[i - 1] = GlExtMul();
    if (i == N) {
      mul[i - 1].a[0] <== old_eval[0];
      mul[i - 1].a[1] <== old_eval[1];
    } else {
      mul[i - 1].a[0] <== add[i].out[0];
      mul[i - 1].a[1] <== add[i].out[1];
    }
    mul[i - 1].b[0] <== alpha[0];
    mul[i - 1].b[1] <== alpha[1];

    add[i - 1].a[0] <== in[i - 1][0];
    add[i - 1].a[1] <== in[i - 1][1];
    add[i - 1].b[0] <== mul[i - 1].out[0];
    add[i - 1].b[1] <== mul[i - 1].out[1];
  }

  out[0] <== add[0].out[0];
  out[1] <== add[0].out[1];
}

// A working but slow implementation
template RandomAccess(N) {
  signal input a[N];
  signal input idx;
  signal output out;

  component cIsEqual[N];
  signal sum[N + 1];
  sum[0] <== 0;
  for (var i = 0; i < N; i++) {
    cIsEqual[i] = IsEqual();
    cIsEqual[i].in[0] <== idx;
    cIsEqual[i].in[1] <== i;
    sum[i + 1] <== cIsEqual[i].out * a[i] + sum[i];
  }

  out <== sum[N];
}

template RandomAccess2(N, M) {
  signal input a[N][M];
  signal input idx;
  signal output out[M];

  component ra[M];
  for (var i = 0; i < M; i++) {
    ra[i] = RandomAccess(N);
    ra[i].idx <== idx;
    for (var j = 0; j < N; j++) {
      ra[i].a[j] <== a[j][i];
    }
    out[i] <== ra[i].out;
  }
}
