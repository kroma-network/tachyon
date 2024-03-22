pragma circom 2.0.0;

include "external/kroma_network_circomlib/circuits/bitify.circom";
include "external/kroma_network_circomlib/circuits/binsum.circom";

template Adder() {
    signal input a;
    signal input b;
    signal output out;

    var i;

    component n2ba = Num2Bits(32);
    component n2bb = Num2Bits(32);
    component sum = BinSum(32, 2);
    component b2n = Bits2Num(32);

    n2ba.in <== a;
    n2bb.in <== b;

    for (i = 0; i < 32; i++) {
        sum.in[0][i] <== n2ba.out[i];
        sum.in[1][i] <== n2bb.out[i];
    }

    for (i = 0; i < 32; i++) {
        b2n.in[i] <== sum.out[i];
    }

    out <== b2n.out;
}

component main = Adder();
