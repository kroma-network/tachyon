pragma circom 2.1.0;
include "./goldilocks.circom";
include "external/kroma_network_circomlib/circuits/poseidon.circom";
include "external/kroma_network_circomlib/circuits/bitify.circom";

function GL_CONST(i) {
    var const[30*12] = [
        0xb585f766f2144405, 0x7746a55f43921ad7, 0xb2fb0d31cee799b4, 0x0f6760a4803427d7,
        0xe10d666650f4e012, 0x8cae14cb07d09bf1, 0xd438539c95f63e9f, 0xef781c7ce35b4c3d,
        0xcdc4a239b0c44426, 0x277fa208bf337bff, 0xe17653a29da578a1, 0xc54302f225db2c76,
        0x86287821f722c881, 0x59cd1a8a41c18e55, 0xc3b919ad495dc574, 0xa484c4c5ef6a0781,
        0x308bbd23dc5416cc, 0x6e4a40c18f30c09c, 0x9a2eedb70d8f8cfa, 0xe360c6e0ae486f38,
        0xd5c7718fbfc647fb, 0xc35eae071903ff0b, 0x849c2656969c4be7, 0xc0572c8c08cbbbad,
        0xe9fa634a21de0082, 0xf56f6d48959a600d, 0xf7d713e806391165, 0x8297132b32825daf,
        0xad6805e0e30b2c8a, 0xac51d9f5fcf8535e, 0x502ad7dc18c2ad87, 0x57a1550c110b3041,
        0x66bbd30e6ce0e583, 0x0da2abef589d644e, 0xf061274fdb150d61, 0x28b8ec3ae9c29633,
        0x92a756e67e2b9413, 0x70e741ebfee96586, 0x019d5ee2af82ec1c, 0x6f6f2ed772466352,
        0x7cf416cfe7e14ca1, 0x61df517b86a46439, 0x85dc499b11d77b75, 0x4b959b48b9c10733,
        0xe8be3e5da8043e57, 0xf5c0bc1de6da8699, 0x40b12cbf09ef74bf, 0xa637093ecb2ad631,
        0x3cc3f892184df408, 0x2e479dc157bf31bb, 0x6f49de07a6234346, 0x213ce7bede378d7b,
        0x5b0431345d4dea83, 0xa2de45780344d6a1, 0x7103aaf94a7bf308, 0x5326fc0d97279301,
        0xa9ceb74fec024747, 0x27f8ec88bb21b1a3, 0xfceb4fda1ded0893, 0xfac6ff1346a41675,
        0x7131aa45268d7d8c, 0x9351036095630f9f, 0xad535b24afc26bfb, 0x4627f5c6993e44be,
        0x645cf794b8f1cc58, 0x241c70ed0af61617, 0xacb8e076647905f1, 0x3737e9db4c4f474d,
        0xe7ea5e33e75fffb6, 0x90dee49fc9bfc23a, 0xd1b1edf76bc09c92, 0x0b65481ba645c602,
        0x99ad1aab0814283b, 0x438a7c91d416ca4d, 0xb60de3bcc5ea751c, 0xc99cab6aef6f58bc,
        0x69a5ed92a72ee4ff, 0x5e7b329c1ed4ad71, 0x5fc0ac0800144885, 0x32db829239774eca,
        0x0ade699c5830f310, 0x7cc5583b10415f21, 0x85df9ed2e166d64f, 0x6604df4fee32bcb1,
        0xeb84f608da56ef48, 0xda608834c40e603d, 0x8f97fe408061f183, 0xa93f485c96f37b89,
        0x6704e8ee8f18d563, 0xcee3e9ac1e072119, 0x510d0e65e2b470c1, 0xf6323f486b9038f0,
        0x0b508cdeffa5ceef, 0xf2417089e4fb3cbd, 0x60e75c2890d15730, 0xa6217d8bf660f29c,
        0x7159cd30c3ac118e, 0x839b4e8fafead540, 0x0d3f3e5e82920adc, 0x8f7d83bddee7bba8,
        0x780f2243ea071d06, 0xeb915845f3de1634, 0xd19e120d26b6f386, 0x016ee53a7e5fecc6,
        0xcb5fd54e7933e477, 0xacb8417879fd449f, 0x9c22190be7f74732, 0x5d693c1ba3ba3621,
        0xdcef0797c2b69ec7, 0x3d639263da827b13, 0xe273fd971bc8d0e7, 0x418f02702d227ed5,
        0x8c25fda3b503038c, 0x2cbaed4daec8c07c, 0x5f58e6afcdd6ddc2, 0x284650ac5e1b0eba,
        0x635b337ee819dab5, 0x9f9a036ed4f2d49f, 0xb93e260cae5c170e, 0xb0a7eae879ddb76d,
        0xd0762cbc8ca6570c, 0x34c6efb812b04bf5, 0x40bf0ab5fa14c112, 0xb6b570fc7c5740d3,
        0x5a27b9002de33454, 0xb1a5b165b6d2b2d2, 0x8722e0ace9d1be22, 0x788ee3b37e5680fb,
        0x14a726661551e284, 0x98b7672f9ef3b419, 0xbb93ae776bb30e3a, 0x28fd3b046380f850,
        0x30a4680593258387, 0x337dc00c61bd9ce1, 0xd5eca244c7a4ff1d, 0x7762638264d279bd,
        0xc1e434bedeefd767, 0x0299351a53b8ec22, 0xb2d456e4ad251b80, 0x3e9ed1fda49cea0b,
        0x2972a92ba450bed8, 0x20216dd77be493de, 0xadffe8cf28449ec6, 0x1c4dbb1c4c27d243,
        0x15a16a8a8322d458, 0x388a128b7fd9a609, 0x2300e5d6baedf0fb, 0x2f63aa8647e15104,
        0xf1c36ce86ecec269, 0x27181125183970c9, 0xe584029370dca96d, 0x4d9bbc3e02f1cfb2,
        0xea35bc29692af6f8, 0x18e21b4beabb4137, 0x1e3b9fc625b554f4, 0x25d64362697828fd,
        0x5a3f1bb1c53a9645, 0xdb7f023869fb8d38, 0xb462065911d4e1fc, 0x49c24ae4437d8030,
        0xd793862c112b0566, 0xaadd1106730d8feb, 0xc43b6e0e97b0d568, 0xe29024c18ee6fca2,
        0x5e50c27535b88c66, 0x10383f20a4ff9a87, 0x38e8ee9d71a45af8, 0xdd5118375bf1a9b9,
        0x775005982d74d7f7, 0x86ab99b4dde6c8b0, 0xb1204f603f51c080, 0xef61ac8470250ecf,
        0x1bbcd90f132c603f, 0x0cd1dabd964db557, 0x11a3ae5beb9d1ec9, 0xf755bfeea585d11d,
        0xa3b83250268ea4d7, 0x516306f4927c93af, 0xddb4ac49c9efa1da, 0x64bb6dec369d4418,
        0xf9cc95c22b4c1fcc, 0x08d37f755f4ae9f6, 0xeec49b613478675b, 0xf143933aed25e0b0,
        0xe4c5dd8255dfc622, 0xe7ad7756f193198e, 0x92c2318b87fff9cb, 0x739c25f8fd73596d,
        0x5636cac9f16dfed0, 0xdd8f909a938e0172, 0xc6401fe115063f5b, 0x8ad97b33f1ac1455,
        0x0c49366bb25e8513, 0x0784d3d2f1698309, 0x530fb67ea1809a81, 0x410492299bb01f49,
        0x139542347424b9ac, 0x9cb0bd5ea1a1115e, 0x02e3f615c38f49a1, 0x985d4f4a9c5291ef,
        0x775b9feafdcd26e7, 0x304265a6384f0f2d, 0x593664c39773012c, 0x4f0a2e5fb028f2ce,
        0xdd611f1000c17442, 0xd8185f9adfea4fd0, 0xef87139ca9a3ab1e, 0x3ba71336c34ee133,
        0x7d3a455d56b70238, 0x660d32e130182684, 0x297a863f48cd1f43, 0x90e0a736a751ebb7,
        0x549f80ce550c4fd3, 0x0f73b2922f38bd64, 0x16bf1f73fb7a9c3f, 0x6d1f5a59005bec17,
        0x02ff876fa5ef97c4, 0xc5cb72a2a51159b0, 0x8470f39d2d5c900e, 0x25abb3f1d39fcb76,
        0x23eb8cc9b372442f, 0xd687ba55c64f6364, 0xda8d9e90fd8ff158, 0xe3cbdc7d2fe45ea7,
        0xb9a8c9b3aee52297, 0xc0d28a5c10960bd3, 0x45d7ac9b68f71a34, 0xeeb76e397069e804,
        0x3d06c8bd1514e2d9, 0x9c9c98207cb10767, 0x65700b51aedfb5ef, 0x911f451539869408,
        0x7ae6849fbc3a0ec6, 0x3bb340eba06afe7e, 0xb46e9d8b682ea65e, 0x8dcf22f9a3b34356,
        0x77bdaeda586257a7, 0xf19e400a5104d20d, 0xc368a348e46d950f, 0x9ef1cd60e679f284,
        0xe89cd854d5d01d33, 0x5cd377dc8bb882a2, 0xa7b0fb7883eee860, 0x7684403ec392950d,
        0x5fa3f06f4fed3b52, 0x8df57ac11bc04831, 0x2db01efa1e1e1897, 0x54846de4aadb9ca2,
        0xba6745385893c784, 0x541d496344d2c75b, 0xe909678474e687fe, 0xdfe89923f6c9c2ff,
        0xece5a71e0cfedc75, 0x5ff98fd5d51fe610, 0x83e8941918964615, 0x5922040b47f150c1,
        0xf97d750e3dd94521, 0x5080d4c2b86f56d7, 0xa7de115b56c78d70, 0x6a9242ac87538194,
        0xf7856ef7f9173e44, 0x2265fc92feb0dc09, 0x17dfc8e4f7ba8a57, 0x9001a64209f21db8,
        0x90004c1371b893c5, 0xb932b7cf752e5545, 0xa0b1df81b6fe59fc, 0x8ef1dd26770af2c2,
        0x0541a4f9cfbeed35, 0x9e61106178bfc530, 0xb3767e80935d8af2, 0x0098d5782065af06,
        0x31d191cd5c1466c7, 0x410fefafa319ac9d, 0xbdf8f242e316c4ab, 0x9e8cd55b57637ed0,
        0xde122bebe9a39368, 0x4d001fd58f002526, 0xca6637000eb4a9f8, 0x2f2339d624f91f78,
        0x6d1a7918c80df518, 0xdf9a4939342308e9, 0xebc2151ee6c8398c, 0x03cc2ba8a1116515,
        0xd341d037e840cf83, 0x387cb5d25af4afcc, 0xbba2515f22909e87, 0x7248fe7705f38e47,
        0x4d61e56a525d225a, 0x262e963c8da05d3d, 0x59e89b094d220ec2, 0x055d5b52b78b9c5e,
        0x82b27eb33514ef99, 0xd30094ca96b7ce7b, 0xcf5cb381cd0a1535, 0xfeed4db6919e5a7c,
        0x41703f53753be59f, 0x5eeea940fcde8b6f, 0x4cd1f1b175100206, 0x4a20358574454ec0,
        0x1478d361dbbf9fac, 0x6f02dc07d141875c, 0x296a202ed8e556a2, 0x2afd67999bf32ee5,
        0x7acfd96efa95491d, 0x6798ba0c0abb2c6d, 0x34c6f57b26c92122, 0x5736e1bad206b5de,
        0x20057d2a0056521b, 0x3dea5bd5d0578bd7, 0x16e50d897d4634ac, 0x29bff3ecb9b7a6e3,
        0x475cd3205a3bdcde, 0x18a42105c31b7e88, 0x023e7414af663068, 0x15147108121967d7,
        0xe4a3dff1d7d6fef9, 0x01a8d1a588085737, 0x11b4c74eda62beef, 0xe587cc0d69a73346,
        0x1ff7327017aa2a6e, 0x594e29c42473d06b, 0xf6f31db1899b12d5, 0xc02ac5e47312d3ca,
        0xe70201e960cb78b8, 0x6f90ff3b6a65f108, 0x42747a7245e7fa84, 0xd1f507e43ab749b2,
        0x1c86d265f15750cd, 0x3996ce73dd832c1c, 0x8e7fba02983224bd, 0xba0dec7103255dd4,
        0x9e9cbd781628fc5b, 0xdae8645996edd6a5, 0xdebe0853b1a1d378, 0xa49229d24d014343,
        0x7be5b9ffda905e1c, 0xa3c95eaec244aa30, 0x0230bca8f4df0544, 0x4135c2bebfe148c6,
        0x166fc0cc438a3c72, 0x3762b59a8ae83efa, 0xe8928a4c89114750, 0x2a440b51a4945ee5,
        0x80cefd2b7d99ff83, 0xbb9879c6e61fd62a, 0x6e7c8f1a84265034, 0x164bb2de1bbeddc8,
        0xf3c12fe54d5c653b, 0x40b9e922ed9771e2, 0x551f5b0fbe7b1840, 0x25032aa7c4cb1811,
        0xaaed34074b164346, 0x8ffd96bbf9c9c81d, 0x70fc91eb5937085c, 0x7f795e2a5f915440,
        0x4543d9df5476d3cb, 0xf172d73e004fc90d, 0xdfd1c4febcc81238, 0xbc8dfb627fe558fc
    ];
    return const[i];
}

template MDS_GL() {
    signal input in[12];
    signal output out[12];
    component reduce[12];
    for (var i = 0; i < 12; i++) {
        reduce[i] = GlReduce(74);
    }

    reduce[ 0].x <== 25*in[0] + 15*in[1] + 41*in[2] + 16*in[3] +  2*in[4] + 28*in[5] + 13*in[6] + 13*in[7] + 39*in[8] + 18*in[9] + 34*in[10] + 20*in[11];
    reduce[ 1].x <== 20*in[0] + 17*in[1] + 15*in[2] + 41*in[3] + 16*in[4] +  2*in[5] + 28*in[6] + 13*in[7] + 13*in[8] + 39*in[9] + 18*in[10] + 34*in[11];
    reduce[ 2].x <== 34*in[0] + 20*in[1] + 17*in[2] + 15*in[3] + 41*in[4] + 16*in[5] +  2*in[6] + 28*in[7] + 13*in[8] + 13*in[9] + 39*in[10] + 18*in[11];
    reduce[ 3].x <== 18*in[0] + 34*in[1] + 20*in[2] + 17*in[3] + 15*in[4] + 41*in[5] + 16*in[6] +  2*in[7] + 28*in[8] + 13*in[9] + 13*in[10] + 39*in[11];
    reduce[ 4].x <== 39*in[0] + 18*in[1] + 34*in[2] + 20*in[3] + 17*in[4] + 15*in[5] + 41*in[6] + 16*in[7] +  2*in[8] + 28*in[9] + 13*in[10] + 13*in[11];
    reduce[ 5].x <== 13*in[0] + 39*in[1] + 18*in[2] + 34*in[3] + 20*in[4] + 17*in[5] + 15*in[6] + 41*in[7] + 16*in[8] +  2*in[9] + 28*in[10] + 13*in[11];
    reduce[ 6].x <== 13*in[0] + 13*in[1] + 39*in[2] + 18*in[3] + 34*in[4] + 20*in[5] + 17*in[6] + 15*in[7] + 41*in[8] + 16*in[9] +  2*in[10] + 28*in[11];
    reduce[ 7].x <== 28*in[0] + 13*in[1] + 13*in[2] + 39*in[3] + 18*in[4] + 34*in[5] + 20*in[6] + 17*in[7] + 15*in[8] + 41*in[9] + 16*in[10] +  2*in[11];
    reduce[ 8].x <==  2*in[0] + 28*in[1] + 13*in[2] + 13*in[3] + 39*in[4] + 18*in[5] + 34*in[6] + 20*in[7] + 17*in[8] + 15*in[9] + 41*in[10] + 16*in[11];
    reduce[ 9].x <== 16*in[0] +  2*in[1] + 28*in[2] + 13*in[3] + 13*in[4] + 39*in[5] + 18*in[6] + 34*in[7] + 20*in[8] + 17*in[9] + 15*in[10] + 41*in[11];
    reduce[10].x <== 41*in[0] + 16*in[1] +  2*in[2] + 28*in[3] + 13*in[4] + 13*in[5] + 39*in[6] + 18*in[7] + 34*in[8] + 20*in[9] + 17*in[10] + 15*in[11];
    reduce[11].x <== 15*in[0] + 41*in[1] + 16*in[2] +  2*in[3] + 28*in[4] + 13*in[5] + 13*in[6] + 39*in[7] + 18*in[8] + 34*in[9] + 20*in[10] + 17*in[11];

    for (var i = 0; i < 12; i++) {
        out[i] <== reduce[i].out;
    }
}

template Poseidon_GL(nOuts) {
    signal input in[8];
    signal input capacity[4];
    signal output out[nOuts];

    signal state[31][12];
    component f1_x2[4][12];
    component f1_x4[4][12];
    component f1_x6[4][12];

    component p_x2[22];
    component p_x4[22];
    component p_x6[22];

    component f2_x2[4][12];
    component f2_x4[4][12];
    component f2_x6[4][12];

    component mds[30];

    for (var j=0; j<8; j++) {
        state[0][j] <== in[j];
    }
    for (var j=0; j<4; j++) {
        state[0][8+j] <== capacity[j];
    }

    for (var i=0; i<4; i++) {
        mds[i] = MDS_GL();
        for (var j=0; j<12; j++) {
            var c = GL_CONST(i*12+j);
            f1_x2[i][j] = GlReduce(66);
            f1_x4[i][j] = GlReduce(66);
            f1_x6[i][j] = GlReduce(66);
            f1_x2[i][j].x <== (state[i][j] + c) * (state[i][j] + c);
            f1_x4[i][j].x <== f1_x2[i][j].out * f1_x2[i][j].out;
            f1_x6[i][j].x <== f1_x2[i][j].out * f1_x4[i][j].out;
            mds[i].in[j] <== (state[i][j] + c) * f1_x6[i][j].out;
        }
        for (var j=0; j<12; j++) {
            state[i+1][j] <== mds[i].out[j];
        }
    }

    for (var i=0; i<22; i++) {
        var c = GL_CONST((4+i)*12);
        mds[4+i] = MDS_GL();
        p_x2[i] = GlReduce(66);
        p_x4[i] = GlReduce(66);
        p_x6[i] = GlReduce(66);
        p_x2[i].x <== (state[4+i][0]+c) * (state[4+i][0]+c);
        p_x4[i].x <== p_x2[i].out * p_x2[i].out;
        p_x6[i].x <== p_x2[i].out * p_x4[i].out;
        mds[4+i].in[0] <== (state[4+i][0]+c) * p_x6[i].out;
        for (var j=1; j<12; j++) {
            var c = GL_CONST((4+i)*12 +j);
            mds[4+i].in[j] <== state[4+i][j] + c;
        }

        for (var j=0; j<12; j++) {
            state[4+i+1][j] <== mds[4+i].out[j];
        }
    }

    for (var i=0; i<4; i++) {
        mds[26+i] = MDS_GL();
        for (var j=0; j<12; j++) {
            var c = GL_CONST((26+i)*12+j);
            f2_x2[i][j] = GlReduce(66);
            f2_x4[i][j] = GlReduce(66);
            f2_x6[i][j] = GlReduce(66);
            f2_x2[i][j].x <== (state[26+i][j]+c) * (state[26+i][j]+c);
            f2_x4[i][j].x <== f2_x2[i][j].out * f2_x2[i][j].out;
            f2_x6[i][j].x <== f2_x2[i][j].out * f2_x4[i][j].out;
            mds[26+i].in[j] <== (state[26+i][j]+c) * f2_x6[i][j].out;
        }
        for (var j=0; j<12; j++) {
            state[26+i+1][j] <== mds[26+i].out[j];
        }
    }

    for (var j=0; j<nOuts; j++) {
        out[j] <== state[30][j];
    }
}

template Poseidon_BN(nOuts) {
    signal input in[8];
    signal input capacity[4];
    signal output out[nOuts];

    assert(nOuts <= 12);
    component pEx = PoseidonEx(4, 4);
    pEx.initialState <== 0;
    pEx.inputs[0] <== in[0] * 2 ** 128 + in[1] * 2 ** 64 + in[2];
    pEx.inputs[1] <== in[3] * 2 ** 128 + in[4] * 2 ** 64 + in[5];
    pEx.inputs[2] <== in[6] * 2 ** 128 + in[7] * 2 ** 64 + capacity[0];
    pEx.inputs[3] <== capacity[1] * 2 ** 128 + capacity[2] * 2 ** 64 + capacity[3];

    component nBits[4];
    signal gl_hashes[12][64];
    var e2;
    for (var i = 0; i < 4; i++) {
      nBits[i] = Num2Bits(254);
      nBits[i].in <== pEx.out[i];
      for (var j = 0; j < 3; j++) {
        gl_hashes[i * 3 + j][0] <== nBits[i].out[(2 - j) * 64];
        e2 = 2;
        for (var k = 1; k < 64; k++) {
          gl_hashes[i * 3 + j][k] <== gl_hashes[i * 3 + j][k - 1] + nBits[i].out[(2 - j) * 64 + k] * e2;
          e2 = e2 + e2;
        }
      }
    }

    for (var i = 0; i < nOuts; i++) {
      out[i] <== gl_hashes[i][63];
    }
}

template HashNoPad_BN(nInputs, nOutputs) {
    signal input in[nInputs];
    signal input capacity[4];
    signal output out[nOutputs];
    assert(nOutputs <= 12);

    var nHash = (nInputs + 7) \ 8;
    component cPoseidon[nHash];
    component tmpHash[nHash][12];

    for (var i = 0; i < nHash; i++) {
        cPoseidon[i] = Poseidon_BN(12);
    }
    cPoseidon[0].capacity[0] <== capacity[0];
    cPoseidon[0].capacity[1] <== capacity[1];
    cPoseidon[0].capacity[2] <== capacity[2];
    cPoseidon[0].capacity[3] <== capacity[3];

    for (var i = 0; i < nHash; i++) {
        for (var j = 0; j < 8; j++) {
            var index = i * 8 + j;
            if (index >= nInputs) {
                if (i > 0) {
                  cPoseidon[i].in[j] <== cPoseidon[i-1].out[j];
                } else {
                  cPoseidon[i].in[j] <== 0;
                }
            } else {
                cPoseidon[i].in[j] <== in[index];
            }
        }
        if (i > 0) {
            cPoseidon[i].capacity[0] <== cPoseidon[i-1].out[8];
            cPoseidon[i].capacity[1] <== cPoseidon[i-1].out[9];
            cPoseidon[i].capacity[2] <== cPoseidon[i-1].out[10];
            cPoseidon[i].capacity[3] <== cPoseidon[i-1].out[11];
        }
    }

    component cGlReduce[nOutputs];
    for (var i = 0; i < nOutputs; i++) {
        cGlReduce[i] = GlReduce(1);
        cGlReduce[i].x <== cPoseidon[nHash - 1].out[i];
        out[i] <== cGlReduce[i].out;
    }
}

template HashNoPad_GL(nInputs, nOutputs) {
    signal input in[nInputs];
    signal input capacity[4];
    signal output out[nOutputs];
    assert(nOutputs <= 12);

    var nHash = (nInputs + 7) \ 8;
    component cPoseidon[nHash];
    component tmpHash[nHash][12];

    for (var i = 0; i < nHash; i++) {
        cPoseidon[i] = Poseidon_GL(12);
    }
    cPoseidon[0].capacity[0] <== capacity[0];
    cPoseidon[0].capacity[1] <== capacity[1];
    cPoseidon[0].capacity[2] <== capacity[2];
    cPoseidon[0].capacity[3] <== capacity[3];

    for (var i = 0; i < nHash; i++) {
        for (var j = 0; j < 8; j++) {
            var index = i * 8 + j;
            if (index >= nInputs) {
                if (i > 0) {
                  cPoseidon[i].in[j] <== cPoseidon[i-1].out[j];
                } else {
                  cPoseidon[i].in[j] <== 0;
                }
            } else {
                cPoseidon[i].in[j] <== in[index];
            }
        }
        if (i > 0) {
            cPoseidon[i].capacity[0] <== cPoseidon[i-1].out[8];
            cPoseidon[i].capacity[1] <== cPoseidon[i-1].out[9];
            cPoseidon[i].capacity[2] <== cPoseidon[i-1].out[10];
            cPoseidon[i].capacity[3] <== cPoseidon[i-1].out[11];
        }
    }

    for (var i = 0; i < nOutputs; i++) {
        out[i] <== cPoseidon[nHash - 1].out[i];
    }
}
