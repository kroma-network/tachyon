use core::fmt;
use ff::PrimeField;
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_bn254_fr::{Bn254Fr, DiffusionMatrixBN254, FFBn254Fr};
use p3_field::AbstractField;
use p3_poseidon2::{DiffusionPermutation, Poseidon2, Poseidon2ExternalMatrixHL};
use p3_symmetric::Permutation;
use std::time::Instant;
use tachyon_rs::math::{
    elliptic_curves::bn::bn254::Fr as CppBn254Fr, finite_fields::baby_bear::BabyBear as CppBabyBear,
};
use zkhash::ark_ff::{BigInteger, Field, PrimeField as ark_PrimeField};
use zkhash::fields::{babybear::FpBabyBear as ark_FpBabyBear, bn256::FpBN256 as ark_FpBN256};
use zkhash::poseidon2::{
    poseidon2_instance_babybear::RC16 as BabyBearRC16, poseidon2_instance_bn256::RC3 as BN256RC3,
};

fn bn254_from_ark_ff(input: ark_FpBN256) -> Bn254Fr {
    let bytes = input.into_bigint().to_bytes_le();

    let mut res = <FFBn254Fr as PrimeField>::Repr::default();

    for (i, digit) in res.as_mut().iter_mut().enumerate() {
        *digit = bytes[i];
    }

    let value = FFBn254Fr::from_repr(res);

    if value.is_some().into() {
        Bn254Fr {
            value: value.unwrap(),
        }
    } else {
        panic!("Invalid field element")
    }
}

fn baby_bear_from_ark_ff(input: ark_FpBabyBear) -> BabyBear {
    BabyBear::from_canonical_u32(input.into_bigint().0[0] as u32)
}

fn run_poseidon2<
    const WIDTH: usize,
    const D: u64,
    const ROUNDS_F: usize,
    const ROUNDS_P: usize,
    NativeF: p3_field::PrimeField + fmt::Debug,
    F: Field,
    R,
    DiffusionMatrix: DiffusionPermutation<NativeF, WIDTH>,
>(
    duration: *mut u64,
    rc: &Vec<Vec<F>>,
    from_ark_ff: &dyn Fn(F) -> NativeF,
    diffusion_matrix: DiffusionMatrix,
) -> *mut R {
    // Copy over round constants from zkhash.
    let mut round_constants: Vec<[NativeF; WIDTH]> = rc
        .iter()
        .map(|vec| {
            vec.iter()
                .cloned()
                .map(from_ark_ff)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect();
    let internal_start = ROUNDS_F / 2;
    let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
    let internal_round_constants = round_constants
        .drain(internal_start..internal_end)
        .map(|vec| vec[0])
        .collect::<Vec<_>>();
    let external_round_constants = round_constants;

    let poseidon2 = Poseidon2::<NativeF, Poseidon2ExternalMatrixHL, DiffusionMatrix, WIDTH, D>::new(
        ROUNDS_F,
        external_round_constants,
        Poseidon2ExternalMatrixHL,
        ROUNDS_P,
        internal_round_constants,
        diffusion_matrix,
    );

    let mut input = (0..WIDTH)
        .map(|_i| NativeF::zero())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let start = Instant::now();
    for _ in 0..10000 {
        poseidon2.permute_mut(&mut input);
    }
    unsafe {
        duration.write(start.elapsed().as_micros() as u64);
    }

    Box::into_raw(Box::new(input[1])) as *mut R
}

#[no_mangle]
pub extern "C" fn run_poseidon2_plonky3_baby_bear(duration: *mut u64) -> *mut CppBabyBear {
    run_poseidon2::<16, 7, 8, 13, BabyBear, ark_FpBabyBear, CppBabyBear, DiffusionMatrixBabyBear>(
        duration,
        &BabyBearRC16,
        &baby_bear_from_ark_ff,
        DiffusionMatrixBabyBear::default(),
    )
}

#[no_mangle]
pub extern "C" fn run_poseidon2_plonky3_bn254_fr(duration: *mut u64) -> *mut CppBn254Fr {
    run_poseidon2::<3, 5, 8, 56, Bn254Fr, ark_FpBN256, CppBn254Fr, DiffusionMatrixBN254>(
        duration,
        &BN256RC3,
        &bn254_from_ark_ff,
        DiffusionMatrixBN254,
    )
}
