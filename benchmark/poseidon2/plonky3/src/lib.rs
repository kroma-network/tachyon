use ff::PrimeField;
use p3_bn254_fr::{Bn254Fr, DiffusionMatrixBN254, FFBn254Fr};
use p3_field::AbstractField;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixHL};
use p3_symmetric::Permutation;
use std::time::Instant;
use tachyon_rs::math::elliptic_curves::bn::bn254::Fr as CppFr;
use zkhash::ark_ff::{BigInteger, PrimeField as ark_PrimeField};
use zkhash::fields::bn256::FpBN256 as ark_FpBN256;
use zkhash::poseidon2::poseidon2_instance_bn256::RC3;

fn bn254_from_ark_ff(input: ark_FpBN256) -> Bn254Fr {
    let bytes = input.into_bigint().to_bytes_le();

    let mut res = <FFBn254Fr as PrimeField>::Repr::default();

    for (i, digit) in res.0.as_mut().iter_mut().enumerate() {
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

#[no_mangle]
pub extern "C" fn run_poseidon_plonky3_bn254_fr(duration: *mut u64) -> *mut CppFr {
    const WIDTH: usize = 3;
    const D: u64 = 5;
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 56;

    // Copy over round constants from zkhash.
    let mut round_constants: Vec<[Bn254Fr; WIDTH]> = RC3
        .iter()
        .map(|vec| {
            vec.iter()
                .cloned()
                .map(bn254_from_ark_ff)
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

    let poseidon =
        Poseidon2::<Bn254Fr, Poseidon2ExternalMatrixHL, DiffusionMatrixBN254, WIDTH, D>::new(
            ROUNDS_F,
            external_round_constants,
            Poseidon2ExternalMatrixHL,
            ROUNDS_P,
            internal_round_constants,
            DiffusionMatrixBN254,
        );

    let mut input = (0..3)
        .map(|_i| Bn254Fr::zero())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let start = Instant::now();
    poseidon.permute_mut(&mut input);
    unsafe {
        duration.write(start.elapsed().as_micros() as u64);
    }

    Box::into_raw(Box::new(input[1])) as *mut CppFr
}
