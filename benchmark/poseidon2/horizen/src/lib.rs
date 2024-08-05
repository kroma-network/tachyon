use std::{sync::Arc, time::Instant};
use tachyon_rs::math::{
    elliptic_curves::bn::bn254::Fr as CppBn254Fr, finite_fields::baby_bear::BabyBear as CppBabyBear,
};
use zkhash::{
    ark_ff::PrimeField,
    poseidon2::{
        poseidon2::Poseidon2, poseidon2_instance_babybear::POSEIDON2_BABYBEAR_16_PARAMS,
        poseidon2_instance_bn256::POSEIDON2_BN256_PARAMS, poseidon2_params::Poseidon2Params,
    },
};

fn run_poseidon<F: PrimeField + std::convert::From<i32>, R>(
    duration: *mut u64,
    params: &Arc<Poseidon2Params<F>>,
) -> *mut R {
    let poseidon = Poseidon2::new(params);

    let t = poseidon.get_t();
    let input: Vec<F> = (0..t).map(|_| F::from(0)).collect();

    let start = Instant::now();
    let state = poseidon.permutation(&input);
    unsafe {
        duration.write(start.elapsed().as_micros() as u64);
    }

    Box::into_raw(Box::new(state[1])) as *mut R
}

#[no_mangle]
pub extern "C" fn run_poseidon_horizen_baby_bear(duration: *mut u64) -> *mut CppBabyBear {
    run_poseidon::<_, CppBabyBear>(duration, &POSEIDON2_BABYBEAR_16_PARAMS)
}

#[no_mangle]
pub extern "C" fn run_poseidon_horizen_bn254_fr(duration: *mut u64) -> *mut CppBn254Fr {
    run_poseidon::<_, CppBn254Fr>(duration, &POSEIDON2_BN256_PARAMS)
}
