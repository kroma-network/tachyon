use std::time::Instant;
use tachyon_rs::math::elliptic_curves::bn::bn254::Fr as CppFr;
use zkhash::{
    fields::bn256::FpBN256,
    poseidon2::{poseidon2::Poseidon2, poseidon2_instance_bn256::POSEIDON2_BN256_PARAMS},
};

#[no_mangle]
pub extern "C" fn run_poseidon_horizen(duration: *mut u64) -> *mut CppFr {
    let poseidon = Poseidon2::new(&POSEIDON2_BN256_PARAMS);

    let t = poseidon.get_t();
    let input: Vec<FpBN256> = (0..t).map(|_i| FpBN256::from(0)).collect();

    let start = Instant::now();
    let state = poseidon.permutation(&input);
    unsafe {
        duration.write(start.elapsed().as_micros() as u64);
    }

    Box::into_raw(Box::new(state[1])) as *mut CppFr
}
