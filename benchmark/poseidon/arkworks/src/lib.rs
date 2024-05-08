use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::traits::find_poseidon_ark_and_mds;
use ark_crypto_primitives::sponge::poseidon::{PoseidonConfig, PoseidonSponge};
use ark_crypto_primitives::sponge::CryptographicSponge;
use std::time::Instant;
use tachyon_rs::math::elliptic_curves::bn::bn254::Fr as CppFr;

#[no_mangle]
pub extern "C" fn run_poseidon_arkworks(duration: *mut u64) -> *mut CppFr {
    let (ark, mds) = find_poseidon_ark_and_mds::<Fr>(254, 8, 8, 63, 0);
    let poseidon_config = PoseidonConfig {
        full_rounds: 8,
        partial_rounds: 63,
        alpha: 5,
        ark,
        mds,
        rate: 8,
        capacity: 1,
    };

    let mut sponge = PoseidonSponge::new(&poseidon_config);

    let start = Instant::now();
    sponge.permute();
    unsafe {
        duration.write(start.elapsed().as_micros() as u64);
    }

    Box::into_raw(Box::new(sponge.state[1])) as *mut CppFr
}
