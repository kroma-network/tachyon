use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::traits::find_poseidon_ark_and_mds;
use ark_crypto_primitives::sponge::poseidon::{PoseidonConfig, PoseidonSponge};
use ark_crypto_primitives::sponge::{CryptographicSponge, FieldBasedCryptographicSponge};
use std::{mem, slice, time::Instant};
use tachyon_rs::math::elliptic_curves::bn::bn254::Fr as CppFr;

#[no_mangle]
pub extern "C" fn run_poseidon_arkworks(
    pre_images: *const CppFr,
    absorbing_num: usize,
    squeezing_num: usize,
    duration: *mut u64,
) -> *mut CppFr {
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

    let pre_images = unsafe {
        let pre_images: &[CppFr] = slice::from_raw_parts(pre_images, absorbing_num);
        let pre_images: &[Fr] = mem::transmute(pre_images);

        pre_images
    };
    let mut sponge = PoseidonSponge::new(&poseidon_config);

    let start = Instant::now();
    for pre_image in pre_images {
        sponge.absorb(pre_image);
    }
    let squeezed_elements = sponge.squeeze_native_field_elements(squeezing_num);
    unsafe {
        duration.write(start.elapsed().as_micros() as u64);
    }

    Box::into_raw(Box::new(squeezed_elements[0])) as *mut CppFr
}
