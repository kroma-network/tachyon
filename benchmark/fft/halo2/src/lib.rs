use halo2_proofs::arithmetic::best_fft;
use halo2_proofs::halo2curves::bn256::Fr;
use std::{mem, slice, time::Instant};
use tachyon_rs::math::elliptic_curves::bn::bn254::Fr as CppFr;

#[no_mangle]
pub extern "C" fn run_fft_halo2(
    coeffs: *const CppFr,
    n: usize,
    omega: *const CppFr,
    k: u32,
    duration: *mut u64,
) -> *mut CppFr {
    let ret = unsafe {
        let coeffs: &[CppFr] = slice::from_raw_parts(coeffs, n);

        let coeffs: &[Fr] = mem::transmute(coeffs);
        let omega: Fr = mem::transmute(*omega);

        let start = Instant::now();
        let ret = best_fft(&mut coeffs.to_vec(), omega, k);
        duration.write(start.elapsed().as_micros() as u64);
        ret
    };
    Box::into_raw(Box::new(ret)) as *mut CppFr
}
