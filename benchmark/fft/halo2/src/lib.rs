use halo2_proofs::arithmetic::{best_fft, parallelize, Field};
use halo2_proofs::halo2curves::{bn256::Fr, Group};
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
        let mut coeffs = coeffs.to_vec();
        let omega: Fr = mem::transmute(*omega);

        let start = Instant::now();
        best_fft(&mut coeffs, omega, k);
        duration.write(start.elapsed().as_micros() as u64);
        coeffs
    };
    Box::into_raw(ret.into_boxed_slice()) as *mut CppFr
}

#[no_mangle]
pub extern "C" fn run_ifft_halo2(
    coeffs: *const CppFr,
    n: usize,
    omega_inv: *const CppFr,
    k: u32,
    duration: *mut u64,
) -> *mut CppFr {
    let ret = unsafe {
        let coeffs: &[CppFr] = slice::from_raw_parts(coeffs, n);

        let coeffs: &[Fr] = mem::transmute(coeffs);
        let mut coeffs = coeffs.to_vec();
        let omega_inv: Fr = mem::transmute(*omega_inv);
        let divisor = Fr::from(1 << k).invert().unwrap();

        let start = Instant::now();
        best_fft(&mut coeffs, omega_inv, k);
        parallelize(&mut coeffs, |coeffs, _| {
            for coeff in coeffs {
                coeff.group_scale(&divisor);
            }
        });
        duration.write(start.elapsed().as_micros() as u64);
        coeffs
    };
    Box::into_raw(ret.into_boxed_slice()) as *mut CppFr
}
