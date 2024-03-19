use ark_poly::domain::{EvaluationDomain, Radix2EvaluationDomain};
use ark_test_curves::bn254::Fr;
use std::{mem, slice, time::Instant};
use tachyon_rs::math::elliptic_curves::bn::bn254::Fr as CppFr;

#[no_mangle]
pub extern "C" fn run_fft_arkworks(
    coeffs: *const CppFr,
    n: usize,
    _omega: *const CppFr,
    _k: u32,
    duration: *mut u64,
) -> *mut CppFr {
    let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
    let ret = unsafe {
        let coeffs: &[CppFr] = slice::from_raw_parts(coeffs, n);

        let coeffs: &[Fr] = mem::transmute(coeffs);
        let mut coeffs = coeffs.to_vec();

        let start = Instant::now();
        domain.fft_in_place(&mut coeffs);
        duration.write(start.elapsed().as_micros() as u64);
        coeffs
    };
    Box::into_raw(ret.into_boxed_slice()) as *mut CppFr
}

#[no_mangle]
pub extern "C" fn run_ifft_arkworks(
    coeffs: *const CppFr,
    n: usize,
    _omega_inv: *const CppFr,
    _k: u32,
    duration: *mut u64,
) -> *mut CppFr {
    let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
    let ret = unsafe {
        let coeffs: &[CppFr] = slice::from_raw_parts(coeffs, n);

        let coeffs: &[Fr] = mem::transmute(coeffs);
        let mut coeffs = coeffs.to_vec();

        let start = Instant::now();
        domain.ifft_in_place(&mut coeffs);
        duration.write(start.elapsed().as_micros() as u64);
        coeffs
    };
    Box::into_raw(ret.into_boxed_slice()) as *mut CppFr
}
