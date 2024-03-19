use bellman_ce::{domain::EvaluationDomain, domain::Scalar, pairing::bn256::Bn256, worker::Worker};
use std::{mem, slice, time::Instant};
use tachyon_rs::math::elliptic_curves::bn::bn254::Fr as CppFr;

#[no_mangle]
pub extern "C" fn run_fft_bellman(
    coeffs: *const CppFr,
    n: usize,
    _omega: *const CppFr,
    _k: u32,
    duration: *mut u64,
) -> *mut CppFr {
    let ret = unsafe {
        let coeffs: &[CppFr] = slice::from_raw_parts(coeffs, n);

        let coeffs: &[Scalar<Bn256>] = mem::transmute(coeffs);
        let coeffs = coeffs.to_vec();

        let worker = Worker::new();
        let mut domain = EvaluationDomain::from_coeffs(coeffs).unwrap();

        let start = Instant::now();
        domain.fft(&worker);
        duration.write(start.elapsed().as_micros() as u64);
        domain.into_coeffs()
    };
    Box::into_raw(ret.into_boxed_slice()) as *mut CppFr
}

#[no_mangle]
pub extern "C" fn run_ifft_bellman(
    coeffs: *const CppFr,
    n: usize,
    _omega_inv: *const CppFr,
    _k: u32,
    duration: *mut u64,
) -> *mut CppFr {
    let ret = unsafe {
        let coeffs: &[CppFr] = slice::from_raw_parts(coeffs, n);

        let coeffs: &[Scalar<Bn256>] = mem::transmute(coeffs);
        let mut coeffs = coeffs.to_vec();

        let worker = Worker::new();
        let mut domain = EvaluationDomain::from_coeffs(coeffs).unwrap();

        let start = Instant::now();
        domain.ifft(&worker);
        duration.write(start.elapsed().as_micros() as u64);
        domain.into_coeffs()
    };
    Box::into_raw(ret.into_boxed_slice()) as *mut CppFr
}
