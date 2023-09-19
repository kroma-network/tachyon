use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::VariableBaseMSM;
use std::{mem, slice, time::Instant};
use tachyon_rs::math::elliptic_curves::bn::bn254::{
    Fr as CppFr, G1AffinePoint as CppG1AffinePoint, G1JacobianPoint as CppG1JacobianPoint,
};

#[no_mangle]
pub extern "C" fn run_msm_arkworks(
    bases: *const CppG1AffinePoint,
    scalars: *const CppFr,
    size: usize,
    duration: *mut u64,
) -> *mut CppG1JacobianPoint {
    let ret = unsafe {
        let bases: &[CppG1AffinePoint] = slice::from_raw_parts(bases, size);
        let scalars: &[CppFr] = slice::from_raw_parts(scalars, size);

        let mut bases_vec = Vec::<G1Affine>::new();
        bases_vec.reserve_exact(bases.len());
        for i in 0..size {
            bases_vec.push(G1Affine {
                x: mem::transmute(bases[i].x),
                y: mem::transmute(bases[i].y),
                infinity: bases[i].infinity,
            });
        }

        let scalars: &[Fr] = mem::transmute(scalars);
        let start = Instant::now();
        let ret = G1Projective::msm(bases_vec.as_slice(), &scalars).unwrap();
        duration.write(start.elapsed().as_micros() as u64);
        ret
    };
    Box::into_raw(Box::new(ret)) as *mut CppG1JacobianPoint
}
