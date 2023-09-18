use bellman_ce::{
    multiexp,
    pairing::CurveAffine,
    pairing::{
        bn256::{Fr, FrRepr, G1Affine},
        ff::PrimeField,
    },
    worker::Worker,
};
use std::{mem, slice, time::Instant};

#[repr(C, align(32))]
pub struct CppG1Affine {
    pub x: CppFq,
    pub y: CppFq,
    pub infinity: bool,
}

#[repr(C)]
pub struct CppG1Jacobian {
    pub x: CppFq,
    pub y: CppFq,
    pub z: CppFq,
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct CppFq(pub [u64; 4]);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct CppFr(pub [u64; 4]);

#[no_mangle]
pub extern "C" fn run_msm_bellman(
    bases: *const CppG1Affine,
    scalars: *const CppFr,
    size: usize,
    duration: *mut u64,
) -> *mut CppG1Jacobian {
    let ret = unsafe {
        let bases: &[CppG1Affine] = slice::from_raw_parts(bases, size);
        let scalars: &[CppFr] = slice::from_raw_parts(scalars, size);

        let mut bases_vec = Vec::<G1Affine>::new();
        bases_vec.reserve_exact(bases.len());
        for i in 0..size {
            let x = mem::transmute(bases[i].x);
            let y = mem::transmute(bases[i].y);
            bases_vec.push(G1Affine::from_xy_checked(x, y).unwrap());
        }
        let pool = Worker::new();
        let scalars: &[Fr] = mem::transmute(scalars);
        let scalars_repr: Vec<FrRepr> = scalars.iter().map(|&fr| fr.into_repr()).collect();
        let start = Instant::now();
        let result = multiexp::dense_multiexp(&pool, &bases_vec, scalars_repr.as_slice());
        duration.write(start.elapsed().as_micros() as u64);
        let ret = match result {
            Ok(g1_point) => g1_point,
            Err(_) => panic!("multiexp failed"),
        };
        ret
    };
    Box::into_raw(Box::new(ret)) as *mut CppG1Jacobian
}
