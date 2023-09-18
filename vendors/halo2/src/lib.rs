#[cxx::bridge(namespace = "tachyon::halo2")]
mod ffi {
    // Rust types and signatures exposed to C++.
    extern "Rust" {
        type CppMSM;
        type CppMSMGpu;
        type CppG1Affine;
        type CppG1Jacobian;
        type CppFq;
        type CppFr;
    }

    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("tachyon_halo2/include/msm.h");

        fn create_msm(degree: u8) -> Box<CppMSM>;
        fn destroy_msm(msm: Box<CppMSM>);
        unsafe fn msm(
            msm: *mut CppMSM,
            bases: &[CppG1Affine],
            scalars: &[CppFr],
        ) -> Box<CppG1Jacobian>;
        #[cfg(feature = "gpu")]
        fn create_msm_gpu(degree: u8, algorithm: i32) -> Box<CppMSMGpu>;
        #[cfg(feature = "gpu")]
        fn destroy_msm_gpu(msm: Box<CppMSMGpu>);
        #[cfg(feature = "gpu")]
        unsafe fn msm_gpu(
            msm: *mut CppMSMGpu,
            bases: &[CppG1Affine],
            scalars: &[CppFr],
        ) -> Box<CppG1Jacobian>;
    }
}

pub struct CppMSM;
pub struct CppMSMGpu;

#[repr(C)]
pub struct CppG1Affine {
    pub x: CppFq,
    pub y: CppFq,
}

#[repr(C)]
pub struct CppG1Jacobian {
    pub x: CppFq,
    pub y: CppFq,
    pub z: CppFq,
}

#[repr(transparent)]
pub struct CppFq(pub [u64; 4]);

#[repr(transparent)]
pub struct CppFr(pub [u64; 4]);

mod test {
    use crate::{ffi, CppFr, CppG1Affine};
    use halo2_proofs::arithmetic::best_multiexp;
    use halo2curves::{
        bn256::{Fr, G1Affine, G1},
        group::ff::Field,
    };
    use std::{mem, time::Instant};

    struct Timer {
        now: Instant,
    }

    impl Timer {
        fn new() -> Timer {
            Timer {
                now: Instant::now(),
            }
        }

        fn reset(&mut self) {
            self.now = Instant::now();
        }

        fn end(&self, message: &str) {
            println!("{}, elapsed: {:?}", message, self.now.elapsed());
        }
    }

    #[test]
    fn test_msm() {
        use rand_core::OsRng;

        let degree = 10;
        let n = 1u64 << degree;

        let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::random(OsRng)).collect();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();

        let mut timer = Timer::new();
        let expected = {
            let ret = best_multiexp(&scalars, &bases);
            timer.end("best_multiexp");
            ret
        };

        unsafe {
            timer.reset();
            let bases: Vec<CppG1Affine> = mem::transmute(bases);
            let scalars: Vec<CppFr> = mem::transmute(scalars);

            let mut msm = ffi::create_msm(degree);

            let actual = ffi::msm(&mut *msm, &bases, &scalars);
            let actual: Box<G1> = mem::transmute(actual);
            timer.end("msm");
            assert_eq!(*actual, expected);

            ffi::destroy_msm(msm);
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_msm_gpu() {
        use rand_core::OsRng;

        let degree = 10;
        let n = 1u64 << degree;

        let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::random(OsRng)).collect();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();

        let mut timer = Timer::new();
        let mut msm = ffi::create_msm_gpu(degree, 0);
        timer.end("init_msm_gpu");

        let expected = {
            timer.reset();
            let ret = best_multiexp(&scalars, &bases);
            timer.end("best_multiexp");
            ret
        };

        unsafe {
            timer.reset();
            let bases: Vec<CppG1Affine> = mem::transmute(bases);
            let scalars: Vec<CppFr> = mem::transmute(scalars);

            let actual = ffi::msm_gpu(&mut *msm, &bases, &scalars);
            let actual: Box<G1> = mem::transmute(actual);
            timer.end("msm_gpu");
            assert_eq!(*actual, expected);
        }

        ffi::destroy_msm_gpu(msm);
    }
}
