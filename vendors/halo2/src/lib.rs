#[cfg(test)]
mod test {
    use halo2_proofs::arithmetic::best_multiexp;
    use halo2curves::{
        bn256::{Fr, G1Affine, G1},
        group::ff::Field,
    };
    use std::{mem, time::Instant};
    use tachyon_rs::math::elliptic_curves::bn::bn254::{ffi, Fr as CppFr, G1Point2 as CppG1Point2};

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
            let bases: Vec<CppG1Point2> = mem::transmute(bases);
            let scalars: Vec<CppFr> = mem::transmute(scalars);

            let mut msm = ffi::create_g1_msm(degree);

            let actual = ffi::g1_point2_msm(&mut *msm, &bases, &scalars);
            let actual: Box<G1> = mem::transmute(actual);
            timer.end("msm");
            assert_eq!(*actual, expected);

            ffi::destroy_g1_msm(msm);
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
        let mut msm = ffi::create_g1_msm_gpu(degree, 0);
        timer.end("init_msm_gpu");

        let expected = {
            timer.reset();
            let ret = best_multiexp(&scalars, &bases);
            timer.end("best_multiexp");
            ret
        };

        unsafe {
            timer.reset();
            let bases: Vec<CppG1Point2> = mem::transmute(bases);
            let scalars: Vec<CppFr> = mem::transmute(scalars);

            let actual = ffi::g1_point2_msm_gpu(&mut *msm, &bases, &scalars);
            let actual: Box<G1> = mem::transmute(actual);
            timer.end("msm_gpu");
            assert_eq!(*actual, expected);
        }

        ffi::destroy_g1_msm_gpu(msm);
    }
}
