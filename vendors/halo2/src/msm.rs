#[cfg(test)]
mod test {
    use crate::bn254::{ffi, Fr as CppFr, G1Point2 as CppG1Point2};
    use halo2_proofs::arithmetic::best_multiexp;
    use halo2curves::{
        bn256::{Fr, G1Affine, G1},
        group::{ff::Field, Curve, Group},
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

    struct TestSet {
        bases: Vec<G1Affine>,
        scalars: Vec<Fr>,
    }

    impl TestSet {
        fn create(n: usize) -> TestSet {
            use rand_core::OsRng;

            let mut base = G1::random(OsRng);
            TestSet {
                bases: (0..n)
                    .map(|_| {
                        let ret = base.to_affine();
                        base = base.double();
                        ret
                    })
                    .collect(),
                scalars: (0..n).map(|_| Fr::random(OsRng)).collect(),
            }
        }
    }

    #[test]
    fn test_msm() {
        let degree = 10;
        let n = 1usize << degree;

        let test_set = TestSet::create(n);

        let mut timer = Timer::new();
        let expected = {
            let ret = best_multiexp(&test_set.scalars, &test_set.bases);
            timer.end("best_multiexp");
            ret
        };

        unsafe {
            timer.reset();
            let bases: Vec<CppG1Point2> = mem::transmute(test_set.bases);
            let scalars: Vec<CppFr> = mem::transmute(test_set.scalars);

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
        let degree = 10;
        let n = 1usize << degree;

        let test_set = TestSet::create(n);

        let mut timer = Timer::new();
        let mut msm = ffi::create_g1_msm_gpu(degree, 0);
        timer.end("init_msm_gpu");

        let expected = {
            timer.reset();
            let ret = best_multiexp(&test_set.scalars, &test_set.bases);
            timer.end("best_multiexp");
            ret
        };

        unsafe {
            timer.reset();
            let bases: Vec<CppG1Point2> = mem::transmute(test_set.bases);
            let scalars: Vec<CppFr> = mem::transmute(test_set.scalars);

            let actual = ffi::g1_point2_msm_gpu(&mut *msm, &bases, &scalars);
            let actual: Box<G1> = mem::transmute(actual);
            timer.end("msm_gpu");
            assert_eq!(*actual, expected);
        }

        ffi::destroy_g1_msm_gpu(msm);
    }
}
