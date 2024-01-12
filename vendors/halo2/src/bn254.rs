use tachyon_rs::math::elliptic_curves::bn::bn254::{
    Fr as FrImpl, G1JacobianPoint as G1JacobianPointImpl, G1Point2 as G1Point2Impl,
};

pub struct G1MSM;
pub struct G1MSMGpu;
pub struct G1JacobianPoint(pub G1JacobianPointImpl);
pub struct G1Point2(pub G1Point2Impl);
pub struct Fr(pub FrImpl);

#[cxx::bridge(namespace = "tachyon::halo2_api::bn254")]
pub mod ffi {
    extern "Rust" {
        type G1MSM;
        type G1MSMGpu;
        type G1JacobianPoint;
        type G1Point2;
        type Fr;
    }

    unsafe extern "C++" {
        include!("vendors/halo2/include/bn254_msm.h");
        #[cfg(feature = "gpu")]
        include!("vendors/halo2/include/bn254_msm_gpu.h");

        fn create_g1_msm(degree: u8) -> Box<G1MSM>;
        fn destroy_g1_msm(msm: Box<G1MSM>);
        unsafe fn g1_point2_msm(
            msm: *mut G1MSM,
            bases: &[G1Point2],
            scalars: &[Fr],
        ) -> Box<G1JacobianPoint>;
        #[cfg(feature = "gpu")]
        fn create_g1_msm_gpu(degree: u8, algorithm: i32) -> Box<G1MSMGpu>;
        #[cfg(feature = "gpu")]
        fn destroy_g1_msm_gpu(msm: Box<G1MSMGpu>);
        #[cfg(feature = "gpu")]
        unsafe fn g1_point2_msm_gpu(
            msm: *mut G1MSMGpu,
            bases: &[G1Point2],
            scalars: &[Fr],
        ) -> Box<G1JacobianPoint>;
    }

    unsafe extern "C++" {
        include!("vendors/halo2/include/bn254_prover.h");

        fn create_proof(degree: u8);
    }
}
