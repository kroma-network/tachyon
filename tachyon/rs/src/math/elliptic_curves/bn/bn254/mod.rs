use crate::math::{
    elliptic_curves::{AffinePoint, JacobianPoint, PointXYZZ, ProjectivePoint},
    finite_fields::PrimeField,
    geometry::{Point2, Point3, Point4},
};

pub type Fq = PrimeField<4>;
pub type Fr = PrimeField<4>;

pub type G1AffinePoint = AffinePoint<Fq>;
pub type G1JacobianPoint = JacobianPoint<Fq>;
pub type G1ProjectivePoint = ProjectivePoint<Fq>;
pub type G1PointXYZZ = PointXYZZ<Fq>;

pub type G1Point2 = Point2<Fq>;
pub type G1Point3 = Point3<Fq>;
pub type G1Point4 = Point4<Fq>;

pub struct G1MSM;

pub struct G1MSMGpu;

#[cxx::bridge(namespace = "tachyon::rs::math::bn254")]
pub mod ffi {
    extern "Rust" {
        type G1MSM;
        type G1MSMGpu;
        type G1AffinePoint;
        type G1JacobianPoint;
        type G1Point2;
        type Fr;
    }

    unsafe extern "C++" {
        include!("tachyon_rs/math/elliptic_curves/bn/bn254/msm.h");
        #[cfg(feature = "gpu")]
        include!("tachyon_rs/math/elliptic_curves/bn/bn254/msm_gpu.h");

        fn create_g1_msm(degree: u8) -> Box<G1MSM>;
        fn destroy_g1_msm(msm: Box<G1MSM>);
        unsafe fn g1_affine_msm(
            msm: *mut G1MSM,
            bases: &[G1AffinePoint],
            scalars: &[Fr],
        ) -> Box<G1JacobianPoint>;
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
        unsafe fn g1_affine_msm_gpu(
            msm: *mut G1MSMGpu,
            bases: &[G1AffinePoint],
            scalars: &[Fr],
        ) -> Box<G1JacobianPoint>;
        #[cfg(feature = "gpu")]
        unsafe fn g1_point2_msm_gpu(
            msm: *mut G1MSMGpu,
            bases: &[G1Point2],
            scalars: &[Fr],
        ) -> Box<G1JacobianPoint>;
    }
}
