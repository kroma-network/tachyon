pub use crate::math::finite_fields::PrimeField;
use zeroize::Zeroize;

#[repr(C, align(32))]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct AffinePoint<PrimeField> {
    pub x: PrimeField,
    pub y: PrimeField,
    pub infinity: bool,
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct JacobianPoint<PrimeField> {
    pub x: PrimeField,
    pub y: PrimeField,
    pub z: PrimeField,
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct ProjectivePoint<PrimeField> {
    pub x: PrimeField,
    pub y: PrimeField,
    pub z: PrimeField,
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct PointXYZZ<PrimeField> {
    pub x: PrimeField,
    pub y: PrimeField,
    pub zz: PrimeField,
    pub zzz: PrimeField,
}

pub mod bn;
