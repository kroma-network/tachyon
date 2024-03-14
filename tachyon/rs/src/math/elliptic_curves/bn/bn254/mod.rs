use crate::math::{
    elliptic_curves::short_weierstrass::{AffinePoint, JacobianPoint, PointXYZZ, ProjectivePoint},
    finite_fields::{Fq2, PrimeField},
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

pub type G2AffinePoint = AffinePoint<Fq2<Fq>>;
