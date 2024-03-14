use crate::math::base::BigInt;
use zeroize::Zeroize;

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct PrimeField<const N: usize>(pub BigInt<N>);

#[repr(C, align(32))]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct Fq2<PrimeField> {
    pub c0: PrimeField,
    pub c1: PrimeField,
}

#[repr(C, align(32))]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct Fq6<Fq2> {
    pub c0: Fq2,
    pub c1: Fq2,
    pub c2: Fq2,
}

#[repr(C, align(32))]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct Fq12<Fq6> {
    pub c0: Fq6,
    pub c1: Fq6,
}
