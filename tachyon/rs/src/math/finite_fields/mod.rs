pub mod baby_bear;

use crate::math::base::BigInt;
use zeroize::Zeroize;

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct PrimeField<const N: usize>(pub BigInt<N>);

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct Fp2<PrimeField> {
    pub c0: PrimeField,
    pub c1: PrimeField,
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct Fp4<PrimeField> {
    pub c0: PrimeField,
    pub c1: PrimeField,
    pub c2: PrimeField,
    pub c3: PrimeField,
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct Fp6<Fp2> {
    pub c0: Fp2,
    pub c1: Fp2,
    pub c2: Fp2,
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct Fp12<Fp6> {
    pub c0: Fp6,
    pub c1: Fp6,
}
