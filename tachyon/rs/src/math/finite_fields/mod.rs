use crate::math::base::BigInt;
use zeroize::Zeroize;

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct PrimeField<const N: usize>(pub BigInt<N>);
