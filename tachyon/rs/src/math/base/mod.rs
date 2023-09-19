use zeroize::Zeroize;

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Zeroize)]
pub struct BigInt<const N: usize>(pub [u64; N]);
