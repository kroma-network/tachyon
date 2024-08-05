use crate::consts::RNGType;

pub trait SerializableRng {
    fn state(&self) -> Vec<u8>;
    fn rng_type() -> RNGType;
}
