use crate::{consts::RNGType, rng::SerializableRng};

#[cxx::bridge(namespace = "tachyon::halo2_api")]
pub mod ffi {
    unsafe extern "C++" {
        include!("vendors/scroll_halo2/include/xor_shift_rng.h");

        type XORShiftRng;

        fn new_xor_shift_rng(seed: [u8; 16]) -> UniquePtr<XORShiftRng>;
        fn next_u32(self: Pin<&mut XORShiftRng>) -> u32;
        fn clone(&self) -> UniquePtr<XORShiftRng>;
        fn state(&self) -> Vec<u8>;
    }
}

pub struct XORShiftRng {
    inner: cxx::UniquePtr<ffi::XORShiftRng>,
}

impl SerializableRng for XORShiftRng {
    fn state(&self) -> Vec<u8> {
        self.inner.state()
    }

    fn rng_type() -> RNGType {
        RNGType::XORShift
    }
}

impl Clone for XORShiftRng {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl rand_core::SeedableRng for XORShiftRng {
    type Seed = [u8; 16];

    fn from_seed(seed: Self::Seed) -> Self {
        Self {
            inner: ffi::new_xor_shift_rng(seed),
        }
    }
}

impl rand_core::RngCore for XORShiftRng {
    fn next_u32(&mut self) -> u32 {
        self.inner.pin_mut().next_u32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        rand_core::impls::next_u64_via_u32(self)
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        rand_core::impls::fill_bytes_via_next(self, dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use rand_core::{RngCore, SeedableRng};

    use crate::{consts::XOR_SHIFT_SEED, rng::SerializableRng};

    #[test]
    fn test_rng() {
        let mut rng = rand_xorshift::XorShiftRng::from_seed(XOR_SHIFT_SEED);
        let mut rng_tachyon = crate::xor_shift_rng::XORShiftRng::from_seed(XOR_SHIFT_SEED);

        const LEN: i32 = 100;
        let random_u64s = (0..LEN).map(|_| rng.next_u64()).collect::<Vec<_>>();
        let random_u64s_tachyon = (0..LEN).map(|_| rng_tachyon.next_u64()).collect::<Vec<_>>();
        assert_eq!(random_u64s, random_u64s_tachyon);
    }

    #[test]
    fn test_clone() {
        let mut rng = crate::xor_shift_rng::XORShiftRng::from_seed(XOR_SHIFT_SEED);
        let mut rng_clone = rng.clone();

        const LEN: i32 = 100;
        let random_u64s = (0..LEN).map(|_| rng.next_u64()).collect::<Vec<_>>();
        let random_u64s_clone = (0..LEN).map(|_| rng_clone.next_u64()).collect::<Vec<_>>();
        assert_eq!(random_u64s, random_u64s_clone);
    }

    #[test]
    fn test_state() {
        let rng = crate::xor_shift_rng::XORShiftRng::from_seed(XOR_SHIFT_SEED);
        assert_eq!(
            rng.state(),
            vec![89, 98, 190, 93, 118, 61, 49, 141, 23, 219, 55, 50, 84, 6, 188, 229]
        );
    }
}
