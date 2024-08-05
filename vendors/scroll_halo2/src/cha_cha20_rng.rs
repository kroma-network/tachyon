use crate::{consts::RNGType, rng::SerializableRng};

#[cxx::bridge(namespace = "tachyon::halo2_api")]
pub mod ffi {
    unsafe extern "C++" {
        include!("vendors/scroll_halo2/include/cha_cha20_rng.h");

        type ChaCha20Rng;

        fn new_cha_cha20_rng(seed: [u8; 32]) -> UniquePtr<ChaCha20Rng>;
        fn next_u32(self: Pin<&mut ChaCha20Rng>) -> u32;
        fn clone(&self) -> UniquePtr<ChaCha20Rng>;
        fn state(&self) -> Vec<u8>;
    }
}

pub struct ChaCha20Rng {
    inner: cxx::UniquePtr<ffi::ChaCha20Rng>,
}

impl SerializableRng for ChaCha20Rng {
    fn state(&self) -> Vec<u8> {
        self.inner.state()
    }

    fn rng_type() -> RNGType {
        RNGType::ChaCha20
    }
}

impl Clone for ChaCha20Rng {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl rand_core::SeedableRng for ChaCha20Rng {
    type Seed = [u8; 32];

    fn from_seed(seed: Self::Seed) -> Self {
        Self {
            inner: ffi::new_cha_cha20_rng(seed),
        }
    }
}

impl rand_core::RngCore for ChaCha20Rng {
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

    use crate::{consts::CHA_CHA20_SEED, rng::SerializableRng};

    #[test]
    fn test_rng() {
        let mut rng = rand_chacha::ChaCha20Rng::from_seed(CHA_CHA20_SEED);
        let mut rng_tachyon = crate::cha_cha20_rng::ChaCha20Rng::from_seed(CHA_CHA20_SEED);

        const LEN: i32 = 100;
        let random_u64s = (0..LEN).map(|_| rng.next_u64()).collect::<Vec<_>>();
        let random_u64s_tachyon = (0..LEN).map(|_| rng_tachyon.next_u64()).collect::<Vec<_>>();
        assert_eq!(random_u64s, random_u64s_tachyon);
    }

    #[test]
    fn test_clone() {
        let mut rng = crate::cha_cha20_rng::ChaCha20Rng::from_seed(CHA_CHA20_SEED);
        let mut rng_clone = rng.clone();

        const LEN: i32 = 100;
        let random_u64s = (0..LEN).map(|_| rng.next_u64()).collect::<Vec<_>>();
        let random_u64s_clone = (0..LEN).map(|_| rng_clone.next_u64()).collect::<Vec<_>>();
        assert_eq!(random_u64s, random_u64s_clone);
    }

    #[test]
    fn test_state() {
        let rng = crate::cha_cha20_rng::ChaCha20Rng::from_seed(CHA_CHA20_SEED);
        assert_eq!(
            rng.state(),
            vec![
                16, 0, 0, 0, 0, 0, 0, 0, 101, 120, 112, 97, 110, 100, 32, 51, 50, 45, 98, 121, 116,
                101, 32, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0
            ]
        );
    }
}
