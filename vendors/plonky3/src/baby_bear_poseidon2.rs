use std::{fmt::Debug, marker::PhantomData};

use p3_challenger::{CanObserve, CanSample};
use p3_field::Field;
use p3_symmetric::CryptographicPermutation;
use tachyon_rs::math::finite_fields::baby_bear::BabyBear as BabyBearImpl;

pub struct BabyBear(pub BabyBearImpl);

#[cxx::bridge(namespace = "tachyon::plonky3_api::baby_bear_poseidon2")]
pub mod ffi {
    extern "Rust" {
        type BabyBear;
    }

    unsafe extern "C++" {
        include!("vendors/plonky3/include/baby_bear_poseidon2_duplex_challenger.h");

        type DuplexChallenger;

        fn new_duplex_challenger() -> UniquePtr<DuplexChallenger>;
        fn observe(self: Pin<&mut DuplexChallenger>, value: &BabyBear);
        fn sample(self: Pin<&mut DuplexChallenger>) -> Box<BabyBear>;
        fn clone(&self) -> UniquePtr<DuplexChallenger>;
    }
}

pub struct DuplexChallenger<F, P, const WIDTH: usize, const RATE: usize> {
    inner: cxx::UniquePtr<ffi::DuplexChallenger>,
    _marker: PhantomData<(F, P)>,
}

impl<F, P, const WIDTH: usize, const RATE: usize> Clone for DuplexChallenger<F, P, WIDTH, RATE> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _marker: PhantomData,
        }
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> Debug for DuplexChallenger<F, P, WIDTH, RATE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuplexChallenger").finish()
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> DuplexChallenger<F, P, WIDTH, RATE> {
    pub fn new() -> DuplexChallenger<F, P, WIDTH, RATE> {
        DuplexChallenger {
            inner: ffi::new_duplex_challenger(),
            _marker: PhantomData,
        }
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> CanObserve<F>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy + Debug,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, value: F) {
        self.inner
            .pin_mut()
            .observe(unsafe { std::mem::transmute::<_, &BabyBear>(&value) });
    }
}

impl<F, P, const N: usize, const WIDTH: usize, const RATE: usize> CanObserve<[F; N]>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy + Debug,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, values: [F; N]) {
        for value in values {
            self.observe(value);
        }
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> CanObserve<Vec<Vec<F>>>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy + Debug,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, valuess: Vec<Vec<F>>) {
        for values in valuess {
            for value in values {
                self.observe(value);
            }
        }
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> CanSample<F>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Field,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn sample(&mut self) -> F {
        *unsafe { std::mem::transmute::<_, Box<F>>(self.inner.pin_mut().sample()) }
    }
}
