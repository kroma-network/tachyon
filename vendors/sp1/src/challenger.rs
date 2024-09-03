#[cfg(test)]
mod test {
    use crate::baby_bear_poseidon2::DuplexChallenger as TachyonDuplexChallenger;
    use p3_baby_bear::BabyBear;
    use p3_challenger::{CanObserve, CanSample, DuplexChallenger};
    use p3_field::{extension::BinomialExtensionField, AbstractField};
    use sp1_core::utils::baby_bear_poseidon2::{my_perm, Perm};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_duplex_challenger() {
        const WIDTH: usize = 16;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 13;

        const RATE: usize = 8;

        let perm = my_perm();

        let mut tachyon_duplex_challenger: TachyonDuplexChallenger<F, Perm, WIDTH, RATE> =
            TachyonDuplexChallenger::new();

        let mut duplex_challenger: DuplexChallenger<F, Perm, WIDTH, RATE> =
            DuplexChallenger::new(perm);

        (0..20).for_each(|element| {
            tachyon_duplex_challenger.observe(F::from_canonical_u8(element as u8));
            duplex_challenger.observe(F::from_canonical_u8(element as u8));
        });

        for _ in 0..10 {
            assert_eq!(
                <TachyonDuplexChallenger<F, Perm, WIDTH, RATE> as CanSample<F>>::sample(
                    &mut tachyon_duplex_challenger
                ),
                <DuplexChallenger<F, Perm, WIDTH, RATE> as CanSample<F>>::sample(
                    &mut duplex_challenger
                )
            );
        }

        for _ in 0..10 {
            assert_eq!(
                <TachyonDuplexChallenger<F, Perm, WIDTH, RATE> as CanSample<EF>>::sample(
                    &mut tachyon_duplex_challenger
                ),
                <DuplexChallenger<F, Perm, WIDTH, RATE> as CanSample<EF>>::sample(
                    &mut duplex_challenger
                )
            );
        }
    }
}
