#[cfg(test)]
mod test {
    use crate::baby_bear_poseidon2::DuplexChallenger as TachyonDuplexChallenger;
    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use p3_challenger::{CanObserve, CanSample, DuplexChallenger};
    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};

    use zkhash::ark_ff::PrimeField as ark_PrimeField;
    use zkhash::fields::babybear::FpBabyBear as ark_FpBabyBear;
    use zkhash::poseidon2::poseidon2_instance_babybear::RC16;

    type F = BabyBear;

    fn baby_bear_from_ark_ff(input: ark_FpBabyBear) -> BabyBear {
        let v = input.into_bigint();
        BabyBear::from_wrapped_u32(v.0[0] as u32)
    }

    #[test]
    fn test_duplex_challenger() {
        const WIDTH: usize = 16;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 13;

        const RATE: usize = 4;

        // Copy over round constants from zkhash.
        let mut round_constants: Vec<[BabyBear; WIDTH]> = RC16
            .iter()
            .map(|vec| {
                vec.iter()
                    .cloned()
                    .map(baby_bear_from_ark_ff)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();
        let internal_start = ROUNDS_F / 2;
        let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
        let internal_round_constants = round_constants
            .drain(internal_start..internal_end)
            .map(|vec| vec[0])
            .collect::<Vec<_>>();
        let external_round_constants = round_constants;

        type Perm = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
        let perm = Perm::new(
            ROUNDS_F,
            external_round_constants,
            Poseidon2ExternalMatrixGeneral,
            ROUNDS_P,
            internal_round_constants,
            DiffusionMatrixBabyBear,
        );

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
    }
}
