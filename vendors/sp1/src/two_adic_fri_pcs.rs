#[cfg(test)]
mod test {
    use p3_commit::Pcs;
    use p3_commit::PolynomialSpace;
    use p3_dft::TwoAdicSubgroupDft;
    use p3_field::AbstractField;
    use p3_fri::TwoAdicFriPcs;
    use p3_matrix::{bitrev::BitReversableMatrix, dense::RowMajorMatrix, Matrix};
    use p3_util::log2_strict_usize;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use sp1_core::utils::baby_bear_poseidon2::{
        default_fri_config, my_perm, Challenge, ChallengeMmcs, Dft, MyCompress, MyHash, Perm, Val,
        ValMmcs,
    };

    use crate::baby_bear_poseidon2::{
        DuplexChallenger as TachyonDuplexChallenger, TwoAdicFriPcs as TachyonTwoAdicFriPcs,
    };

    type Challenger = TachyonDuplexChallenger<Val, Perm, 16, 8>;

    fn seeded_rng() -> impl Rng {
        ChaCha20Rng::seed_from_u64(0)
    }

    #[test]
    fn test_two_adic_fri_pcs() {
        const ROWS: usize = 32;
        const COLS: usize = 5;
        const LOG_N: usize = 20;

        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let dft = Dft {};
        let fri_config = default_fri_config();

        let pcs = TwoAdicFriPcs::<Val, Dft, ValMmcs, ChallengeMmcs>::new(
            LOG_N, dft, val_mmcs, fri_config,
        );

        let fri_config = default_fri_config();
        let tachyon_pcs =
            TachyonTwoAdicFriPcs::<Val, Dft, ValMmcs, ChallengeMmcs>::new(LOG_N, &fri_config);

        let mut rng = seeded_rng();
        let log_degrees_by_round = [[3, 4], [3, 4]];

        let domains_and_polys_by_round = log_degrees_by_round
            .iter()
            .map(|log_degrees| {
                log_degrees
                    .iter()
                    .map(|&log_degree| {
                        let d = 1 << log_degree;
                        // random width 5-15
                        let width = 5 + rng.gen_range(0..=10);
                        (
                            <TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs> as Pcs<
                                Challenge,
                                Challenger,
                            >>::natural_domain_for_degree(&pcs, d),
                            RowMajorMatrix::<Val>::rand(&mut rng, d, width),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let (commits_by_round, data_by_round): (Vec<_>, Vec<_>) = domains_and_polys_by_round
            .iter()
            .map(|domains_and_polys| {
                <TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs> as Pcs<
                Challenge,
                Challenger,
            >>::commit(&pcs, domains_and_polys.clone())
            })
            .unzip();

        let (tachyon_commits_by_round, tachyon_data_by_round): (Vec<_>, Vec<_>) =
            domains_and_polys_by_round
                .iter()
                .map(|domains_and_polys| {
                    <TachyonTwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs> as Pcs<
                        Challenge,
                        Challenger,
                    >>::commit(&tachyon_pcs, domains_and_polys.clone())
                })
                .unzip();

        assert_eq!(commits_by_round, tachyon_commits_by_round);

        let ldes_vec = domains_and_polys_by_round
            .iter()
            .map(|domains_and_polys| {
                domains_and_polys
                    .into_iter()
                    .map(|(domain, evals)| {
                        assert_eq!(domain.size(), evals.height());
                        let log_n = log2_strict_usize(domain.size());
                        let shift = Val::generator() / domain.shift;
                        // Commit to the bit-reversed LDE.
                        Dft {}
                            .coset_lde_batch(evals.clone(), fri_config.log_blowup, shift)
                            .bit_reverse_rows()
                            .to_row_major_matrix()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        for (i, ldes) in ldes_vec.clone().into_iter().enumerate() {
            for (j, lde) in ldes.into_iter().enumerate() {
                let v = assert_eq!(
                    lde.to_row_major_matrix(),
                    tachyon_data_by_round[i].ldes[j]
                        .clone()
                        .to_row_major_matrix()
                );
            }
        }
        // TODO(chokobole): `std::mem::forget` was used to prevent it from double-free. We need to figure out a more elegant solution.
        std::mem::forget(tachyon_data_by_round);
    }
}
