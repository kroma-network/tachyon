use std::iter;

use ff::BatchInvert;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Challenge, Circuit, Column, ConstraintSystem, Error, Expression, FirstPhase,
        SecondPhase, Selector,
    },
    poly::Rotation,
};
use rand_core::RngCore;

fn rand_2d_array<F: Field, R: RngCore, const W: usize, const H: usize>(rng: &mut R) -> [[F; H]; W] {
    [(); W].map(|_| [(); H].map(|_| F::random(&mut *rng)))
}

fn shuffled<F: Field, R: RngCore, const W: usize, const H: usize>(
    original: [[F; H]; W],
    rng: &mut R,
) -> [[F; H]; W] {
    let mut shuffled = original;

    for row in (1..H).rev() {
        let rand_row = (rng.next_u32() as usize) % row;
        for column in shuffled.iter_mut() {
            column.swap(row, rand_row);
        }
    }

    shuffled
}

#[derive(Clone)]
struct MyConfig<const W: usize> {
    q_shuffle: Selector,
    q_first: Selector,
    q_last: Selector,
    original: [Column<Advice>; W],
    shuffled: [Column<Advice>; W],
    theta: Challenge,
    gamma: Challenge,
    z: Column<Advice>,
}

impl<const W: usize> MyConfig<W> {
    fn configure<F: Field>(meta: &mut ConstraintSystem<F>) -> Self {
        let [q_shuffle, q_first, q_last] = [(); 3].map(|_| meta.selector());
        // First phase
        let original = [(); W].map(|_| meta.advice_column_in(FirstPhase));
        let shuffled = [(); W].map(|_| meta.advice_column_in(FirstPhase));
        let [theta, gamma] = [(); 2].map(|_| meta.challenge_usable_after(FirstPhase));
        // Second phase
        let z = meta.advice_column_in(SecondPhase);

        meta.create_gate("z should start with 1", |meta| {
            let q_first = meta.query_selector(q_first);
            let z = meta.query_advice(z, Rotation::cur());
            let one = Expression::Constant(F::ONE);

            vec![q_first * (one - z)]
        });

        meta.create_gate("z should end with 1", |meta| {
            let q_last = meta.query_selector(q_last);
            let z = meta.query_advice(z, Rotation::cur());
            let one = Expression::Constant(F::ONE);

            vec![q_last * (one - z)]
        });

        meta.create_gate("z should have valid transition", |meta| {
            let q_shuffle = meta.query_selector(q_shuffle);
            let original = original.map(|advice| meta.query_advice(advice, Rotation::cur()));
            let shuffled = shuffled.map(|advice| meta.query_advice(advice, Rotation::cur()));
            let [theta, gamma] = [theta, gamma].map(|challenge| meta.query_challenge(challenge));
            let [z, z_w] =
                [Rotation::cur(), Rotation::next()].map(|rotation| meta.query_advice(z, rotation));

            // Compress
            let original = original
                .iter()
                .cloned()
                .reduce(|acc, a| acc * theta.clone() + a)
                .unwrap();
            let shuffled = shuffled
                .iter()
                .cloned()
                .reduce(|acc, a| acc * theta.clone() + a)
                .unwrap();

            vec![q_shuffle * (z * (original + gamma.clone()) - z_w * (shuffled + gamma))]
        });

        Self {
            q_shuffle,
            q_first,
            q_last,
            original,
            shuffled,
            theta,
            gamma,
            z,
        }
    }
}

#[derive(Clone, Default)]
struct MyCircuit<F: Field, const W: usize, const H: usize> {
    original: Value<[[F; H]; W]>,
    shuffled: Value<[[F; H]; W]>,
}

impl<F: Field, const W: usize, const H: usize> MyCircuit<F, W, H> {
    fn rand<R: RngCore>(rng: &mut R) -> Self {
        let original = rand_2d_array::<F, _, W, H>(rng);
        let shuffled = shuffled(original, rng);

        Self {
            original: Value::known(original),
            shuffled: Value::known(shuffled),
        }
    }
}

impl<F: Field, const W: usize, const H: usize> Circuit<F> for MyCircuit<F, W, H> {
    type Config = MyConfig<W>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        MyConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let theta = layouter.get_challenge(config.theta);
        let gamma = layouter.get_challenge(config.gamma);

        layouter.assign_region(
            || "Shuffle original into shuffled",
            |mut region| {
                // Keygen
                config.q_first.enable(&mut region, 0)?;
                config.q_last.enable(&mut region, H)?;
                for offset in 0..H {
                    config.q_shuffle.enable(&mut region, offset)?;
                }

                // First phase
                for (idx, (&column, values)) in config
                    .original
                    .iter()
                    .zip(self.original.transpose_array().iter())
                    .enumerate()
                {
                    for (offset, &value) in values.transpose_array().iter().enumerate() {
                        region.assign_advice(
                            || format!("original[{}][{}]", idx, offset),
                            column,
                            offset,
                            || value,
                        )?;
                    }
                }
                for (idx, (&column, values)) in config
                    .shuffled
                    .iter()
                    .zip(self.shuffled.transpose_array().iter())
                    .enumerate()
                {
                    for (offset, &value) in values.transpose_array().iter().enumerate() {
                        region.assign_advice(
                            || format!("shuffled[{}][{}]", idx, offset),
                            column,
                            offset,
                            || value,
                        )?;
                    }
                }

                // Second phase
                let z = self.original.zip(self.shuffled).zip(theta).zip(gamma).map(
                    |(((original, shuffled), theta), gamma)| {
                        let mut product = vec![F::ZERO; H];
                        for (idx, product) in product.iter_mut().enumerate() {
                            let mut compressed = F::ZERO;
                            for value in shuffled.iter() {
                                compressed *= theta;
                                compressed += value[idx];
                            }

                            *product = compressed + gamma
                        }

                        product.iter_mut().batch_invert();

                        for (idx, product) in product.iter_mut().enumerate() {
                            let mut compressed = F::ZERO;
                            for value in original.iter() {
                                compressed *= theta;
                                compressed += value[idx];
                            }

                            *product *= compressed + gamma
                        }

                        #[allow(clippy::let_and_return)]
                        let z = iter::once(F::ONE)
                            .chain(product)
                            .scan(F::ONE, |state, cur| {
                                *state *= &cur;
                                Some(*state)
                            })
                            .collect::<Vec<_>>();

                        #[cfg(feature = "sanity-checks")]
                        assert_eq!(F::ONE, *z.last().unwrap());

                        z
                    },
                );
                for (offset, value) in z.transpose_vec(H + 1).into_iter().enumerate() {
                    region.assign_advice(
                        || format!("z[{}]", offset),
                        config.z,
                        offset,
                        || value,
                    )?;
                }

                Ok(())
            },
        )
    }
}

#[cfg(test)]
mod test {
    use halo2_proofs::{
        plonk::{create_proof, keygen_pk2},
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, ProverSHPLONK},
        },
        transcript::{Blake2bWrite, Challenge255, PoseidonWrite, TranscriptWriterBuffer},
    };
    use halo2curves::bn256::{Bn256, Fr};
    use rand_core::SeedableRng;

    use crate::{
        bn254::{
            Blake2bWrite as TachyonBlake2bWrite, GWCProver, PoseidonWrite as TachyonPoseidonWrite,
            ProvingKey as TachyonProvingKey, SHPlonkProver, Sha256Write as TachyonSha256Write,
            TachyonProver,
        },
        consts::{TranscriptType, SEED},
        prover::create_proof as tachyon_create_proof,
        sha::ShaWrite,
        xor_shift_rng::XORShiftRng,
    };

    use super::MyCircuit;

    const W: usize = 2;
    const H: usize = 8;
    const K: u32 = 4;

    #[test]
    fn test_create_gwc_proof() {
        let mut rng_for_table = XORShiftRng::from_seed(SEED);
        let circuit = MyCircuit::<Fr, W, H>::rand(&mut rng_for_table);

        let s = Fr::from(2);
        let params = ParamsKZG::<Bn256>::unsafe_setup_with_s(K, s);
        let pk = keygen_pk2(&params, &circuit).unwrap();

        let rng = XORShiftRng::from_seed(SEED);

        let halo2_proof = {
            let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

            create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, _, _>(
                &params,
                &pk,
                &[circuit.clone(), circuit.clone()],
                &[&[], &[]],
                rng.clone(),
                &mut transcript,
            )
            .expect("proof generation should not fail");

            transcript.finalize()
        };

        let tachyon_proof = {
            let mut prover =
                GWCProver::<KZGCommitmentScheme<Bn256>>::new(TranscriptType::Blake2b as u8, K, &s);

            let (mut tachyon_pk, fixed_values) = {
                let mut pk_bytes: Vec<u8> = vec![];
                pk.write_including_cs(&mut pk_bytes).unwrap();
                let fixed_values = pk.drop_but_fixed_values();
                (TachyonProvingKey::from(pk_bytes.as_slice()), fixed_values)
            };
            let mut transcript = TachyonBlake2bWrite::init(vec![]);

            tachyon_create_proof::<_, _, _, _, _, _>(
                &mut prover,
                &mut tachyon_pk,
                &[circuit.clone(), circuit],
                &[&[], &[]],
                fixed_values,
                rng,
                &mut transcript,
            )
            .expect("proof generation should not fail");

            let mut proof = transcript.finalize();
            let proof_last = prover.get_proof();
            proof.extend_from_slice(&proof_last);
            proof
        };
        assert_eq!(halo2_proof, tachyon_proof);
    }

    #[test]
    fn test_create_shplonk_proof_with_various_transcripts() {
        let mut rng_for_table = XORShiftRng::from_seed(SEED);
        let circuit = MyCircuit::<Fr, W, H>::rand(&mut rng_for_table);

        let s = Fr::from(2);
        let params = ParamsKZG::<Bn256>::unsafe_setup_with_s(K, s);
        let pk = keygen_pk2(&params, &circuit).unwrap();
        let mut pk_bytes: Vec<u8> = vec![];
        pk.write_including_cs(&mut pk_bytes).unwrap();

        let rng = XORShiftRng::from_seed(SEED);

        let halo2_blake2b_proof = {
            let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

            halo2_proofs::plonk::create_proof::<
                KZGCommitmentScheme<Bn256>,
                ProverSHPLONK<_>,
                _,
                _,
                _,
                _,
            >(
                &params,
                &pk,
                &[circuit.clone(), circuit.clone()],
                &[&[], &[]],
                rng.clone(),
                &mut transcript,
            )
            .expect("proof generation should not fail");

            transcript.finalize()
        };

        let tachyon_blake2b_proof = {
            let mut prover = SHPlonkProver::<KZGCommitmentScheme<Bn256>>::new(
                TranscriptType::Blake2b as u8,
                K,
                &s,
            );

            let mut tachyon_pk = TachyonProvingKey::from(pk_bytes.as_slice());
            let mut transcript = TachyonBlake2bWrite::init(vec![]);

            tachyon_create_proof::<_, _, _, _, _, _>(
                &mut prover,
                &mut tachyon_pk,
                &[circuit.clone(), circuit.clone()],
                &[&[], &[]],
                pk.fixed_values.clone(),
                rng.clone(),
                &mut transcript,
            )
            .expect("proof generation should not fail");

            let mut proof = transcript.finalize();
            let proof_last = prover.get_proof();
            proof.extend_from_slice(&proof_last);
            proof
        };
        assert_eq!(halo2_blake2b_proof, tachyon_blake2b_proof);

        let halo2_poseidon_proof = {
            let mut transcript = PoseidonWrite::<_, _, Challenge255<_>>::init(vec![]);

            halo2_proofs::plonk::create_proof::<
                KZGCommitmentScheme<Bn256>,
                ProverSHPLONK<_>,
                _,
                _,
                _,
                _,
            >(
                &params,
                &pk,
                &[circuit.clone(), circuit.clone()],
                &[&[], &[]],
                rng.clone(),
                &mut transcript,
            )
            .expect("proof generation should not fail");

            transcript.finalize()
        };

        let tachyon_poseidon_proof = {
            let mut prover = SHPlonkProver::<KZGCommitmentScheme<Bn256>>::new(
                TranscriptType::Poseidon as u8,
                K,
                &s,
            );

            let mut tachyon_pk = TachyonProvingKey::from(pk_bytes.as_slice());
            let mut transcript = TachyonPoseidonWrite::init(vec![]);

            tachyon_create_proof::<_, _, _, _, _, _>(
                &mut prover,
                &mut tachyon_pk,
                &[circuit.clone(), circuit.clone()],
                &[&[], &[]],
                pk.fixed_values.clone(),
                rng.clone(),
                &mut transcript,
            )
            .expect("proof generation should not fail");

            let mut proof = transcript.finalize();
            let proof_last = prover.get_proof();
            proof.extend_from_slice(&proof_last);
            proof
        };
        assert_eq!(halo2_poseidon_proof, tachyon_poseidon_proof);

        let halo2_sha256_proof = {
            let mut transcript = ShaWrite::<_, _, Challenge255<_>, sha2::Sha256>::init(vec![]);

            halo2_proofs::plonk::create_proof::<
                KZGCommitmentScheme<Bn256>,
                ProverSHPLONK<_>,
                _,
                _,
                _,
                _,
            >(
                &params,
                &pk,
                &[circuit.clone(), circuit.clone()],
                &[&[], &[]],
                rng.clone(),
                &mut transcript,
            )
            .expect("proof generation should not fail");

            transcript.finalize()
        };

        let tachyon_sha256_proof = {
            let mut prover = SHPlonkProver::<KZGCommitmentScheme<Bn256>>::new(
                TranscriptType::Sha256 as u8,
                K,
                &s,
            );

            let mut tachyon_pk = TachyonProvingKey::from(pk_bytes.as_slice());
            let mut transcript = TachyonSha256Write::init(vec![]);

            tachyon_create_proof::<_, _, _, _, _, _>(
                &mut prover,
                &mut tachyon_pk,
                &[circuit.clone(), circuit],
                &[&[], &[]],
                pk.fixed_values,
                rng,
                &mut transcript,
            )
            .expect("proof generation should not fail");

            let mut proof = transcript.finalize();
            let proof_last = prover.get_proof();
            proof.extend_from_slice(&proof_last);
            proof
        };
        assert_eq!(halo2_sha256_proof, tachyon_sha256_proof);
    }
}
