use std::marker::PhantomData;

use ff::{Field, PrimeField};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Expression, Selector, TableColumn},
    poly::Rotation,
};

#[derive(Clone, Default)]
struct SimpleLookupCircuit<F: Field> {
    _marker: PhantomData<F>,
}

#[derive(Clone)]
struct SimpleLookupConfig {
    selector: Selector,
    table: TableColumn,
    advice: Column<Advice>,
}

impl<F: PrimeField> Circuit<F> for SimpleLookupCircuit<F> {
    type Config = SimpleLookupConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> SimpleLookupConfig {
        let config = SimpleLookupConfig {
            selector: meta.complex_selector(),
            table: meta.lookup_table_column(),
            advice: meta.advice_column(),
        };

        meta.lookup("lookup", |meta| {
            let selector = meta.query_selector(config.selector);
            let not_selector = Expression::Constant(F::ONE) - selector.clone();
            let advice = meta.query_advice(config.advice, Rotation::cur());
            vec![(selector * advice + not_selector, config.table)]
        });

        config
    }

    fn synthesize(
        &self,
        config: SimpleLookupConfig,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        layouter.assign_table(
            || "3-bit table",
            |mut table| {
                for row in 0u64..(1 << 3) {
                    table.assign_cell(
                        || format!("row {}", row),
                        config.table,
                        row as usize,
                        || Value::known(F::from(row + 1)),
                    )?;
                }

                Ok(())
            },
        )?;

        layouter.assign_region(
            || "assign values",
            |mut region| {
                for offset in 0u64..(1 << 4) {
                    config.selector.enable(&mut region, offset as usize)?;
                    region.assign_advice(
                        || format!("offset {}", offset),
                        config.advice,
                        offset as usize,
                        || Value::known(F::from((offset % 8) + 1)),
                    )?;
                }

                Ok(())
            },
        )
    }
}

#[cfg(test)]
mod test {
    use std::marker::PhantomData;

    use halo2_proofs::{
        plonk::keygen_pk2,
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, ProverSHPLONK},
        },
        transcript::{Blake2bWrite, Challenge255, PoseidonWrite, TranscriptWriterBuffer},
    };
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use rand_core::SeedableRng;

    use crate::{
        bn254::{
            Blake2bWrite as TachyonBlake2bWrite, GWCProver, PoseidonWrite as TachyonPoseidonWrite,
            ProvingKey as TachyonProvingKey, SHPlonkProver, Sha256Write as TachyonSha256Write,
            TachyonProver,
        },
        circuits::simple_lookup_circuit::SimpleLookupCircuit,
        consts::{TranscriptType, SEED},
        prover::create_proof as tachyon_create_proof,
        sha::ShaWrite,
        xor_shift_rng::XORShiftRng,
    };

    #[test]
    fn test_create_gwc_proof() {
        // ANCHOR: test-circuit
        // The number of rows in our circuit cannot exceed 2ᵏ. Since our example
        // circuit is very small, we can pick a very small value here.
        let k = 5;

        // Instantiate the circuit.
        let circuit = SimpleLookupCircuit::<Fr> {
            _marker: PhantomData,
        };

        // Arrange the public input.
        let public_inputs = vec![];
        let public_inputs2 = vec![&public_inputs[..], &public_inputs[..]];

        // Given the correct public input, our circuit will verify.
        let s = Fr::from(2);
        let params = ParamsKZG::<Bn256>::unsafe_setup_with_s(k, s.clone());
        let pk = keygen_pk2(&params, &circuit).expect("vk should not fail");

        let rng = XORShiftRng::from_seed(SEED);

        let halo2_proof = {
            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);

            halo2_proofs::plonk::create_proof::<
                KZGCommitmentScheme<Bn256>,
                ProverGWC<_>,
                _,
                _,
                _,
                _,
            >(
                &params,
                &pk,
                &[circuit.clone(), circuit.clone()],
                public_inputs2.as_slice(),
                rng.clone(),
                &mut transcript,
            )
            .expect("proof generation should not fail");

            transcript.finalize()
        };

        let tachyon_proof = {
            let mut prover =
                GWCProver::<KZGCommitmentScheme<Bn256>>::new(TranscriptType::Blake2b as u8, k, &s);

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
                public_inputs2.as_slice(),
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
        // ANCHOR_END: test-circuit
    }

    #[test]
    fn test_create_shplonk_proof_with_various_transcripts() {
        // ANCHOR: test-circuit
        // The number of rows in our circuit cannot exceed 2ᵏ. Since our example
        // circuit is very small, we can pick a very small value here.
        let k = 5;

        // Instantiate the circuit.
        let circuit = SimpleLookupCircuit::<Fr> {
            _marker: PhantomData,
        };

        // Arrange the public input.
        let public_inputs = vec![];
        let public_inputs2 = vec![&public_inputs[..], &public_inputs[..]];

        // Given the correct public input, our circuit will verify.
        let s = Fr::from(2);
        let params = ParamsKZG::<Bn256>::unsafe_setup_with_s(k, s.clone());
        let pk = keygen_pk2(&params, &circuit).expect("vk should not fail");
        let mut pk_bytes: Vec<u8> = vec![];
        pk.write_including_cs(&mut pk_bytes).unwrap();

        let rng = XORShiftRng::from_seed(SEED);

        let halo2_blake2b_proof = {
            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);

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
                public_inputs2.as_slice(),
                rng.clone(),
                &mut transcript,
            )
            .expect("proof generation should not fail");

            transcript.finalize()
        };

        let tachyon_blake2b_proof = {
            let mut prover = SHPlonkProver::<KZGCommitmentScheme<Bn256>>::new(
                TranscriptType::Blake2b as u8,
                k,
                &s,
            );

            let mut tachyon_pk = TachyonProvingKey::from(pk_bytes.as_slice());
            let mut transcript = TachyonBlake2bWrite::init(vec![]);

            tachyon_create_proof::<_, _, _, _, _, _>(
                &mut prover,
                &mut tachyon_pk,
                &[circuit.clone(), circuit.clone()],
                public_inputs2.as_slice(),
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
            let mut transcript = PoseidonWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);

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
                public_inputs2.as_slice(),
                rng.clone(),
                &mut transcript,
            )
            .expect("proof generation should not fail");

            transcript.finalize()
        };

        let tachyon_poseidon_proof = {
            let mut prover = SHPlonkProver::<KZGCommitmentScheme<Bn256>>::new(
                TranscriptType::Poseidon as u8,
                k,
                &s,
            );

            let mut tachyon_pk = TachyonProvingKey::from(pk_bytes.as_slice());
            let mut transcript = TachyonPoseidonWrite::init(vec![]);

            tachyon_create_proof::<_, _, _, _, _, _>(
                &mut prover,
                &mut tachyon_pk,
                &[circuit.clone(), circuit.clone()],
                public_inputs2.as_slice(),
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
            let mut transcript =
                ShaWrite::<_, G1Affine, Challenge255<_>, sha2::Sha256>::init(vec![]);

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
                public_inputs2.as_slice(),
                rng.clone(),
                &mut transcript,
            )
            .expect("proof generation should not fail");

            transcript.finalize()
        };

        let tachyon_sha256_proof = {
            let mut prover = SHPlonkProver::<KZGCommitmentScheme<Bn256>>::new(
                TranscriptType::Sha256 as u8,
                k,
                &s,
            );

            let mut tachyon_pk = TachyonProvingKey::from(pk_bytes.as_slice());
            let mut transcript = TachyonSha256Write::init(vec![]);

            tachyon_create_proof::<_, _, _, _, _, _>(
                &mut prover,
                &mut tachyon_pk,
                &[circuit.clone(), circuit],
                public_inputs2.as_slice(),
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
        // ANCHOR_END: test-circuit
    }
}
