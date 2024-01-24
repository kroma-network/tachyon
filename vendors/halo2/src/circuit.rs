#[cfg(test)]
mod test {
    use crate::bn254::{
        Blake2bWrite as TachyonBlake2bWrite, SHPlonkProver,
        SHPlonkProvingKey as TachyonSHPlonkProvingKey,
    };
    use crate::prover::create_proof as tachyon_create_proof;
    use crate::xor_shift_rng::XORShiftRng;
    use crate::{circuits::simple_circuit::SimpleCircuit, consts::SEED};
    use halo2_proofs::{
        circuit::Value,
        plonk::keygen_pk2,
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::ProverSHPLONK,
        },
        transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
    };
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use rand_core::SeedableRng;

    #[test]
    fn test_create_proof() {
        // ANCHOR: test-circuit
        // The number of rows in our circuit cannot exceed 2ᵏ. Since our example
        // circuit is very small, we can pick a very small value here.
        let k = 4;

        // Prepare the private and public inputs to the circuit!
        let constant = Fr::from(7);
        let a = Fr::from(2);
        let b = Fr::from(3);
        let c = constant * a.square() * b.square();

        // Instantiate the circuit with the private inputs.
        let circuit = SimpleCircuit {
            constant,
            a: Value::known(a),
            b: Value::known(b),
        };

        // Arrange the public input. We expose the multiplication result in row 0
        // of the instance column, so we position it there in our public inputs.
        let public_inputs = vec![c];
        let public_inputs2 = vec![&public_inputs[..]];
        let public_inputs3 = vec![&public_inputs2[..]];

        // Given the correct public input, our circuit will verify.
        let s = Fr::from(2);
        let params = ParamsKZG::<Bn256>::unsafe_setup_with_s(k, s.clone());
        let pk = keygen_pk2(&params, &circuit).expect("vk should not fail");

        let rng = XORShiftRng::from_seed(SEED);

        let halo2_proof = {
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
                &[circuit.clone()],
                public_inputs3.as_slice(),
                rng.clone(),
                &mut transcript,
            )
            .expect("proof generation should not fail");

            transcript.finalize()
        };

        let tachyon_proof = {
            let mut prover = SHPlonkProver::new(k, &s);

            let mut pk_bytes: Vec<u8> = vec![];
            pk.write(&mut pk_bytes, halo2_proofs::SerdeFormat::RawBytesUnchecked)
                .unwrap();
            let mut tachyon_pk = TachyonSHPlonkProvingKey::from(pk_bytes.as_slice());
            let mut transcript = TachyonBlake2bWrite::init(vec![]);
            let domain = &pk.vk.domain;

            tachyon_create_proof::<_, _>(
                &mut prover,
                &mut tachyon_pk,
                &[circuit],
                public_inputs3.as_slice(),
                rng,
                &mut transcript,
                &domain,
            )
            .expect("proof generation should not fail");

            transcript.finalize()
        };
        // TODO(chokobole): Need to compare `halo2_proof` and `tachyon_proof`.
        // ANCHOR_END: test-circuit
    }
}
