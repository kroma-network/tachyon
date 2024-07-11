#[cfg(test)]
mod test {
    use crate::bn254::ProvingKey;
    use crate::circuits::simple_circuit::SimpleCircuit;
    use halo2_proofs::{circuit::Value, plonk::keygen_pk2, poly::kzg::commitment::ParamsKZG};
    use halo2curves::bn256::{Bn256, Fr, G1Affine};

    #[test]
    fn test_proving_key() {
        // ANCHOR: test-circuit
        // The number of rows in our circuit cannot exceed 2ᵏ. Since our example
        // circuit is very small, we can pick a very small value here.
        let k = 4;

        // Prepare the private and public inputs to the circuit!
        let constant = Fr::from(7);
        let a = Fr::from(2);
        let b = Fr::from(3);

        // Instantiate the circuit with the private inputs.
        let circuit = SimpleCircuit {
            constant,
            a: Value::known(a),
            b: Value::known(b),
        };

        // Given the correct public input, our circuit will verify.
        let s = Fr::from(2);
        let params = ParamsKZG::<Bn256>::unsafe_setup_with_s(k, s);
        let pk = keygen_pk2(&params, &circuit).expect("vk should not fail");
        let mut pk_bytes: Vec<u8> = vec![];
        pk.write_including_cs(&mut pk_bytes).unwrap();
        let tachyon_pk = ProvingKey::<G1Affine>::from(pk_bytes.as_slice());

        assert_eq!(
            pk.get_vk().cs().advice_column_phase,
            tachyon_pk.advice_column_phases()
        );
        assert_eq!(
            pk.get_vk().cs().blinding_factors(),
            tachyon_pk.blinding_factors() as usize
        );
        assert_eq!(
            pk.get_vk().cs().challenge_phase,
            tachyon_pk.challenge_phases()
        );
        assert_eq!(*pk.get_vk().cs().constants(), tachyon_pk.constants());
        assert_eq!(
            pk.get_vk().cs().num_advice_columns,
            tachyon_pk.num_advice_columns()
        );
        assert_eq!(
            pk.get_vk().cs().num_challenges(),
            tachyon_pk.num_challenges()
        );
        assert_eq!(
            pk.get_vk().cs().num_instance_columns,
            tachyon_pk.num_instance_columns()
        );
        let phases = pk.get_vk().cs().phases().collect::<Vec<_>>();
        assert_eq!(phases, tachyon_pk.phases());
    }
}
