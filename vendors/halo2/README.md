# Halo2

## How to migrate from Halo2 to Tachyon

Tachyon can easily replace existing Halo2 implementations that generate proofs and currently supports the _BN256 (or Bn254)_ curve along with _GWC_ and _SHPlonk_ provers.

### Changing configurations

1. Replace the `halo2` library in your program with [tachyon-halo2](https://github.com/kroma-network/halo2).

   ```diff
   # In your Cargo.toml
   - halo2_proofs = { git = "https://github.com/zcash/halo2.git" }
   + halo2_proofs = { git = "https://github.com/kroma-network/halo2.git", branch = "scroll-halo2-v1.1-with-tachyon" }
   ```

2. Install the Tachyon debian package. See [Debian packaging](../../docs/how_to_use/how_to_build.md#debian-packaging).

### Editing code

1. Create the Proving Key for Tachyon

   ```diff
     use halo2_proofs::{
   +   bn254::ProvingKey as TachyonProvingKey,
       halo2curves::bn256::{Bn256, Fr},
       plonk::keygen_pk2,
       poly::kzg::commitment::ParamsKZG,
     };

     // Use just your `circuit` and `instances` so.
     let circuit = YourCircuit<_>;
     let instances = <YourInstances>

     // `k`: log of degree, `s`: toxic-waste secret.
     let k: u32 = 5;
     let s = Fr::from(2);

     // Generate `params`.
     let params = ParamsKZG<Bn256>::unsafe_setup_with_s(k, s);

     // Generate `pk` as proving key.
     let pk = keygen_pk2(&params, &circuit).expect("keygen_pk2 should not fail");
   + let (mut tachyon_pk, fixed_values) = {
   +     // Please remember to release `pk` and `pk_bytes` immediately after
   +     // deserialization, as it can occupy big memory.
   +     let mut pk_bytes: Vec<u8> = vec![];
   +     pk.write(&mut pk_bytes, halo2_proofs::SerdeFormat::RawBytesUnchecked)
   +         .unwrap();
   +     fixed_values = pk.drop_but_fixed_values();
   +     (TachyonProvingKey::from(pk_bytes.as_slice()), fixed_values)
   + };
   ```

2. Replace the transcript

   ```diff
   -  use halo2_proofs::transcript::{
   -     halo2curves::bn256::G1Affine,
   -     Challenge255, PoseidonWrite,
   - };
   + use halo2_proofs::bn254::PoseidonWrite as TachyonPoseidonWrite;

   - let mut transcript = PoseidonWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
   + let mut transcript = TachyonPoseidonWrite::init(vec![]);
   ```

   Tachyon supports `PoseidonWrite`, `Blake2bWrite`, and `Sha256Write` as transcript methods.

3. Replace the RNG (Random Number Generator)

   ```diff
   - use rand_core::{RngCore, SeedableRng};
   - use rand_xorshift::XorShiftRng;
   + use halo2_proofs::xor_shift_rng::XORShiftRng;
   + use rand_core::SeedableRng;

     let seed: [u8; 16] = <SpecificSeed>;
   - let rng = rand_xorshift::XorShiftRng::from_seed(seed);
   + let rng = XORShiftRng::from_seed(seed);
   ```

4. Create the Prover for Tachyon

   ```diff
   + use halo2_proofs::{
   +     bn254::{
   +         GWCProver as TachyonGWCProver,
   +         TachyonProver,
   +     },
   +     consts::{
   +         LSType,
   +         TranscriptType,
   +     },
   +     halo2curves::bn256::Bn256,
   +     poly::kzg::commitment::KZGCommitmentScheme,
   + }
   +
   + let mut prover = {
   +     let mut params_bytes = vec![];
   +     // Please remember to release `params` and `params_bytes` immediately after
   +     // deserialization, as it can occupy big memory.
   +     params.write(&mut params_bytes).unwrap();
   +     drop(params);
   +     TachyonGWCProver::<KZGCommitmentScheme<Bn256>>::from_params(
   +         LSType::Halo2 as u8,
   +         TranscriptType::Poseidon as u8,
   +         k,
   +         params_bytes.as_slice(),
   +     )
   + };
   ```

   Tachyon supports `GWCProver` or `SHPlonkProver` as a prover. Note that the first argument of `XXXProver::from_params()` and `transcript` must refer to the same type.

5. Create the Proof with Tachyon

   ```diff
     use halo2_proofs::{
   -     halo2curves::bn256::Bn256,
   -     poly::kzg::{commitment::KZGCommitmentScheme, multiopen::ProverSHPLONK},
   +     plonk::tachyon::create_proof as tachyon_create_proof,
     }

   +

   - create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<Bn256>, _, _, _, _>(
   -     &params,
   -     &pk,
   + tachyon_create_proof::<_, _, _, _, _>(
   +     &mut prover,
   +     &mut tachyon_pk,
         &[&circuit],
         &[&instances],
   +     fixed_values,
         rng,
         &mut transcript,
     )
     .expect("proof generation should not fail");

   - let proof = transcript.finalize();
   + let mut proof = transcript.finalize();
   + let proof_last = prover.get_proof();
   + proof.extend_from_slice(&proof_last);
   ```
