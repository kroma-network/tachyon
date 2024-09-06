pub mod baby_bear_poseidon2;
pub mod challenger;
pub mod two_adic_fri_pcs;
pub mod util;

#[cfg(test)]
mod test {
    #[cfg(not(debug_assertions))]
    use anyhow::Result;
    #[cfg(not(debug_assertions))]
    use sp1_core::io::SP1Stdin;
    #[cfg(not(debug_assertions))]
    use sp1_core::{runtime::SP1Context, utils::setup_logger};
    #[cfg(not(debug_assertions))]
    use sp1_prover::{components::DefaultProverComponents, SP1Prover};

    #[test]
    // This causes an stack overflow error on debug mode.
    #[cfg(not(debug_assertions))]
    fn test_e2e() -> Result<()> {
        use sp1_core::utils::SP1ProverOpts;

        setup_logger();
        let elf = include_bytes!("tests/fibonacci/elf/riscv32im-succinct-zkvm-elf");

        let prover = SP1Prover::<DefaultProverComponents>::new();

        let (pk, vk) = prover.setup(elf);

        let stdin = SP1Stdin::new();
        let core_proof =
            prover.prove_core(&pk, &stdin, SP1ProverOpts::default(), SP1Context::default())?;

        prover.verify(&core_proof.proof, &vk)?;

        Ok(())
    }
}
