#[cfg(test)]
mod test {
    use anyhow::Result;
    use sp1_core::io::SP1Stdin;
    use sp1_core::utils::setup_logger;
    use sp1_prover::SP1Prover;

    #[test]
    // This causes an stack overflow error on debug mode.
    #[cfg(not(debug_assertions))]
    fn test_e2e() -> Result<()> {
        setup_logger();
        let elf = include_bytes!("tests/fibonacci/elf/riscv32im-succinct-zkvm-elf");

        let prover = SP1Prover::new();

        let (pk, vk) = prover.setup(elf);

        let stdin = SP1Stdin::new();
        let core_proof = prover.prove_core(&pk, &stdin)?;

        prover.verify(&core_proof.proof, &vk)?;

        Ok(())
    }
}
