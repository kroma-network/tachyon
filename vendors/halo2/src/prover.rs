// This is taken and modified from https://github.com/kroma-network/halo2/blob/9922fbb853201d8ad9feb82bd830a031d7c290b1/halo2_proofs/src/plonk/prover.rs#L37-L430.

use std::{
    collections::{BTreeSet, HashMap},
    io::Write,
    ops::RangeTo,
};

use crate::bn254::{
    AdviceSingle, Blake2bWrite as TachyonBlake2bWrite, Evals, InstanceSingle, RationalEvals,
    SHPlonkProver as TachyonSHPlonkProver, SHPlonkProvingKey as TachyonSHPlonkProvingKey,
};
use crate::xor_shift_rng::XORShiftRng as TachyonXORShiftRng;
use ff::Field;
use halo2_proofs::{
    circuit::Value,
    plonk::{
        sealed, Advice, Any, Assigned, Assignment, Challenge, Circuit, Column, ConstraintSystem,
        Error, Fixed, FloorPlanner, Instance, Selector,
    },
    poly::commitment::Blind,
    transcript::{Challenge255, Transcript, TranscriptWrite},
};
use halo2curves::{
    bn256::{Fr, G1Affine, G1},
    group::{prime::PrimeCurveAffine, Curve},
};

/// This creates a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
pub fn create_proof<'params, W: Write, ConcreteCircuit: Circuit<Fr>>(
    prover: &mut TachyonSHPlonkProver,
    pk: &mut TachyonSHPlonkProvingKey,
    circuits: &[ConcreteCircuit],
    instances: &[&[&[Fr]]],
    mut rng: TachyonXORShiftRng,
    transcript: &mut TachyonBlake2bWrite<W, G1Affine, Challenge255<G1Affine>>,
) -> Result<(), Error> {
    for instance in instances.iter() {
        if instance.len() != pk.num_instance_columns() {
            return Err(Error::InvalidInstances);
        }
    }

    prover.set_extended_domain(pk);
    // Hash verification key into transcript
    transcript.common_scalar(pk.transcript_repr(prover))?;

    let mut meta = ConstraintSystem::default();
    let config = ConcreteCircuit::configure(&mut meta);

    // Selector optimizations cannot be applied here; use the ConstraintSystem
    // from the verification key.

    let instance: Vec<InstanceSingle> = instances
        .iter()
        .map(|instance| -> Result<InstanceSingle, Error> {
            let instance_values = instance
                .iter()
                .map(|values| {
                    let mut poly = prover.empty_evals();
                    assert_eq!(poly.len(), prover.n() as usize);
                    if values.len() > (poly.len() - (pk.blinding_factors() + 1)) {
                        return Err(Error::InstanceTooLarge);
                    }

                    for i in 0..values.len() {
                        // NOTE(chokobole): I removed the P::QUERY_INSTANCE if statements since I can't make it compilable with the statement.
                        // See https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/prover.rs#L91.
                        transcript.common_scalar(values[i])?;
                        poly.set_value(i, &values[i]);
                    }
                    Ok(poly)
                })
                .collect::<Result<Vec<_>, _>>()?;

            // NOTE(chokobole): I removed the P::QUERY_INSTANCE if statements since I can't make it compilable with the statement.
            // See https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/prover.rs#L100-L117.
            let instance_polys: Vec<_> = instance_values
                .iter()
                .map(|evals| prover.ifft(evals))
                .collect();

            Ok(InstanceSingle {
                instance_values,
                instance_polys,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    struct WitnessCollection<'a, F: Field> {
        k: u32,
        current_phase: sealed::Phase,
        advice: Vec<RationalEvals>,
        challenges: &'a HashMap<usize, F>,
        instances: &'a [&'a [F]],
        usable_rows: RangeTo<usize>,
        _marker: std::marker::PhantomData<F>,
    }

    impl<'a, F: Field> Assignment<F> for WitnessCollection<'a, F> {
        fn enter_region<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about regions in this context.
        }

        fn exit_region(&mut self) {
            // Do nothing; we don't care about regions in this context.
        }

        fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Do nothing
        }

        fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            self.instances
                .get(column.index())
                .and_then(|column| column.get(row))
                .map(|v| Value::known(*v))
                .ok_or(Error::BoundsFailure)
        }

        fn assign_advice<V, VR, A, AR>(
            &mut self,
            _: A,
            column: Column<Advice>,
            row: usize,
            to: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Assigned<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Ignore assignment of advice column in different phase than current one.
            if self.current_phase.0 < column.column_type().phase.0 {
                return Ok(());
            }

            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            let rational_evals = self
                .advice
                .get_mut(column.index())
                .ok_or(Error::BoundsFailure)?;

            let value = to().into_field().assign()?;
            match &value {
                Assigned::Zero => rational_evals.set_zero(row),
                Assigned::Trivial(numerator) => {
                    let numerator = unsafe { std::mem::transmute::<_, &Fr>(numerator) };
                    rational_evals.set_trivial(row, numerator);
                }
                Assigned::Rational(numerator, denominator) => {
                    let numerator = unsafe { std::mem::transmute::<_, &Fr>(numerator) };
                    let denominator = unsafe { std::mem::transmute::<_, &Fr>(denominator) };
                    rational_evals.set_rational(row, numerator, denominator)
                }
            }

            Ok(())
        }

        fn assign_fixed<V, VR, A, AR>(
            &mut self,
            _: A,
            _: Column<Fixed>,
            _: usize,
            _: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Assigned<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn copy(
            &mut self,
            _: Column<Any>,
            _: usize,
            _: Column<Any>,
            _: usize,
        ) -> Result<(), Error> {
            // We only care about advice columns here

            Ok(())
        }

        fn fill_from_row(
            &mut self,
            _: Column<Fixed>,
            _: usize,
            _: Value<Assigned<F>>,
        ) -> Result<(), Error> {
            Ok(())
        }

        fn get_challenge(&self, challenge: Challenge) -> Value<F> {
            self.challenges
                .get(&challenge.index())
                .cloned()
                .map(Value::known)
                .unwrap_or_else(Value::unknown)
        }

        fn push_namespace<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about namespaces in this context.
        }

        fn pop_namespace(&mut self, _: Option<String>) {
            // Do nothing; we don't care about namespaces in this context.
        }
    }

    let (advice, challenges) = {
        let num_advice_columns = pk.num_advice_columns();
        let num_challenges = pk.num_challenges();
        let mut advice = vec![
            AdviceSingle {
                advice_polys: vec![prover.empty_evals(); num_advice_columns],
                advice_blinds: vec![Blind::default(); num_advice_columns],
            };
            instances.len()
        ];
        #[cfg(feature = "phase-check")]
        let mut advice_assignments =
            vec![vec![prover.empty_rational_evals(); num_advice_columns]; instances.len()];
        let mut challenges = HashMap::<usize, Fr>::with_capacity(num_challenges);

        let unusable_rows_start = prover.n() as usize - (pk.blinding_factors() + 1);
        for current_phase in pk.phases() {
            let column_indices = meta
                .advice_column_phase
                .iter()
                .enumerate()
                .filter_map(|(column_index, phase)| {
                    if current_phase == *phase {
                        Some(column_index)
                    } else {
                        None
                    }
                })
                .collect::<BTreeSet<_>>();

            for (_circuit_idx, ((circuit, advice), instances)) in circuits
                .iter()
                .zip(advice.iter_mut())
                .zip(instances)
                .enumerate()
            {
                let mut witness = WitnessCollection {
                    k: prover.k(),
                    current_phase,
                    advice: vec![prover.empty_rational_evals(); num_advice_columns],
                    instances,
                    challenges: &challenges,
                    // The prover will not be allowed to assign values to advice
                    // cells that exist within inactive rows, which include some
                    // number of blinding factors and an extra row for use in the
                    // permutation argument.
                    usable_rows: ..unusable_rows_start,
                    _marker: std::marker::PhantomData,
                };

                // Synthesize the circuit to obtain the witness and other information.
                ConcreteCircuit::FloorPlanner::synthesize(
                    &mut witness,
                    circuit,
                    config.clone(),
                    pk.constants(),
                )?;

                #[cfg(feature = "phase-check")]
                {
                    let advice_column_phases = pk.advice_column_phases();
                    for (idx, advice_col) in witness.advice.iter().enumerate() {
                        if advice_column_phases[idx].0 < current_phase.0 {
                            if advice_assignments[circuit_idx][idx].values != advice_col.values {
                                log::error!(
                                    "advice column {}(at {:?}) changed when {:?}",
                                    idx,
                                    advice_column_phases[idx],
                                    current_phase
                                );
                            }
                        }
                    }
                }

                let mut advice_assigned_values = witness
                    .advice
                    .into_iter()
                    .enumerate()
                    .filter_map(|(column_index, advice)| {
                        if column_indices.contains(&column_index) {
                            #[cfg(feature = "phase-check")]
                            {
                                advice_assignments[circuit_idx][column_index] = advice.clone();
                            }
                            Some(advice)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                let mut advice_values = vec![Evals::zero(); advice_assigned_values.len()];
                prover.batch_evaluate(
                    advice_assigned_values.as_mut_slice(),
                    advice_values.as_mut_slice(),
                );

                // Add blinding factors to advice columns
                for advice_values in &mut advice_values {
                    //for cell in &mut advice_values[unusable_rows_start..] {
                    //*cell = C::Scalar::random(&mut rng);
                    //*cell = C::Scalar::one();
                    //}
                    let idx = advice_values.len() - 1;
                    advice_values.set_value(idx, &Fr::one());
                }

                // Compute commitments to advice column polynomials
                let blinds: Vec<_> = advice_values
                    .iter()
                    .map(|_| Blind(Fr::random(&mut rng)))
                    .collect();
                let advice_commitments_projective: Vec<_> = advice_values
                    .iter()
                    .zip(blinds.iter())
                    .map(|(poly, _)| prover.commit_lagrange(poly))
                    .collect();
                let mut advice_commitments =
                    vec![G1Affine::identity(); advice_commitments_projective.len()];
                G1::batch_normalize(&advice_commitments_projective, &mut advice_commitments);
                let advice_commitments = advice_commitments;
                drop(advice_commitments_projective);

                for commitment in &advice_commitments {
                    transcript.write_point(*commitment)?;
                }
                for ((column_index, advice_values), blind) in
                    column_indices.iter().zip(advice_values).zip(blinds)
                {
                    advice.advice_polys[*column_index] = advice_values;
                    advice.advice_blinds[*column_index] = blind;
                }
            }

            for (index, phase) in pk.challenge_phases().iter().enumerate() {
                if current_phase == *phase {
                    let existing =
                        challenges.insert(index, *transcript.squeeze_challenge_scalar::<()>());
                    assert!(existing.is_none());
                }
            }
        }

        assert_eq!(challenges.len(), num_challenges);
        let challenges = (0..num_challenges)
            .map(|index| challenges.remove(&index).unwrap())
            .collect::<Vec<_>>();

        (advice, challenges)
    };

    prover.set_rng(rng.state().as_slice());
    prover.set_transcript(transcript.state().as_slice());

    let challenges = unsafe { std::mem::transmute::<_, Vec<crate::bn254::Fr>>(challenges) };
    prover.create_proof(pk, instance, advice, challenges);
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::bn254::SHPlonkProver as TachyonSHPlonkProver;
    use ff::Field;
    use halo2_proofs::poly::{
        commitment::{Blind, Params, ParamsProver},
        kzg::commitment::ParamsKZG,
        EvaluationDomain,
    };
    use halo2curves::bn256::{Bn256, Fr};
    use rand_core::OsRng;

    #[test]
    fn test_params() {
        let k = 4;
        const N: u64 = 16;
        let s = Fr::from(2);
        let params = ParamsKZG::<Bn256>::unsafe_setup_with_s(k, s.clone());
        let prover = TachyonSHPlonkProver::new(k, &s);
        assert_eq!(prover.n(), N);

        let domain = EvaluationDomain::new(1, k);
        let scalars = (0..N).map(|_| Fr::random(OsRng)).collect::<Vec<_>>();
        let mut evals = prover.empty_evals();
        for i in 0..scalars.len() {
            evals.set_value(i, &scalars[i]);
        }
        let lagrange = domain.lagrange_from_vec(scalars.clone());
        assert_eq!(
            params.commit_lagrange(&lagrange, Blind::default()),
            prover.commit_lagrange(&evals)
        );
        let cpp_poly = prover.ifft(&evals);
        let poly = domain.lagrange_to_coeff(lagrange);
        assert_eq!(
            params.commit(&poly, Blind::default()),
            prover.commit(&cpp_poly)
        );
    }
}
