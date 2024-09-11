use std::fmt::Debug;
use std::time::Instant;

use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs, PolynomialSpace};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field};
use p3_fri::{FriConfig, TwoAdicFriPcs, TwoAdicFriPcsProof};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixHL};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::log2_strict_usize;

use tachyon_rs::math::finite_fields::baby_bear::BabyBear as CppBabyBear;
use zkhash::ark_ff::PrimeField as ark_PrimeField;
use zkhash::poseidon2::poseidon2_instance_babybear::RC16 as BabyBearRC16;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2<Val, Poseidon2ExternalMatrixHL, DiffusionMatrixBabyBear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

type ValMmcs =
    FieldMerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

type Dft = Radix2DitParallel;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

fn get_perm(rounds_f: usize, rounds_p: usize) -> Perm {
    let mut round_constants: Vec<[BabyBear; 16]> = BabyBearRC16
        .iter()
        .map(|vec| {
            vec.iter()
                .cloned()
                .map(|ark_ff| BabyBear::from_canonical_u32(ark_ff.into_bigint().0[0] as u32))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect();
    let internal_start = rounds_f / 2;
    let internal_end = (rounds_f / 2) + rounds_p;
    let internal_round_constants = round_constants
        .drain(internal_start..internal_end)
        .map(|vec| vec[0])
        .collect::<Vec<_>>();
    let external_round_constants = round_constants;

    Perm::new(
        rounds_f,
        external_round_constants,
        Poseidon2ExternalMatrixHL,
        rounds_p,
        internal_round_constants,
        DiffusionMatrixBabyBear,
    )
}

fn get_pcs(log_blowup: usize, log_n: usize) -> (MyPcs, Challenger) {
    let perm = get_perm(8, 13);

    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());

    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let fri_config = FriConfig {
        log_blowup,
        num_queries: 10,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    let pcs = MyPcs::new(log_n, Dft {}, val_mmcs, fri_config);
    (pcs, Challenger::new(perm.clone()))
}

fn do_test_fri<Challenger, P>(
    (pcs, challenger): &(P, Challenger),
    degrees_by_round: Vec<Vec<usize>>,
    data: *const Val,
    batch_size: usize,
    duration: *mut u64,
) -> *mut CppBabyBear
where
    P: Pcs<Challenge, Challenger>,
    P::Domain: PolynomialSpace<Val = Val>,
    Challenge: ExtensionField<Val>,
    P: Pcs<
        Challenge,
        Challenger,
        Proof = TwoAdicFriPcsProof<Val, Challenge, ValMmcs, ChallengeMmcs>,
    >,
    Challenger: Clone + CanObserve<P::Commitment> + FieldChallenger<Val>,
    <P as Pcs<Challenge, Challenger>>::Commitment: Debug,
{
    let num_rounds = degrees_by_round.len();
    let mut p_challenger = challenger.clone();

    let domains_and_polys_by_round: Vec<Vec<_>> = degrees_by_round
        .iter()
        .enumerate()
        .map(|(r, degrees)| {
            degrees
                .iter()
                .map(|&degree| {
                    let size = degree * batch_size;
                    let values: Vec<Val> = unsafe {
                        std::slice::from_raw_parts(data.add(r) as *mut Val, size).to_vec()
                    };
                    (
                        pcs.natural_domain_for_degree(degree),
                        RowMajorMatrix::<Val>::new(values, batch_size),
                    )
                })
                .collect()
        })
        .collect();

    let start = Instant::now();
    let (commits_by_round, data_by_round): (Vec<_>, Vec<_>) = domains_and_polys_by_round
        .iter()
        .map(|domains_and_polys| pcs.commit(domains_and_polys.clone()))
        .unzip();
    assert_eq!(commits_by_round.len(), num_rounds);
    assert_eq!(data_by_round.len(), num_rounds);
    p_challenger.observe_slice(&commits_by_round);

    let zeta: Challenge = p_challenger.sample_ext_element();

    let points_by_round: Vec<_> = degrees_by_round
        .iter()
        .map(|log_degrees| vec![vec![zeta]; log_degrees.len()])
        .collect();
    let data_and_points = data_by_round.iter().zip(points_by_round).collect();
    let (_opening_by_round, proof) = pcs.open(data_and_points, &mut p_challenger);
    unsafe {
        duration.write(start.elapsed().as_micros() as u64);
    }

    let mut ret_values: [Val; 5] = [Val::zero(); 5];
    ret_values[0] = proof.fri_proof.pow_witness;
    ret_values[1..].copy_from_slice(proof.fri_proof.final_poly.as_base_slice());
    Box::into_raw(Box::new(ret_values)) as *mut CppBabyBear
}

#[no_mangle]
pub extern "C" fn run_fri_plonky3_baby_bear(
    data: *const BabyBear,
    input_num: usize,
    round_num: usize,
    max_degree: usize,
    batch_size: usize,
    log_blowup: u32,
    duration: *mut u64,
) -> *mut CppBabyBear {
    let degrees_by_round: Vec<Vec<usize>> = (0..round_num)
        .map(|r| {
            (0..input_num)
                .map(|i| max_degree >> (r + i))
                .collect::<Vec<_>>()
        })
        .collect();

    let (pcs, challenger) = get_pcs(log_blowup as usize, log2_strict_usize(max_degree));

    do_test_fri(
        &(pcs, challenger),
        degrees_by_round,
        data,
        batch_size,
        duration,
    )
}
