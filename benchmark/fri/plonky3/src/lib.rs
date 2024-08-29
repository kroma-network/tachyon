use std::fmt::Debug;
use std::time::Instant;

use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs, PolynomialSpace};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, ExtensionField, Field};
use p3_fri::{FriConfig, TwoAdicFriPcs};
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

fn do_test_fri<Val, Challenge, Challenger, P>(
    (pcs, _challenger): &(P, Challenger),
    degrees: Vec<usize>,
    data: Vec<RowMajorMatrix<Val>>,
    duration: *mut u64,
) -> *mut CppBabyBear
where
    P: Pcs<Challenge, Challenger>,
    P::Domain: PolynomialSpace<Val = Val>,
    Val: Field,
    Challenge: ExtensionField<Val>,
    Challenger: Clone + CanObserve<P::Commitment> + FieldChallenger<Val>,
    <P as Pcs<Challenge, Challenger>>::Commitment: Debug,
{
    let domains: Vec<_> = degrees
        .iter()
        .map(|&degree| pcs.natural_domain_for_degree(degree))
        .collect();
    let domains_and_polys: Vec<(P::Domain, RowMajorMatrix<Val>)> =
        domains.into_iter().zip(data.into_iter()).collect();

    let start = Instant::now();
    let (commit, _data) = pcs.commit(domains_and_polys);
    unsafe {
        duration.write(start.elapsed().as_micros() as u64);
    }
    Box::into_raw(Box::new(commit)) as *mut CppBabyBear
}

#[no_mangle]
pub extern "C" fn run_fri_plonky3_baby_bear(
    data: *const BabyBear,
    raw_degrees: *const usize,
    num_of_degrees: usize,
    batch_size: usize,
    log_blowup: u32,
    duration: *mut u64,
) -> *mut CppBabyBear {
    let degrees =
        unsafe { std::slice::from_raw_parts(raw_degrees as *mut usize, num_of_degrees).to_vec() };

    let (pcs, challenger) = get_pcs(
        log_blowup as usize,
        log2_strict_usize(*degrees.last().unwrap()),
    );

    let polys: Vec<RowMajorMatrix<Val>> = degrees
        .iter()
        .map(|&degree| {
            let size = degree * batch_size;
            let values: Vec<BabyBear> =
                unsafe { std::slice::from_raw_parts(data as *mut BabyBear, size).to_vec() };
            RowMajorMatrix::<BabyBear>::new(values, batch_size)
        })
        .collect();
    do_test_fri(&(pcs, challenger), degrees, polys, duration)
}
