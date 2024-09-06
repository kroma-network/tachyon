use std::{fmt::Debug, io::Cursor, marker::PhantomData, pin::Pin};

use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, Pcs, PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PackedField, PackedValue, PrimeField64, TwoAdicField};
use p3_fri::{FriConfig, VerificationError};
use p3_matrix::{
    bitrev::BitReversableMatrix,
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicPermutation, Hash};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use tachyon_rs::math::finite_fields::{baby_bear::BabyBear as TachyonBabyBearImpl, Fp4};
use tracing::instrument;

use crate::util::Readable;

pub struct TachyonBabyBear(pub TachyonBabyBearImpl);
pub struct TachyonBabyBear4(pub Fp4<TachyonBabyBearImpl>);

#[cxx::bridge(namespace = "tachyon::sp1_api::baby_bear_poseidon2")]
pub mod ffi {
    extern "Rust" {
        type TachyonBabyBear;
        type TachyonBabyBear4;
    }

    unsafe extern "C++" {
        include!("vendors/sp1/include/baby_bear_poseidon2_commitment_vec.h");

        type CommitmentVec;

        fn new_commitment_vec(rounds: usize) -> UniquePtr<CommitmentVec>;
        fn set(self: Pin<&mut CommitmentVec>, round: usize, commitment: &[TachyonBabyBear]);
    }

    unsafe extern "C++" {
        include!("vendors/sp1/include/baby_bear_poseidon2_domains.h");

        type Domains;

        fn new_domains(rounds: usize) -> UniquePtr<Domains>;
        fn allocate(self: Pin<&mut Domains>, round: usize, size: usize);
        fn set(
            self: Pin<&mut Domains>,
            round: usize,
            idx: usize,
            log_n: u32,
            shift: &TachyonBabyBear,
        );
    }

    unsafe extern "C++" {
        include!("vendors/sp1/include/baby_bear_poseidon2_duplex_challenger.h");

        type DuplexChallenger;

        fn new_duplex_challenger() -> UniquePtr<DuplexChallenger>;
        fn observe(self: Pin<&mut DuplexChallenger>, value: &TachyonBabyBear);
        fn sample(self: Pin<&mut DuplexChallenger>) -> Box<TachyonBabyBear>;
        fn clone(&self) -> UniquePtr<DuplexChallenger>;
    }

    unsafe extern "C++" {
        include!("vendors/sp1/include/baby_bear_poseidon2_fri_proof.h");

        type FriProof;

        fn clone(&self) -> UniquePtr<FriProof>;
    }

    unsafe extern "C++" {
        include!("vendors/sp1/include/baby_bear_poseidon2_opened_values.h");

        type OpenedValues;

        fn new_opened_values(rounds: usize) -> UniquePtr<OpenedValues>;
        fn allocate_outer(self: Pin<&mut OpenedValues>, round: usize, rows: usize, cols: usize);
        fn allocate_inner(
            self: Pin<&mut OpenedValues>,
            round: usize,
            row: usize,
            cols: usize,
            size: usize,
        );
        fn set(
            self: Pin<&mut OpenedValues>,
            round: usize,
            row: usize,
            col: usize,
            idx: usize,
            value: &TachyonBabyBear4,
        );
    }

    unsafe extern "C++" {
        include!("vendors/sp1/include/baby_bear_poseidon2_opening_points.h");

        type OpeningPoints;

        fn new_opening_points(rounds: usize) -> UniquePtr<OpeningPoints>;
        fn clone(&self) -> UniquePtr<OpeningPoints>;
        fn allocate(self: Pin<&mut OpeningPoints>, round: usize, rows: usize, cols: usize);
        fn set(
            self: Pin<&mut OpeningPoints>,
            round: usize,
            row: usize,
            col: usize,
            point: &TachyonBabyBear4,
        );
    }

    unsafe extern "C++" {
        include!("vendors/sp1/include/baby_bear_poseidon2_opening_proof.h");

        type OpeningProof;

        fn serialize_to_opened_values(&self) -> Vec<u8>;
        fn take_fri_proof(self: Pin<&mut OpeningProof>) -> UniquePtr<FriProof>;
    }

    unsafe extern "C++" {
        include!("vendors/sp1/include/baby_bear_poseidon2_prover_data.h");

        type ProverData;

        fn write_commit(&self, values: &mut [TachyonBabyBear]);
        fn clone(&self) -> UniquePtr<ProverData>;
    }

    unsafe extern "C++" {
        include!("vendors/sp1/include/baby_bear_poseidon2_prover_data_vec.h");

        type ProverDataVec;

        fn new_prover_data_vec() -> UniquePtr<ProverDataVec>;
        fn clone(&self) -> UniquePtr<ProverDataVec>;
    }

    unsafe extern "C++" {
        include!("vendors/sp1/include/baby_bear_poseidon2_two_adic_fri_pcs.h");

        type TwoAdicFriPcs;

        fn new_two_adic_fri_pcs(
            log_blowup: usize,
            num_queries: usize,
            proof_of_work_bits: usize,
        ) -> UniquePtr<TwoAdicFriPcs>;
        fn allocate_ldes(&self, size: usize);
        fn coset_lde_batch(
            &self,
            values: &mut [TachyonBabyBear],
            cols: usize,
            shift: &TachyonBabyBear,
        ) -> &mut [TachyonBabyBear];
        fn commit(&self, prover_data_vec: &ProverDataVec) -> UniquePtr<ProverData>;
        fn do_open(
            &self,
            prover_data_vec: &ProverDataVec,
            opening_points: &OpeningPoints,
            challenger: Pin<&mut DuplexChallenger>,
        ) -> UniquePtr<OpeningProof>;
        fn do_verify(
            &self,
            commitments: &CommitmentVec,
            domains: &Domains,
            opening_points: &OpeningPoints,
            opened_values: &OpenedValues,
            proof: &FriProof,
            challenger: Pin<&mut DuplexChallenger>,
        ) -> bool;
    }
}

pub struct CommitmentVec<Val> {
    inner: cxx::UniquePtr<ffi::CommitmentVec>,
    _marker: PhantomData<Val>,
}

impl<Val> Debug for CommitmentVec<Val> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommitmentVec").finish()
    }
}

impl<Val> CommitmentVec<Val> {
    pub fn new(inner: cxx::UniquePtr<ffi::CommitmentVec>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    pub fn set(&mut self, round: usize, commitment: &[Val]) {
        self.inner
            .pin_mut()
            .set(round, unsafe { std::mem::transmute(commitment) })
    }
}

pub struct Domains<Val> {
    inner: cxx::UniquePtr<ffi::Domains>,
    _marker: PhantomData<Val>,
}

impl<Val> Debug for Domains<Val> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Domains").finish()
    }
}

impl<Val> Domains<Val> {
    pub fn new(inner: cxx::UniquePtr<ffi::Domains>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    pub fn allocate(&mut self, round: usize, size: usize) {
        self.inner.pin_mut().allocate(round, size)
    }

    pub fn set(&mut self, round: usize, idx: usize, log_n: u32, shift: Val) {
        self.inner
            .pin_mut()
            .set(round, idx, log_n, unsafe { std::mem::transmute(&shift) })
    }
}

pub struct DuplexChallenger<F, P, const WIDTH: usize, const RATE: usize> {
    inner: cxx::UniquePtr<ffi::DuplexChallenger>,
    _marker: PhantomData<(F, P)>,
}

// NOTE(chokobole): This is needed by `GrindingChallenger` trait.
// See https://github.com/Plonky3/Plonky3/blob/eeb4e37/challenger/src/grinding_challenger.rs#L8-L9.
unsafe impl Sync for ffi::DuplexChallenger {}

impl<F, P, const WIDTH: usize, const RATE: usize> Clone for DuplexChallenger<F, P, WIDTH, RATE> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _marker: PhantomData,
        }
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> Debug for DuplexChallenger<F, P, WIDTH, RATE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuplexChallenger").finish()
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> DuplexChallenger<F, P, WIDTH, RATE> {
    pub fn new() -> DuplexChallenger<F, P, WIDTH, RATE> {
        DuplexChallenger {
            inner: ffi::new_duplex_challenger(),
            _marker: PhantomData,
        }
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> FieldChallenger<F>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: PrimeField64,
    P: CryptographicPermutation<[F; WIDTH]>,
{
}

impl<F, P, const WIDTH: usize, const RATE: usize> CanObserve<F>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, value: F) {
        self.inner
            .pin_mut()
            .observe(unsafe { std::mem::transmute(&value) });
    }
}

impl<F, P, const N: usize, const WIDTH: usize, const RATE: usize> CanObserve<[F; N]>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, values: [F; N]) {
        for value in values {
            self.observe(value);
        }
    }
}

impl<F, P, const N: usize, const WIDTH: usize, const RATE: usize> CanObserve<Hash<F, F, N>>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, values: Hash<F, F, N>) {
        for value in values {
            self.observe(value);
        }
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> CanObserve<Vec<Vec<F>>>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, valuess: Vec<Vec<F>>) {
        for values in valuess {
            for value in values {
                self.observe(value);
            }
        }
    }
}

impl<F, EF, P, const WIDTH: usize, const RATE: usize> CanSample<EF>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn sample(&mut self) -> EF {
        EF::from_base_fn(|_| *unsafe {
            std::mem::transmute::<_, Box<F>>(self.inner.pin_mut().sample())
        })
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> CanSampleBits<usize>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: PrimeField64,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        debug_assert!(bits < (usize::BITS as usize));
        debug_assert!((1 << bits) < F::ORDER_U64);
        let rand_f: F = self.sample();
        let rand_usize = rand_f.as_canonical_u64() as usize;
        rand_usize & ((1 << bits) - 1)
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> GrindingChallenger
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: PrimeField64,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    type Witness = F;

    #[instrument(name = "grind for proof-of-work witness", skip_all)]
    fn grind(&mut self, bits: usize) -> Self::Witness {
        let witness = (0..F::ORDER_U64)
            .into_par_iter()
            .map(|i| F::from_canonical_u64(i))
            .find_any(|witness| self.clone().check_witness(bits, *witness))
            .expect("failed to find witness");
        assert!(self.check_witness(bits, witness));
        witness
    }
}

pub trait HasCXXDuplexChallenger {
    fn get_inner_pin_mut(&mut self) -> Pin<&mut ffi::DuplexChallenger>;
}

impl<F, P, const WIDTH: usize, const RATE: usize> HasCXXDuplexChallenger
    for DuplexChallenger<F, P, WIDTH, RATE>
{
    fn get_inner_pin_mut(&mut self) -> Pin<&mut ffi::DuplexChallenger> {
        self.inner.pin_mut()
    }
}

pub struct FriProof {
    inner: cxx::UniquePtr<ffi::FriProof>,
}

impl Serialize for FriProof {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        todo!("Not implemented yet")
    }
}

impl<'de> Deserialize<'de> for FriProof {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        todo!("Not implemented yet")
    }
}

impl Clone for FriProof {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl FriProof {
    pub fn new(inner: cxx::UniquePtr<ffi::FriProof>) -> Self {
        Self { inner }
    }
}

pub struct OpenedValues<Val> {
    inner: cxx::UniquePtr<ffi::OpenedValues>,
    _marker: PhantomData<Val>,
}

impl<Val> OpenedValues<Val> {
    pub fn new(inner: cxx::UniquePtr<ffi::OpenedValues>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    pub fn allocate_outer(&mut self, round: usize, rows: usize, cols: usize) {
        self.inner.pin_mut().allocate_outer(round, rows, cols)
    }

    pub fn allocate_inner(&mut self, round: usize, row: usize, cols: usize, size: usize) {
        self.inner.pin_mut().allocate_inner(round, row, cols, size)
    }

    pub fn set(&mut self, round: usize, row: usize, col: usize, idx: usize, value: &Val) {
        self.inner
            .pin_mut()
            .set(round, row, col, idx, unsafe { std::mem::transmute(value) })
    }
}

pub struct OpeningPoints<Val> {
    inner: cxx::UniquePtr<ffi::OpeningPoints>,
    _marker: PhantomData<Val>,
}

impl<Val: Clone> Clone for OpeningPoints<Val> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Val> OpeningPoints<Val> {
    pub fn new(inner: cxx::UniquePtr<ffi::OpeningPoints>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    pub fn allocate(&mut self, round: usize, rows: usize, cols: usize) {
        self.inner.pin_mut().allocate(round, rows, cols)
    }

    pub fn set(&mut self, round: usize, row: usize, col: usize, point: &Val) {
        self.inner
            .pin_mut()
            .set(round, row, col, unsafe { std::mem::transmute(point) })
    }
}
pub struct OpeningProof {
    inner: cxx::UniquePtr<ffi::OpeningProof>,
}

impl OpeningProof {
    pub fn new(inner: cxx::UniquePtr<ffi::OpeningProof>) -> Self {
        Self { inner }
    }

    pub fn serialize_to_opened_values<Challenge>(&self) -> p3_commit::OpenedValues<Challenge> {
        let buffer = self.inner.serialize_to_opened_values();
        let mut reader = Cursor::new(buffer);
        let values = p3_commit::OpenedValues::<[u32; 4]>::read_from(&mut reader).unwrap();
        unsafe { std::mem::transmute(values) }
    }

    pub fn take_fri_proof(&mut self) -> FriProof {
        FriProof::new(self.inner.pin_mut().take_fri_proof())
    }
}

pub struct ProverData<Val> {
    inner: cxx::UniquePtr<ffi::ProverData>,
    pub ldes: Vec<DenseMatrix<Val>>,
}

impl<Val: Clone> Clone for ProverData<Val> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            ldes: self.ldes.clone(),
        }
    }
}

impl<Val> Debug for ProverData<Val> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProverData").finish()
    }
}

impl<Val> ProverData<Val> {
    pub fn new(inner: cxx::UniquePtr<ffi::ProverData>) -> Self {
        Self {
            inner,
            ldes: vec![],
        }
    }

    pub fn write_commit(&self, values: &mut [BabyBear]) {
        self.inner
            .write_commit(unsafe { std::mem::transmute(values) })
    }
}

pub struct ProverDataVec<Val> {
    inner: cxx::UniquePtr<ffi::ProverDataVec>,
    _marker: PhantomData<Val>,
}

impl<Val: Clone> Clone for ProverDataVec<Val> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Val> Debug for ProverDataVec<Val> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProverDataVec").finish()
    }
}

impl<Val> ProverDataVec<Val> {
    pub fn new(inner: cxx::UniquePtr<ffi::ProverDataVec>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

pub struct TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    log_n: usize,
    inner: cxx::UniquePtr<ffi::TwoAdicFriPcs>,
    prover_data_vec: ProverDataVec<Val>,
    _marker: PhantomData<(Val, Dft, InputMmcs, FriMmcs)>,
}

impl<Val, Dft, InputMmcs, FriMmcs> Debug for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwoAdicFriPcs").finish()
    }
}

impl<Val, Dft, InputMmcs, FriMmcs> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField,
{
    pub fn new(
        log_n: usize,
        fri_config: &FriConfig<FriMmcs>,
    ) -> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
        Self {
            log_n,
            inner: ffi::new_two_adic_fri_pcs(
                fri_config.log_blowup,
                fri_config.num_queries,
                fri_config.proof_of_work_bits,
            ),
            prover_data_vec: ProverDataVec::new(ffi::new_prover_data_vec()),
            _marker: PhantomData,
        }
    }

    pub fn allocate_ldes(&self, size: usize) {
        self.inner.allocate_ldes(size)
    }

    pub fn coset_lde_batch(
        &self,
        evals: &mut RowMajorMatrix<Val>,
        shift: Val,
    ) -> RowMajorMatrix<Val> {
        unsafe {
            let data = self.inner.coset_lde_batch(
                std::mem::transmute(evals.values.as_mut_slice()),
                evals.width,
                std::mem::transmute(&shift),
            );

            let vec = Vec::from_raw_parts(
                std::mem::transmute(data.as_mut_ptr()),
                data.len(),
                data.len(),
            );
            RowMajorMatrix::new(vec, evals.width)
        }
    }

    pub fn do_commit(&self) -> ProverData<Val> {
        ProverData::new(self.inner.commit(&self.prover_data_vec.inner))
    }

    pub fn do_open<Challenge>(
        &self,
        opening_points: &OpeningPoints<Challenge>,
        challenger: Pin<&mut ffi::DuplexChallenger>,
    ) -> OpeningProof {
        OpeningProof::new(self.inner.do_open(
            &self.prover_data_vec.inner,
            &opening_points.inner,
            challenger,
        ))
    }

    fn do_verify<Challenge>(
        &self,
        commitment_vec: &CommitmentVec<Val>,
        domains: &Domains<Val>,
        opening_points: &OpeningPoints<Challenge>,
        opened_values: &OpenedValues<Challenge>,
        proof: &FriProof,
        challenger: Pin<&mut ffi::DuplexChallenger>,
    ) -> bool {
        self.inner.do_verify(
            &commitment_vec.inner,
            &domains.inner,
            &opening_points.inner,
            &opened_values.inner,
            &proof.inner,
            challenger,
        )
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger: CanObserve<FriMmcs::Commitment>
        + CanSample<Challenge>
        + GrindingChallenger<Witness = Val>
        + HasCXXDuplexChallenger,
    <InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>: Clone,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = Hash<
        <<Val as Field>::Packing as PackedField>::Scalar,
        <<Val as Field>::Packing as PackedValue>::Value,
        8,
    >;
    type ProverData = crate::baby_bear_poseidon2::ProverData<Val>;
    type Proof = FriProof;
    type Error = VerificationError<InputMmcs::Error, FriMmcs::Error>;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        let log_n = log2_strict_usize(degree);
        assert!(log_n <= self.log_n);
        TwoAdicMultiplicativeCoset {
            log_n,
            shift: Val::one(),
        }
    }

    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        self.allocate_ldes(evaluations.len());
        let mut ldes = vec![];
        for (domain, mut evals) in evaluations.into_iter() {
            assert_eq!(domain.size(), evals.height());
            let shift = Val::generator() / domain.shift;
            ldes.push(self.coset_lde_batch(&mut evals, shift));
        }
        let mut prover_data = self.do_commit();
        prover_data.ldes = ldes;
        let mut value = [<<Val as Field>::Packing as PackedValue>::Value::default(); 8];
        prover_data.write_commit(unsafe { std::mem::transmute(value.as_mut_slice()) });
        (Self::Commitment::from(value), prover_data)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> impl Matrix<Val> + 'a {
        assert_eq!(domain.shift, Val::generator());
        let lde = &prover_data.ldes[idx];
        assert!(lde.height() >= domain.size());
        lde.split_rows(domain.size()).0.bit_reverse_rows()
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> (p3_commit::OpenedValues<Challenge>, Self::Proof) {
        let mut opening_points = OpeningPoints::new(ffi::new_opening_points(rounds.len()));
        for (round, (_, matrix)) in rounds.iter().enumerate() {
            opening_points.allocate(round, matrix.len(), matrix[0].len());
            for (row, rows) in matrix.iter().enumerate() {
                for (col, challenge) in rows.iter().enumerate() {
                    opening_points.set(round, row, col, challenge);
                }
            }
        }
        let mut opening_proof = self.do_open(&opening_points, challenger.get_inner_pin_mut());
        (
            opening_proof.serialize_to_opened_values(),
            opening_proof.take_fri_proof(),
        )
    }

    fn verify(
        &self,
        // For each round:
        rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<(
                // its domain,
                Self::Domain,
                // for each point:
                Vec<(
                    // the point,
                    Challenge,
                    // values at the point
                    Vec<Challenge>,
                )>,
            )>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let mut commitment_vec = CommitmentVec::new(ffi::new_commitment_vec(rounds.len()));
        let mut domains = Domains::<Val>::new(ffi::new_domains(rounds.len()));
        let mut opening_points = OpeningPoints::new(ffi::new_opening_points(rounds.len()));
        let mut opened_values = OpenedValues::new(ffi::new_opened_values(rounds.len()));
        for (round, (commitment, matrix)) in rounds.iter().enumerate() {
            commitment_vec.set(round, commitment.as_ref());
            domains.allocate(round, matrix.len());
            let claims = &matrix[0].1;
            opening_points.allocate(round, matrix.len(), claims.len());
            opened_values.allocate_outer(round, matrix.len(), claims.len());
            for (row, (domain, claims)) in matrix.iter().enumerate() {
                domains.set(round, row, domain.log_n as u32, domain.shift);
                let values = &claims[0].1;
                opened_values.allocate_inner(round, row, claims.len(), values.len());
                for (col, (challenge, values)) in claims.iter().enumerate() {
                    opening_points.set(round, row, col, challenge);
                    for (idx, value) in values.iter().enumerate() {
                        opened_values.set(round, row, col, idx, value);
                    }
                }
            }
        }
        let succeeded = self.do_verify(
            &commitment_vec,
            &domains,
            &opening_points,
            &opened_values,
            proof,
            challenger.get_inner_pin_mut(),
        );
        assert!(succeeded);
        Ok(())
    }
}
