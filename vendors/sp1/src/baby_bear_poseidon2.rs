use std::{fmt::Debug, marker::PhantomData};

use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PackedField, PackedValue, TwoAdicField};
use p3_fri::{FriConfig, TwoAdicFriPcsProof, VerificationError};
use p3_matrix::{
    bitrev::BitReversableMatrix,
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};
use p3_symmetric::{CryptographicPermutation, Hash};
use p3_util::log2_strict_usize;
use tachyon_rs::math::finite_fields::baby_bear::BabyBear as TachyonBabyBearImpl;

pub struct TachyonBabyBear(pub TachyonBabyBearImpl);

#[cxx::bridge(namespace = "tachyon::sp1_api::baby_bear_poseidon2")]
pub mod ffi {
    extern "Rust" {
        type TachyonBabyBear;
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
    }
}

pub struct DuplexChallenger<F, P, const WIDTH: usize, const RATE: usize> {
    inner: cxx::UniquePtr<ffi::DuplexChallenger>,
    _marker: PhantomData<(F, P)>,
}

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

impl<F, P, const WIDTH: usize, const RATE: usize> CanObserve<F>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy + Debug,
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
    F: Copy + Debug,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, values: [F; N]) {
        for value in values {
            self.observe(value);
        }
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> CanObserve<Vec<Vec<F>>>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy + Debug,
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

impl<F, P, const WIDTH: usize, const RATE: usize> CanSample<F>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Field,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn sample(&mut self) -> F {
        *unsafe { std::mem::transmute::<_, Box<F>>(self.inner.pin_mut().sample()) }
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
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        CanObserve<FriMmcs::Commitment> + CanSample<Challenge> + GrindingChallenger<Witness = Val>,
    <InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>: Clone,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = Hash<
        <<Val as Field>::Packing as PackedField>::Scalar,
        <<Val as Field>::Packing as PackedValue>::Value,
        8,
    >;
    type ProverData = crate::baby_bear_poseidon2::ProverData<Val>;
    type Proof = TwoAdicFriPcsProof<Val, Challenge, InputMmcs, FriMmcs>;
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
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        todo!()
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
        todo!()
    }
}
