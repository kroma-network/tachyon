use std::{
    fmt,
    io::{self, Write},
    marker::PhantomData,
};

use ff::{Field, PrimeField};
use halo2_proofs::{
    plonk::{sealed, Column, Fixed},
    poly::commitment::{Blind, CommitmentScheme},
    transcript::{
        Challenge255, EncodedChallenge, Transcript, TranscriptWrite, TranscriptWriterBuffer,
    },
};
use halo2curves::{
    bn256::{G1Affine, G2Affine},
    ff::FromUniformBytes,
    Coordinates, CurveAffine,
};
use num_bigint::BigUint;

use tachyon_rs::math::elliptic_curves::bn::bn254::{
    Fr as FrImpl, G1AffinePoint as G1AffinePointImpl, G1Point2 as G1Point2Impl,
    G1ProjectivePoint as G1ProjectivePointImpl, G2AffinePoint as G2AffinePointImpl,
};

use crate::consts::PCSType;

pub struct G1MSM;
pub struct G1MSMGpu;
pub struct G1AffinePoint(pub G1AffinePointImpl);
pub struct G1ProjectivePoint(pub G1ProjectivePointImpl);
pub struct G1Point2(pub G1Point2Impl);
pub struct G2AffinePoint(pub G2AffinePointImpl);
pub struct Fr(pub FrImpl);
pub struct InstanceSingle {
    pub instance_values: Vec<Evals>,
    pub instance_polys: Vec<Poly>,
}
#[derive(Clone)]
pub struct AdviceSingle {
    pub advice_polys: Vec<Evals>,
    pub advice_blinds: Vec<Blind<halo2curves::bn256::Fr>>,
}

#[cxx::bridge(namespace = "tachyon::halo2_api::bn254")]
pub mod ffi {
    extern "Rust" {
        type G1MSM;
        type G1MSMGpu;
        type G1AffinePoint;
        type G1ProjectivePoint;
        type G1Point2;
        type G2AffinePoint;
        type Fr;
        type InstanceSingle;
        type AdviceSingle;
    }

    unsafe extern "C++" {
        include!("vendors/halo2/include/bn254_msm.h");
        #[cfg(feature = "gpu")]
        include!("vendors/halo2/include/bn254_msm_gpu.h");

        fn create_g1_msm(degree: u8) -> Box<G1MSM>;
        fn destroy_g1_msm(msm: Box<G1MSM>);
        unsafe fn g1_point2_msm(
            msm: *mut G1MSM,
            bases: &[G1Point2],
            scalars: &[Fr],
        ) -> Box<G1ProjectivePoint>;
        #[cfg(feature = "gpu")]
        fn create_g1_msm_gpu(degree: u8) -> Box<G1MSMGpu>;
        #[cfg(feature = "gpu")]
        fn destroy_g1_msm_gpu(msm: Box<G1MSMGpu>);
        #[cfg(feature = "gpu")]
        unsafe fn g1_point2_msm_gpu(
            msm: *mut G1MSMGpu,
            bases: &[G1Point2],
            scalars: &[Fr],
        ) -> Box<G1ProjectivePoint>;
    }

    unsafe extern "C++" {
        include!("vendors/halo2/include/bn254_blake2b_writer.h");

        type Blake2bWriter;

        fn new_blake2b_writer() -> UniquePtr<Blake2bWriter>;
        fn update(self: Pin<&mut Blake2bWriter>, data: &[u8]);
        fn finalize(self: Pin<&mut Blake2bWriter>, result: &mut [u8; 64]);
        fn state(&self) -> Vec<u8>;
    }

    unsafe extern "C++" {
        include!("vendors/halo2/include/bn254_poseidon_writer.h");

        type PoseidonWriter;

        fn new_poseidon_writer() -> UniquePtr<PoseidonWriter>;
        fn update(self: Pin<&mut PoseidonWriter>, data: &[u8]);
        fn squeeze(self: Pin<&mut PoseidonWriter>) -> Box<Fr>;
        fn state(&self) -> Vec<u8>;
    }

    unsafe extern "C++" {
        include!("vendors/halo2/include/bn254_sha256_writer.h");

        type Sha256Writer;

        fn new_sha256_writer() -> UniquePtr<Sha256Writer>;
        fn update(self: Pin<&mut Sha256Writer>, data: &[u8]);
        fn finalize(self: Pin<&mut Sha256Writer>, result: &mut [u8; 32]);
        fn state(&self) -> Vec<u8>;
    }

    unsafe extern "C++" {
        include!("vendors/halo2/include/bn254_proving_key.h");

        type ProvingKey;

        fn new_proving_key(data: &[u8]) -> UniquePtr<ProvingKey>;
        fn advice_column_phases(&self) -> Vec<u8>;
        fn blinding_factors(&self) -> u32;
        fn challenge_phases(&self) -> Vec<u8>;
        fn constants(&self) -> Vec<usize>;
        fn num_advice_columns(&self) -> usize;
        fn num_challenges(&self) -> usize;
        fn num_instance_columns(&self) -> usize;
        fn phases(&self) -> Vec<u8>;
        fn transcript_repr(self: Pin<&mut ProvingKey>, prover: &Prover) -> Box<Fr>;
    }

    unsafe extern "C++" {
        include!("vendors/halo2/include/bn254_evals.h");

        type Evals;

        fn zero_evals() -> UniquePtr<Evals>;
        fn len(&self) -> usize;
        fn set_value(self: Pin<&mut Evals>, idx: usize, value: &Fr);
        fn clone(&self) -> UniquePtr<Evals>;
    }

    unsafe extern "C++" {
        include!("vendors/halo2/include/bn254_rational_evals.h");

        type RationalEvals;

        fn len(&self) -> usize;
        fn create_view(
            self: Pin<&mut RationalEvals>,
            start: usize,
            len: usize,
        ) -> UniquePtr<RationalEvalsView>;
        fn clone(&self) -> UniquePtr<RationalEvals>;
    }

    unsafe extern "C++" {
        include!("vendors/halo2/include/bn254_rational_evals_view.h");

        type RationalEvalsView;

        fn set_zero(self: Pin<&mut RationalEvalsView>, idx: usize);
        fn set_trivial(self: Pin<&mut RationalEvalsView>, idx: usize, numerator: &Fr);
        fn set_rational(
            self: Pin<&mut RationalEvalsView>,
            idx: usize,
            numerator: &Fr,
            denominator: &Fr,
        );
        fn evaluate(&self, idx: usize, value: &mut Fr);
    }

    unsafe extern "C++" {
        include!("vendors/halo2/include/bn254_poly.h");

        type Poly;
    }

    unsafe extern "C++" {
        include!("vendors/halo2/include/bn254_prover.h");

        type Prover;

        fn new_prover(pcs_type: u8, transcript_type: u8, k: u32, s: &Fr) -> UniquePtr<Prover>;
        fn new_prover_from_params(
            pcs_type: u8,
            transcript_type: u8,
            k: u32,
            params: &[u8],
        ) -> UniquePtr<Prover>;
        fn k(&self) -> u32;
        fn n(&self) -> u64;
        fn s_g2(&self) -> &G2AffinePoint;
        fn commit(&self, poly: &Poly) -> Box<G1ProjectivePoint>;
        fn commit_lagrange(&self, evals: &Evals) -> Box<G1ProjectivePoint>;
        fn batch_start(&self, size: usize);
        fn batch_commit(&self, poly: &Poly, idx: usize);
        fn batch_commit_lagrange(&self, evals: &Evals, idx: usize);
        fn batch_end(&self, points: &mut [G1AffinePoint]);
        fn empty_evals(&self) -> UniquePtr<Evals>;
        fn empty_rational_evals(&self) -> UniquePtr<RationalEvals>;
        fn ifft(&self, evals: &Evals) -> UniquePtr<Poly>;
        fn batch_evaluate(
            &self,
            rational_evals: &[UniquePtr<RationalEvals>],
            evals: &mut [UniquePtr<Evals>],
        );
        fn set_rng(self: Pin<&mut Prover>, state: &[u8]);
        fn set_transcript(self: Pin<&mut Prover>, state: &[u8]);
        fn set_extended_domain(self: Pin<&mut Prover>, pk: &ProvingKey);
        fn create_proof(
            self: Pin<&mut Prover>,
            key: Pin<&mut ProvingKey>,
            instance_singles: &mut [InstanceSingle],
            advice_singles: &mut [AdviceSingle],
            challenges: &[Fr],
        );
        fn get_proof(self: &Prover) -> Vec<u8>;
    }
}

pub trait TranscriptWriteState<C: CurveAffine, E: EncodedChallenge<C>>:
    TranscriptWrite<C, E>
{
    fn state(&self) -> Vec<u8>;
}

pub struct Blake2bWrite<W: Write, C: CurveAffine, E: EncodedChallenge<C>> {
    state: cxx::UniquePtr<ffi::Blake2bWriter>,
    writer: W,
    _marker: PhantomData<(W, C, E)>,
}

impl<W: Write, C: CurveAffine> Transcript<C, Challenge255<C>>
    for Blake2bWrite<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn squeeze_challenge(&mut self) -> Challenge255<C> {
        // Prefix to a prover's message soliciting a challenge
        const BLAKE2B_PREFIX_CHALLENGE: u8 = 0;
        self.state.pin_mut().update(&[BLAKE2B_PREFIX_CHALLENGE]);
        let mut result: [u8; 64] = [0; 64];
        self.state.pin_mut().finalize(&mut result);
        Challenge255::<C>::new(&result)
    }

    fn common_point(&mut self, point: C) -> io::Result<()> {
        // Prefix to a prover's message containing a curve point
        const BLAKE2B_PREFIX_POINT: u8 = 1;
        self.state.pin_mut().update(&[BLAKE2B_PREFIX_POINT]);
        let coords: Coordinates<C> = Option::from(point.coordinates()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "cannot write points at infinity to the transcript",
            )
        })?;
        self.state.pin_mut().update(coords.x().to_repr().as_ref());
        self.state.pin_mut().update(coords.y().to_repr().as_ref());

        Ok(())
    }

    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        // Prefix to a prover's message containing a scalar
        const BLAKE2B_PREFIX_SCALAR: u8 = 2;
        self.state.pin_mut().update(&[BLAKE2B_PREFIX_SCALAR]);
        self.state.pin_mut().update(scalar.to_repr().as_ref());
        Ok(())
    }
}

impl<W: Write, C: CurveAffine> TranscriptWrite<C, Challenge255<C>>
    for Blake2bWrite<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn write_point(&mut self, point: C) -> io::Result<()> {
        self.common_point(point)?;
        let compressed = point.to_bytes();
        self.writer.write_all(compressed.as_ref())
    }

    fn write_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        self.common_scalar(scalar)?;
        let data = scalar.to_repr();
        self.writer.write_all(data.as_ref())
    }
}

impl<W: Write, C: CurveAffine> TranscriptWriteState<C, Challenge255<C>>
    for Blake2bWrite<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn state(&self) -> Vec<u8> {
        self.state.state()
    }
}

impl<W: Write, C: CurveAffine> TranscriptWriterBuffer<W, C, Challenge255<C>>
    for Blake2bWrite<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    /// Initialize a transcript given an output buffer.
    fn init(writer: W) -> Self {
        Blake2bWrite {
            state: ffi::new_blake2b_writer(),
            writer: writer,
            _marker: PhantomData,
        }
    }

    fn finalize(self) -> W {
        // TODO: handle outstanding scalars?
        // See https://github.com/zcash/halo2/issues/138.
        self.writer
    }
}

pub struct PoseidonWrite<W: Write, C: CurveAffine, E: EncodedChallenge<C>> {
    state: cxx::UniquePtr<ffi::PoseidonWriter>,
    writer: W,
    _marker: PhantomData<(W, C, E)>,
}

fn field_to_bn<F: PrimeField>(f: &F) -> BigUint {
    BigUint::from_bytes_le(f.to_repr().as_ref())
}

/// Input a big integer `bn`, compute a field element `f`
/// such that `f == bn % F::MODULUS`.
/// Require:
/// - `bn` is less than 512 bits.
/// Return:
/// - `bn mod F::MODULUS` when `bn > F::MODULUS`
pub fn bn_to_field<F: PrimeField>(bn: &BigUint) -> F
where
    F: FromUniformBytes<64>,
{
    let mut buf = bn.to_bytes_le();
    buf.resize(64, 0u8);

    let mut buf_array = [0u8; 64];
    buf_array.copy_from_slice(buf.as_ref());
    F::from_uniform_bytes(&buf_array)
}

/// Input a base field element `b`, output a scalar field
/// element `s` s.t. `s == b % ScalarField::MODULUS`
fn base_to_scalar<C: CurveAffine>(base: &C::Base) -> C::Scalar
where
    C::Scalar: FromUniformBytes<64>,
{
    let bn = field_to_bn(base);
    // bn_to_field will perform a mod reduction
    bn_to_field(&bn)
}

impl<W: Write, C: CurveAffine> Transcript<C, Challenge255<C>>
    for PoseidonWrite<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn squeeze_challenge(&mut self) -> Challenge255<C> {
        let scalar = *unsafe {
            std::mem::transmute::<_, Box<halo2curves::bn256::Fr>>(self.state.pin_mut().squeeze())
        };
        let mut scalar_bytes = scalar.to_repr().as_ref().to_vec();
        scalar_bytes.resize(64, 0u8);
        Challenge255::<C>::new(&scalar_bytes.try_into().unwrap())
    }

    fn common_point(&mut self, point: C) -> io::Result<()> {
        let coords: Coordinates<C> = Option::from(point.coordinates()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "cannot write points at infinity to the transcript",
            )
        })?;
        let x = coords.x();
        let y = coords.y();
        let slice = &[base_to_scalar::<C>(x), base_to_scalar::<C>(y)];
        let bytes = std::mem::size_of::<C::Scalar>() * 2;
        unsafe {
            self.state.pin_mut().update(std::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                bytes,
            ));
        }

        Ok(())
    }

    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        let slice = &[scalar];
        let bytes = std::mem::size_of::<C::Scalar>();
        unsafe {
            self.state.pin_mut().update(std::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                bytes,
            ));
        }

        Ok(())
    }
}

impl<W: Write, C: CurveAffine> TranscriptWrite<C, Challenge255<C>>
    for PoseidonWrite<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn write_point(&mut self, point: C) -> io::Result<()> {
        self.common_point(point)?;
        let compressed = point.to_bytes();
        self.writer.write_all(compressed.as_ref())
    }

    fn write_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        self.common_scalar(scalar)?;
        let data = scalar.to_repr();
        self.writer.write_all(data.as_ref())
    }
}

impl<W: Write, C: CurveAffine, E: EncodedChallenge<C>> PoseidonWrite<W, C, E> {
    /// Initialize a transcript given an output buffer.
    pub fn init(writer: W) -> Self {
        PoseidonWrite {
            state: ffi::new_poseidon_writer(),
            writer,
            _marker: PhantomData,
        }
    }

    /// Conclude the interaction and return the output buffer (writer).
    pub fn finalize(self) -> W {
        // TODO: handle outstanding scalars?
        // See https://github.com/zcash/halo2/issues/138.
        self.writer
    }
}

impl<W: Write, C: CurveAffine> TranscriptWriteState<C, Challenge255<C>>
    for PoseidonWrite<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn state(&self) -> Vec<u8> {
        self.state.state()
    }
}

pub struct Sha256Write<W: Write, C: CurveAffine, E: EncodedChallenge<C>> {
    state: cxx::UniquePtr<ffi::Sha256Writer>,
    writer: W,
    _marker: PhantomData<(W, C, E)>,
}

impl<W: Write, C: CurveAffine> Transcript<C, Challenge255<C>> for Sha256Write<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn squeeze_challenge(&mut self) -> Challenge255<C> {
        const SHA256_PREFIX_CHALLENGE: u8 = 0;
        self.state.pin_mut().update(&[SHA256_PREFIX_CHALLENGE]);
        let mut result: [u8; 32] = [0; 32];
        self.state.pin_mut().finalize(&mut result);

        self.state = ffi::new_sha256_writer();
        self.state.pin_mut().update(result.as_slice());

        let mut bytes = result.to_vec();
        bytes.resize(64, 0u8);
        Challenge255::<C>::new(&bytes.try_into().unwrap())
    }

    fn common_point(&mut self, point: C) -> io::Result<()> {
        const SHA256_PREFIX_POINT: u8 = 1;
        self.state.pin_mut().update(&[0u8; 31]);
        self.state.pin_mut().update(&[SHA256_PREFIX_POINT]);
        let coords: Coordinates<C> = Option::from(point.coordinates()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "cannot write points at infinity to the transcript",
            )
        })?;

        for base in &[coords.x(), coords.y()] {
            let mut buf = base.to_repr().as_ref().to_vec();
            buf.resize(32, 0u8);
            buf.reverse();
            self.state.pin_mut().update(buf.as_slice());
        }

        Ok(())
    }

    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        const SHA256_PREFIX_SCALAR: u8 = 2;
        self.state.pin_mut().update(&[0u8; 31]);
        self.state.pin_mut().update(&[SHA256_PREFIX_SCALAR]);

        {
            let mut buf = scalar.to_repr().as_ref().to_vec();
            buf.resize(32, 0u8);
            buf.reverse();
            self.state.pin_mut().update(buf.as_slice());
        }

        Ok(())
    }
}

impl<W: Write, C: CurveAffine> TranscriptWrite<C, Challenge255<C>>
    for Sha256Write<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn write_point(&mut self, point: C) -> io::Result<()> {
        self.common_point(point)?;

        let coords = point.coordinates();
        let x = coords
            .map(|v| *v.x())
            .unwrap_or(<C as CurveAffine>::Base::ZERO);
        let y = coords
            .map(|v| *v.y())
            .unwrap_or(<C as CurveAffine>::Base::ZERO);

        for base in &[&x, &y] {
            self.writer.write_all(base.to_repr().as_ref())?;
        }

        Ok(())
    }

    fn write_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        self.common_scalar(scalar)?;
        let data = scalar.to_repr();

        self.writer.write_all(data.as_ref())
    }
}

impl<W: Write, C: CurveAffine> TranscriptWriteState<C, Challenge255<C>>
    for Sha256Write<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn state(&self) -> Vec<u8> {
        self.state.state()
    }
}

impl<W: Write, C: CurveAffine, E: EncodedChallenge<C>> Sha256Write<W, C, E> {
    /// Initialize a transcript given an output buffer.
    pub fn init(writer: W) -> Self {
        Sha256Write {
            state: ffi::new_sha256_writer(),
            writer,
            _marker: PhantomData,
        }
    }

    /// Conclude the interaction and return the output buffer (writer).
    pub fn finalize(self) -> W {
        // TODO: handle outstanding scalars?
        // See https://github.com/zcash/halo2/issues/138.
        self.writer
    }
}

pub struct ProvingKey<C: CurveAffine> {
    inner: cxx::UniquePtr<ffi::ProvingKey>,
    _marker: PhantomData<C>,
}

impl<C: CurveAffine> ProvingKey<C> {
    pub fn from(data: &[u8]) -> ProvingKey<C> {
        ProvingKey {
            inner: ffi::new_proving_key(data),
            _marker: PhantomData,
        }
    }

    // NOTE(chokobole): We name this as plural since it contains multi phases.
    // pk.vk.cs.advice_column_phase
    pub fn advice_column_phases(&self) -> Vec<sealed::Phase> {
        unsafe {
            let phases: Vec<sealed::Phase> = std::mem::transmute(self.inner.advice_column_phases());
            phases
        }
    }

    // pk.vk.cs.blinding_factors()
    pub fn blinding_factors(&self) -> u32 {
        self.inner.blinding_factors()
    }

    // NOTE(chokobole): We name this as plural since it contains multi phases.
    // pk.vk.cs.challenge_phase
    pub fn challenge_phases(&self) -> Vec<sealed::Phase> {
        unsafe {
            let phases: Vec<sealed::Phase> = std::mem::transmute(self.inner.challenge_phases());
            phases
        }
    }

    // pk.vk.cs.constants
    pub fn constants(&self) -> Vec<Column<Fixed>> {
        let constants = self
            .inner
            .constants()
            .iter()
            .map(|index| Column {
                index: *index,
                column_type: Fixed,
            })
            .collect::<Vec<_>>();
        constants
    }

    // pk.vk.cs.num_advice_columns
    pub fn num_advice_columns(&self) -> usize {
        self.inner.num_advice_columns()
    }

    // pk.vk.cs.num_challenges
    pub fn num_challenges(&self) -> usize {
        self.inner.num_challenges()
    }

    // pk.vk.cs.num_instance_columns
    pub fn num_instance_columns(&self) -> usize {
        self.inner.num_instance_columns()
    }

    // pk.vk.cs.phases()
    pub fn phases(&self) -> Vec<sealed::Phase> {
        unsafe {
            let phases: Vec<sealed::Phase> = std::mem::transmute(self.inner.phases());
            phases
        }
    }

    // pk.vk.transcript_repr
    pub fn transcript_repr<Scheme: CommitmentScheme, P: TachyonProver<Scheme>>(
        &mut self,
        prover: &P,
    ) -> C::Scalar {
        *unsafe {
            std::mem::transmute::<_, Box<C::Scalar>>(
                self.inner.pin_mut().transcript_repr(prover.inner()),
            )
        }
    }
}

pub struct Evals {
    inner: cxx::UniquePtr<ffi::Evals>,
}

impl Evals {
    pub fn zero() -> Evals {
        Self::new(ffi::zero_evals())
    }

    pub fn new(inner: cxx::UniquePtr<ffi::Evals>) -> Evals {
        Evals { inner }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn set_value(&mut self, idx: usize, fr: &halo2curves::bn256::Fr) {
        let cpp_fr = unsafe { std::mem::transmute::<_, &Fr>(fr) };
        self.inner.pin_mut().set_value(idx, cpp_fr)
    }
}

impl Clone for Evals {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[derive(Debug)]
pub struct RationalEvals {
    inner: cxx::UniquePtr<ffi::RationalEvals>,
}

impl RationalEvals {
    pub fn new(inner: cxx::UniquePtr<ffi::RationalEvals>) -> RationalEvals {
        RationalEvals { inner }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn create_view(&mut self, start: usize, len: usize) -> RationalEvalsView {
        RationalEvalsView::new(self.inner.pin_mut().create_view(start, len))
    }
}

unsafe impl Send for ffi::RationalEvals {}
unsafe impl Sync for ffi::RationalEvals {}

impl fmt::Debug for ffi::RationalEvals {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RationalEvals").finish()
    }
}

impl Clone for RationalEvals {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

pub struct RationalEvalsView {
    inner: cxx::UniquePtr<ffi::RationalEvalsView>,
}

impl RationalEvalsView {
    pub fn new(inner: cxx::UniquePtr<ffi::RationalEvalsView>) -> RationalEvalsView {
        RationalEvalsView { inner }
    }

    pub fn set_zero(&mut self, idx: usize) {
        self.inner.pin_mut().set_zero(idx)
    }

    pub fn set_trivial(&mut self, idx: usize, numerator: &halo2curves::bn256::Fr) {
        let cpp_numerator = unsafe { std::mem::transmute::<_, &Fr>(numerator) };
        self.inner.pin_mut().set_trivial(idx, cpp_numerator)
    }

    pub fn set_rational(
        &mut self,
        idx: usize,
        numerator: &halo2curves::bn256::Fr,
        denominator: &halo2curves::bn256::Fr,
    ) {
        let cpp_numerator = unsafe { std::mem::transmute::<_, &Fr>(numerator) };
        let cpp_denominator = unsafe { std::mem::transmute::<_, &Fr>(denominator) };
        self.inner
            .pin_mut()
            .set_rational(idx, cpp_numerator, cpp_denominator)
    }

    pub fn evaluate(&self, idx: usize, value: &mut halo2curves::bn256::Fr) {
        self.inner
            .evaluate(idx, unsafe { std::mem::transmute::<_, &mut Fr>(value) })
    }
}

unsafe impl Send for ffi::RationalEvalsView {}
unsafe impl Sync for ffi::RationalEvalsView {}

pub struct Poly {
    inner: cxx::UniquePtr<ffi::Poly>,
}

impl Poly {
    pub fn new(inner: cxx::UniquePtr<ffi::Poly>) -> Poly {
        Poly { inner }
    }
}

pub trait TachyonProver<Scheme: CommitmentScheme> {
    const QUERY_INSTANCE: bool;

    fn inner(&self) -> &ffi::Prover;

    fn k(&self) -> u32;

    fn n(&self) -> u64;

    fn s_g2(&self) -> &G2Affine;

    fn commit(&self, poly: &Poly) -> <Scheme::Curve as CurveAffine>::CurveExt;

    fn commit_lagrange(&self, evals: &Evals) -> <Scheme::Curve as CurveAffine>::CurveExt;

    fn batch_start(&self, size: usize);

    fn batch_commit(&self, poly: &Poly, idx: usize);

    fn batch_commit_lagrange(&self, evals: &Evals, idx: usize);

    fn batch_end(&self, points: &mut [G1Affine]);

    fn empty_evals(&self) -> Evals;

    fn empty_rational_evals(&self) -> RationalEvals;

    fn batch_evaluate(&self, rational_evals: &[RationalEvals], evals: &mut [Evals]);

    fn ifft(&self, evals: &Evals) -> Poly;

    fn set_rng(&mut self, state: &[u8]);

    fn set_transcript(&mut self, state: &[u8]);

    fn set_extended_domain(&mut self, pk: &ProvingKey<Scheme::Curve>);

    fn create_proof(
        &mut self,
        key: &mut ProvingKey<Scheme::Curve>,
        instance_singles: &mut [InstanceSingle],
        advice_singles: &mut [AdviceSingle],
        challenges: &[Fr],
    );

    fn get_proof(&self) -> Vec<u8>;

    fn transcript_repr(&self, pk: &mut ProvingKey<Scheme::Curve>) -> Scheme::Scalar;
}

pub struct GWCProver<Scheme: CommitmentScheme> {
    inner: cxx::UniquePtr<ffi::Prover>,
    _marker: PhantomData<Scheme>,
}

impl<Scheme: CommitmentScheme> GWCProver<Scheme> {
    pub fn new(transcript_type: u8, k: u32, s: &halo2curves::bn256::Fr) -> GWCProver<Scheme> {
        let cpp_s = unsafe { std::mem::transmute::<_, &Fr>(s) };
        GWCProver {
            inner: ffi::new_prover(PCSType::GWC as u8, transcript_type, k, cpp_s),
            _marker: PhantomData,
        }
    }

    pub fn from_params(transcript_type: u8, k: u32, params: &[u8]) -> GWCProver<Scheme> {
        GWCProver {
            inner: ffi::new_prover_from_params(PCSType::GWC as u8, transcript_type, k, params),
            _marker: PhantomData,
        }
    }
}

impl<Scheme: CommitmentScheme> TachyonProver<Scheme> for GWCProver<Scheme> {
    const QUERY_INSTANCE: bool = true;

    fn inner(&self) -> &ffi::Prover {
        &self.inner
    }

    fn k(&self) -> u32 {
        self.inner.k()
    }

    fn n(&self) -> u64 {
        self.inner.n()
    }

    fn s_g2(&self) -> &G2Affine {
        unsafe { std::mem::transmute::<_, &G2Affine>(self.inner.s_g2()) }
    }

    fn commit(&self, poly: &Poly) -> <Scheme::Curve as CurveAffine>::CurveExt {
        *unsafe {
            std::mem::transmute::<_, Box<<Scheme::Curve as CurveAffine>::CurveExt>>(
                self.inner.commit(&poly.inner),
            )
        }
    }

    fn commit_lagrange(&self, evals: &Evals) -> <Scheme::Curve as CurveAffine>::CurveExt {
        *unsafe {
            std::mem::transmute::<_, Box<<Scheme::Curve as CurveAffine>::CurveExt>>(
                self.inner.commit_lagrange(&evals.inner),
            )
        }
    }

    fn batch_start(&self, size: usize) {
        self.inner.batch_start(size)
    }

    fn batch_commit(&self, poly: &Poly, idx: usize) {
        self.inner.batch_commit(&poly.inner, idx)
    }

    fn batch_commit_lagrange(&self, evals: &Evals, idx: usize) {
        self.inner.batch_commit_lagrange(&evals.inner, idx)
    }

    fn batch_end(&self, points: &mut [G1Affine]) {
        self.inner
            .batch_end(unsafe { std::mem::transmute::<_, &mut [G1AffinePoint]>(points) })
    }

    fn empty_evals(&self) -> Evals {
        Evals::new(self.inner.empty_evals())
    }

    fn empty_rational_evals(&self) -> RationalEvals {
        RationalEvals::new(self.inner.empty_rational_evals())
    }

    fn batch_evaluate(&self, rational_evals: &[RationalEvals], evals: &mut [Evals]) {
        unsafe {
            let rational_evals: &[cxx::UniquePtr<ffi::RationalEvals>] =
                std::mem::transmute(rational_evals);
            let evals: &mut [cxx::UniquePtr<ffi::Evals>] = std::mem::transmute(evals);
            self.inner.batch_evaluate(rational_evals, evals)
        }
    }

    fn ifft(&self, evals: &Evals) -> Poly {
        Poly::new(self.inner.ifft(&evals.inner))
    }

    fn set_rng(&mut self, state: &[u8]) {
        self.inner.pin_mut().set_rng(state)
    }

    fn set_transcript(&mut self, state: &[u8]) {
        self.inner.pin_mut().set_transcript(state)
    }

    fn set_extended_domain(&mut self, pk: &ProvingKey<Scheme::Curve>) {
        self.inner.pin_mut().set_extended_domain(&pk.inner)
    }

    fn create_proof(
        &mut self,
        key: &mut ProvingKey<Scheme::Curve>,
        instance_singles: &mut [InstanceSingle],
        advice_singles: &mut [AdviceSingle],
        challenges: &[Fr],
    ) {
        self.inner.pin_mut().create_proof(
            key.inner.pin_mut(),
            instance_singles,
            advice_singles,
            challenges,
        )
    }

    fn get_proof(&self) -> Vec<u8> {
        self.inner.get_proof()
    }

    fn transcript_repr(
        &self,
        pk: &mut ProvingKey<<Scheme as CommitmentScheme>::Curve>,
    ) -> Scheme::Scalar {
        pk.transcript_repr(self)
    }
}

pub struct SHPlonkProver<Scheme: CommitmentScheme> {
    inner: cxx::UniquePtr<ffi::Prover>,
    _marker: PhantomData<Scheme>,
}

impl<Scheme: CommitmentScheme> SHPlonkProver<Scheme> {
    pub fn new(transcript_type: u8, k: u32, s: &halo2curves::bn256::Fr) -> SHPlonkProver<Scheme> {
        let cpp_s = unsafe { std::mem::transmute::<_, &Fr>(s) };
        SHPlonkProver {
            inner: ffi::new_prover(PCSType::SHPlonk as u8, transcript_type, k, cpp_s),
            _marker: PhantomData,
        }
    }

    pub fn from_params(transcript_type: u8, k: u32, params: &[u8]) -> SHPlonkProver<Scheme> {
        SHPlonkProver {
            inner: ffi::new_prover_from_params(PCSType::SHPlonk as u8, transcript_type, k, params),
            _marker: PhantomData,
        }
    }
}

impl<Scheme: CommitmentScheme> TachyonProver<Scheme> for SHPlonkProver<Scheme> {
    const QUERY_INSTANCE: bool = false;

    fn inner(&self) -> &ffi::Prover {
        &self.inner
    }

    fn k(&self) -> u32 {
        self.inner.k()
    }

    fn n(&self) -> u64 {
        self.inner.n()
    }

    fn s_g2(&self) -> &G2Affine {
        unsafe { std::mem::transmute::<_, &G2Affine>(self.inner.s_g2()) }
    }

    fn commit(&self, poly: &Poly) -> <Scheme::Curve as CurveAffine>::CurveExt {
        *unsafe {
            std::mem::transmute::<_, Box<<Scheme::Curve as CurveAffine>::CurveExt>>(
                self.inner.commit(&poly.inner),
            )
        }
    }

    fn commit_lagrange(&self, evals: &Evals) -> <Scheme::Curve as CurveAffine>::CurveExt {
        *unsafe {
            std::mem::transmute::<_, Box<<Scheme::Curve as CurveAffine>::CurveExt>>(
                self.inner.commit_lagrange(&evals.inner),
            )
        }
    }

    fn batch_start(&self, size: usize) {
        self.inner.batch_start(size)
    }

    fn batch_commit(&self, poly: &Poly, idx: usize) {
        self.inner.batch_commit(&poly.inner, idx)
    }

    fn batch_commit_lagrange(&self, evals: &Evals, idx: usize) {
        self.inner.batch_commit_lagrange(&evals.inner, idx)
    }

    fn batch_end(&self, points: &mut [G1Affine]) {
        self.inner
            .batch_end(unsafe { std::mem::transmute::<_, &mut [G1AffinePoint]>(points) })
    }

    fn empty_evals(&self) -> Evals {
        Evals::new(self.inner.empty_evals())
    }

    fn empty_rational_evals(&self) -> RationalEvals {
        RationalEvals::new(self.inner.empty_rational_evals())
    }

    fn batch_evaluate(&self, rational_evals: &[RationalEvals], evals: &mut [Evals]) {
        unsafe {
            let rational_evals: &[cxx::UniquePtr<ffi::RationalEvals>] =
                std::mem::transmute(rational_evals);
            let evals: &mut [cxx::UniquePtr<ffi::Evals>] = std::mem::transmute(evals);
            self.inner.batch_evaluate(rational_evals, evals)
        }
    }

    fn ifft(&self, evals: &Evals) -> Poly {
        Poly::new(self.inner.ifft(&evals.inner))
    }

    fn set_rng(&mut self, state: &[u8]) {
        self.inner.pin_mut().set_rng(state)
    }

    fn set_transcript(&mut self, state: &[u8]) {
        self.inner.pin_mut().set_transcript(state)
    }

    fn set_extended_domain(&mut self, pk: &ProvingKey<Scheme::Curve>) {
        self.inner.pin_mut().set_extended_domain(&pk.inner)
    }

    fn create_proof(
        &mut self,
        key: &mut ProvingKey<Scheme::Curve>,
        instance_singles: &mut [InstanceSingle],
        advice_singles: &mut [AdviceSingle],
        challenges: &[Fr],
    ) {
        self.inner.pin_mut().create_proof(
            key.inner.pin_mut(),
            instance_singles,
            advice_singles,
            challenges,
        )
    }

    fn get_proof(&self) -> Vec<u8> {
        self.inner.get_proof()
    }

    fn transcript_repr(
        &self,
        pk: &mut ProvingKey<<Scheme as CommitmentScheme>::Curve>,
    ) -> Scheme::Scalar {
        pk.transcript_repr(self)
    }
}
