// This is taken and modified from https://github.com/kroma-network/halo2-snark-aggregator/blob/2637b51/halo2-snark-aggregator-api/src/transcript/sha.rs.

use std::{
    io::{self, Write},
    marker::PhantomData,
};

use digest::Digest;
use ff::{Field, FromUniformBytes, PrimeField};
use halo2_proofs::transcript::{Challenge255, EncodedChallenge, Transcript, TranscriptWrite};
use halo2curves::{Coordinates, CurveAffine};

#[derive(Debug, Clone)]
pub struct ShaWrite<W: Write, C: CurveAffine, E: EncodedChallenge<C>, D: Digest> {
    state: D,
    writer: W,
    _marker: PhantomData<(C, E)>,
}

impl<W: Write, C: CurveAffine, E: EncodedChallenge<C>, D: Digest> ShaWrite<W, C, E, D> {
    /// Initialize a transcript given an output buffer.
    pub fn init(writer: W) -> Self {
        ShaWrite {
            state: D::new(),
            writer,
            _marker: PhantomData,
        }
    }

    /// Conclude the interaction and return the output buffer (writer).
    pub fn finalize(self) -> W {
        // TODO: handle outstanding scalars? see issue #138
        self.writer
    }
}

impl<W: Write, C: CurveAffine, D: Digest + Clone> TranscriptWrite<C, Challenge255<C>>
    for ShaWrite<W, C, Challenge255<C>, D>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn write_point(&mut self, point: C) -> io::Result<()> {
        self.common_point(point)?;
        // let compressed = point.to_bytes();

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

impl<W: Write, C: CurveAffine, D: Digest + Clone> Transcript<C, Challenge255<C>>
    for ShaWrite<W, C, Challenge255<C>, D>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn squeeze_challenge(&mut self) -> Challenge255<C> {
        const SHA_PREFIX_CHALLENGE: u8 = 0;
        self.state.update(&[SHA_PREFIX_CHALLENGE]);
        let hasher = self.state.clone();
        let result: [u8; 32] = hasher.finalize().as_slice().try_into().unwrap();

        self.state = D::new();
        self.state.update(result);

        let mut bytes = result.to_vec();
        bytes.resize(64, 0u8);
        Challenge255::<C>::new(&bytes.try_into().unwrap())
    }

    fn common_point(&mut self, point: C) -> io::Result<()> {
        const SHA_PREFIX_POINT: u8 = 1;
        self.state.update(&[0u8; 31]);
        self.state.update(&[SHA_PREFIX_POINT]);
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
            self.state.update(buf);
        }

        Ok(())
    }

    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        const SHA_PREFIX_SCALAR: u8 = 2;
        self.state.update(&[0u8; 31]);
        self.state.update(&[SHA_PREFIX_SCALAR]);

        {
            let mut buf = scalar.to_repr().as_ref().to_vec();
            buf.resize(32, 0u8);
            buf.reverse();
            self.state.update(buf);
        }

        Ok(())
    }
}
