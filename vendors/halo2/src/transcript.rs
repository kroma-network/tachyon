#[cfg(test)]
mod test {
    use std::io;

    use crate::bn254::Blake2bWrite as TachyonBlake2bWrite;
    use ff::Field;
    use halo2_proofs::transcript::{
        Blake2bWrite, Challenge255, ChallengeScalar, EncodedChallenge, TranscriptWrite,
        TranscriptWriterBuffer,
    };
    use halo2curves::{
        bn256::{Bn256, Fr, G1Affine},
        pairing::Engine,
    };
    use rand_core::OsRng;

    #[derive(Clone, Copy, Debug)]
    struct Theta;
    type ChallengeTheta<F> = ChallengeScalar<F, Theta>;

    fn squeeze_challenge<
        E: Engine,
        C: EncodedChallenge<E::G1Affine>,
        T: TranscriptWrite<E::G1Affine, C>,
    >(
        transcript: &mut T,
    ) -> ChallengeTheta<E::G1Affine> {
        transcript.squeeze_challenge_scalar()
    }

    fn write_point_to_proof<
        E: Engine,
        C: EncodedChallenge<E::G1Affine>,
        T: TranscriptWrite<E::G1Affine, C>,
    >(
        transcript: &mut T,
        point: E::G1Affine,
    ) -> io::Result<()> {
        transcript.write_point(point)
    }

    fn write_scalar_to_proof<
        E: Engine,
        C: EncodedChallenge<E::G1Affine>,
        T: TranscriptWrite<E::G1Affine, C>,
    >(
        transcript: &mut T,
        scalar: E::Scalar,
    ) -> io::Result<()> {
        transcript.write_scalar(scalar)
    }

    #[test]
    fn test_blake2b_write_scalar_to_proof() {
        let fr = Fr::random(OsRng);
        let proof = {
            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
            write_scalar_to_proof::<Bn256, _, _>(&mut transcript, fr.clone()).unwrap();
            transcript.finalize()
        };
        let proof_tachyon = {
            let mut transcript = TachyonBlake2bWrite::init(vec![]);
            write_scalar_to_proof::<Bn256, _, _>(&mut transcript, fr).unwrap();
            transcript.finalize()
        };
        assert_eq!(proof, proof_tachyon);
    }

    #[test]
    fn test_blake2b_write_point_to_proof() {
        let point = G1Affine::random(OsRng);
        let proof = {
            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
            write_point_to_proof::<Bn256, _, _>(&mut transcript, point.clone()).unwrap();
            transcript.finalize()
        };
        let proof_tachyon = {
            let mut transcript = TachyonBlake2bWrite::init(vec![]);
            write_point_to_proof::<Bn256, _, _>(&mut transcript, point).unwrap();
            transcript.finalize()
        };
        assert_eq!(proof, proof_tachyon);
    }

    #[test]
    fn test_blake2b_squeeze_challenge() {
        let point = G1Affine::random(OsRng);
        let theta = {
            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
            write_point_to_proof::<Bn256, _, _>(&mut transcript, point.clone()).unwrap();
            let theta = squeeze_challenge::<Bn256, _, _>(&mut transcript);
            *theta
        };
        let theta_tachyon = {
            let mut transcript = TachyonBlake2bWrite::init(vec![]);
            write_point_to_proof::<Bn256, _, _>(&mut transcript, point).unwrap();
            let theta = squeeze_challenge::<Bn256, _, _>(&mut transcript);
            *theta
        };
        assert_eq!(theta, theta_tachyon);
    }

    #[test]
    fn test_state() {
        let transcript = TachyonBlake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
        assert_eq!(
            transcript.state(),
            vec![
                72, 201, 189, 242, 103, 230, 9, 106, 59, 167, 202, 132, 133, 174, 103, 187, 43,
                248, 148, 254, 114, 243, 110, 60, 241, 54, 29, 95, 58, 245, 79, 165, 209, 130, 230,
                173, 127, 82, 14, 81, 31, 108, 62, 43, 140, 104, 5, 155, 35, 220, 45, 148, 153,
                244, 215, 109, 24, 79, 13, 112, 107, 164, 144, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0
            ]
        );
    }
}
