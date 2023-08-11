use ark_ec::Group;
use ark_ff::Field;
// We'll use the BLS12-381 G1 curve for this example.
// This group has a prime order `r`, and is associated with a prime field `Fr`.
use ark_std::{ ops::Mul, UniformRand, Zero };
use ark_test_curves::bls12_381::{ Fr as ScalarField, G1Projective as G };
use cxx::let_cxx_string;

#[cxx::bridge(namespace = "benchmark::msm::arkworks")]
mod ffi {
    extern "Rust" {
        fn group_operation(s: &String);
    }

    unsafe extern "C++" {
        include!("benchmark/msm/arkworks/include/arkworks_benchmark.h");
        fn arkworks_benchmark(s: &CxxString);
    }
}

fn group_operation(s: &String) {
    let mut rng = ark_std::test_rng();
    // Let's sample uniformly random group elements:
    let a = G::rand(&mut rng);
    let b = G::rand(&mut rng);

    // We can add elements, ...
    let c = a + b;
    // ... subtract them, ...
    let d = a - b;
    // ... and double them.
    assert_eq!(c + d, a.double());
    // We can also negate elements, ...
    let e = -a;
    // ... and check that negation satisfies the basic group law
    assert_eq!(e + a, G::zero());

    // We can also multiply group elements by elements of the corresponding scalar field
    // (an act known as *scalar multiplication*)
    let scalar = ScalarField::rand(&mut rng);
    let e = c.mul(scalar);
    let f = e.mul(scalar.inverse().unwrap());
    assert_eq!(f, c);
}

fn main() {
    let_cxx_string!(s = "Arkworks");
    ffi::arkworks_benchmark(&s);
}
