use ark_bn254::{ Fr as ScalarField, G1Affine as GAffine, G1Projective as G };
use ark_ec::VariableBaseMSM;
use ark_std::UniformRand;
use std::{ env, mem, time::Instant };

#[cxx::bridge(namespace = "tachyon")]
mod ffi {
    extern "Rust" {
        type CppG1Affine;
        type CppG1Jacobian;
        type CppFq;
        type CppFr;
        fn arkworks_msm(v: Vec<u64>);
    }

    unsafe extern "C++" {
        include!("benchmark/msm/arkworks/include/arkworks_benchmark.h");

        fn get_test_nums(argv: &[String]) -> Vec<u64>;

        fn arkworks_benchmark(
            test_set: &[u64],
            bases: &[CppG1Affine],
            scalars: &[CppFr],
            results_arkworks: &[CppG1Jacobian],
            durations_arkworks: &[f64]
        );
    }
}

#[repr(C, align(32))]
pub struct CppG1Affine {
    pub x: CppFq,
    pub y: CppFq,
    pub infinity: bool,
}

#[repr(C)]
pub struct CppG1Jacobian {
    pub x: CppFq,
    pub y: CppFq,
    pub z: CppFq,
}

#[repr(transparent)]
pub struct CppFq(pub [u64; 4]);

#[repr(transparent)]
pub struct CppFr(pub [u64; 4]);

fn arkworks_msm(test_nums: Vec<u64>) {
    let test_set_size = test_nums.last().unwrap();
    let mut bases = Vec::<GAffine>::new();
    let mut scalars = Vec::<ScalarField>::new();

    println!("Generating random points...");
    let start_gen = Instant::now();
    let mut rng = ark_std::test_rng();
    for _ in 0..*test_set_size {
        let p = GAffine::rand(&mut rng);
        let s = ScalarField::rand(&mut rng);
        bases.push(p);
        scalars.push(s);
    }
    let gen_duration = start_gen.elapsed();
    println!("Generation completed in {:?}", gen_duration);

    let mut results_arkworks = Vec::<G>::new();
    let mut durations_arkworks = Vec::<f64>::new();
    println!();
    println!("Executing Arkworks MSM...");
    for &test_num in &test_nums {
        let start = Instant::now();
        let arkworks_result = G::msm(
            &bases[0..test_num as usize],
            &scalars[0..test_num as usize]
        ).unwrap();
        results_arkworks.push(arkworks_result);
        let duration = start.elapsed();
        let duration_arkworks = duration.as_secs_f64() + (duration.subsec_nanos() as f64) * 1e-9;
        durations_arkworks.push(duration_arkworks);
        println!("calculate: {}", duration_arkworks);
    }

    unsafe {
        let mut bases_tachyon = Vec::<CppG1Affine>::new();
        for i in 0..bases.len() {
            bases_tachyon.push(CppG1Affine {
                x: mem::transmute(bases[i].x),
                y: mem::transmute(bases[i].y),
                infinity: bases[i].infinity,
            });
        }
        let scalars: Vec<CppFr> = mem::transmute(scalars);
        let results_arkworks: Vec<CppG1Jacobian> = mem::transmute(results_arkworks);
        ffi::arkworks_benchmark(
            &test_nums,
            &bases_tachyon,
            &scalars,
            &results_arkworks,
            &durations_arkworks
        );
    }
}

fn main() {
    let argv: Vec<String> = env::args().collect();
    let test_nums = ffi::get_test_nums(&argv);
    arkworks_msm(test_nums);
}
