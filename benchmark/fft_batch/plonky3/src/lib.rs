use p3_baby_bear::BabyBear;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use std::time::Instant;
use tachyon_rs::math::finite_fields::baby_bear::BabyBear as CppBabyBear;

#[no_mangle]
pub extern "C" fn run_fft_batch_plonky3_baby_bear(
    data: *const BabyBear,
    n: usize,
    batch_size: usize,
    duration: *mut u64,
) -> *mut CppBabyBear {
    let size = n * batch_size;
    let values: Vec<BabyBear> = unsafe { Vec::from_raw_parts(data as *mut BabyBear, size, size) };

    let messages = RowMajorMatrix::<BabyBear>::new(values, batch_size);
    let dft = Radix2DitParallel::default();

    let start = Instant::now();
    let dft_result = dft.dft_batch(messages).to_row_major_matrix();
    unsafe {
        duration.write(start.elapsed().as_micros() as u64);
    }
    Box::into_raw(dft_result.values.into_boxed_slice()) as *mut CppBabyBear
}

#[no_mangle]
pub extern "C" fn run_coset_lde_batch_plonky3_baby_bear(
    data: *const BabyBear,
    n: usize,
    batch_size: usize,
    duration: *mut u64,
) -> *mut CppBabyBear {
    let size = n * batch_size;
    let values: Vec<BabyBear> = unsafe { Vec::from_raw_parts(data as *mut BabyBear, size, size) };

    let messages = RowMajorMatrix::<BabyBear>::new(values, batch_size);
    let dft = Radix2DitParallel::default();

    let start = Instant::now();
    let shift = BabyBear::zero();
    let dft_result = dft.coset_lde_batch(messages, 0, shift).to_row_major_matrix();
    unsafe {
        duration.write(start.elapsed().as_micros() as u64);
    }
    Box::into_raw(dft_result.values.into_boxed_slice()) as *mut CppBabyBear
}
