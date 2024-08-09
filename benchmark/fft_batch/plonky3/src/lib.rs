use p3_baby_bear::BabyBear;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::thread_rng;
use std::time::Instant;
use tachyon_rs::math::elliptic_curves::bn::bn254::Fr as CppFr;

#[no_mangle]
pub extern "C" fn run_fft_batch_plonky3_baby_bear(
    duration: *mut u64,
    data: *mut BabyBear,
    n_log: usize,
    batch_size: usize,
) -> *mut RowMajorMatrix<BabyBear> {
    let mut rng = thread_rng();
    let n = 1 << n_log;
    let size = n * batch_size;
    let values: Vec<BabyBear> = unsafe { Vec::from_raw_parts(data, size, size) };

    let messages = RowMajorMatrix::<BabyBear>::new(values, batch_size);
    let dft = Radix2DitParallel::default();

    let start = Instant::now();
    let dft_result = dft.dft_batch(messages).to_row_major_matrix();
    unsafe {
        duration.write(start.elapsed().as_micros() as u64);
    }
    Box::into_raw(Box::new(dft_result)) as *mut RowMajorMatrix<BabyBear>
}
