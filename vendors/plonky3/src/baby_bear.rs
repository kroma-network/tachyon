use core::{iter, slice};
use std::{fmt::Debug, marker::PhantomData, ops::Deref};

use p3_baby_bear::BabyBear;
use p3_field::PackedValue;
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrixView},
    Matrix,
};
use tachyon_rs::math::finite_fields::baby_bear::BabyBear as TachyonBabyBearImpl;

pub struct TachyonBabyBear(pub TachyonBabyBearImpl);

#[cxx::bridge(namespace = "tachyon::plonky3_api::baby_bear")]
pub mod ffi {
    extern "Rust" {
        type TachyonBabyBear;
    }

    unsafe extern "C++" {
        include!("vendors/plonky3/include/baby_bear_row_major_matrix.h");

        type RowMajorMatrix;

        unsafe fn new_row_major_matrix(
            data_ptr: *mut TachyonBabyBear,
            rows: usize,
            cols: usize,
        ) -> UniquePtr<RowMajorMatrix>;
        fn get_rows(self: &RowMajorMatrix) -> usize;
        fn get_cols(self: &RowMajorMatrix) -> usize;
        unsafe fn get_const_data_ptr(self: &RowMajorMatrix) -> *const TachyonBabyBear;
        fn clone(&self) -> UniquePtr<RowMajorMatrix>;
    }
}

pub struct RowMajorMatrix<F> {
    inner: cxx::UniquePtr<ffi::RowMajorMatrix>,
    _marker: PhantomData<F>,
}

unsafe impl Send for ffi::RowMajorMatrix {}
unsafe impl Sync for ffi::RowMajorMatrix {}

impl<F> Clone for RowMajorMatrix<F> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _marker: PhantomData,
        }
    }
}

impl<F> Debug for RowMajorMatrix<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowMajorMatrix").finish()
    }
}

impl<F> RowMajorMatrix<F> {
    pub fn new(data_ptr: *mut BabyBear, rows: usize, cols: usize) -> RowMajorMatrix<F> {
        RowMajorMatrix {
            inner: unsafe { ffi::new_row_major_matrix(std::mem::transmute(data_ptr), rows, cols) },
            _marker: PhantomData,
        }
    }

    pub fn get_cols(&self) -> usize {
        self.inner.get_cols()
    }

    pub fn get_rows(&self) -> usize {
        self.inner.get_rows()
    }

    pub fn get_const_data_ptr(&self) -> *const F {
        unsafe { std::mem::transmute(self.inner.get_const_data_ptr()) }
    }
}

impl<F: Clone + Send + Sync> RowMajorMatrix<F> {
    pub fn split_rows(&self, r: usize) -> (RowMajorMatrixView<F>, RowMajorMatrixView<F>) {
        let rows = self.get_rows();
        let cols = self.get_cols();
        let row = unsafe { std::slice::from_raw_parts(self.get_const_data_ptr(), rows * cols) };
        let (lo, hi) = row.split_at(r * cols);
        (DenseMatrix::new(lo, cols), DenseMatrix::new(hi, cols))
    }
}

impl<F: Clone + Send + Sync> Matrix<F> for RowMajorMatrix<F> {
    fn width(&self) -> usize {
        self.get_cols()
    }

    fn height(&self) -> usize {
        self.get_rows()
    }

    type Row<'a> = iter::Cloned<slice::Iter<'a, F>> where Self: 'a;
    fn row(&self, r: usize) -> Self::Row<'_> {
        let slice = unsafe {
            std::slice::from_raw_parts(
                self.get_const_data_ptr()
                    .offset((r * self.width()) as isize),
                self.width(),
            )
        };
        slice.iter().cloned()
    }

    fn row_slice(&self, r: usize) -> impl Deref<Target = [F]> {
        unsafe {
            std::slice::from_raw_parts(
                self.get_const_data_ptr()
                    .offset((r * self.width()) as isize),
                self.width(),
            )
        }
    }

    fn to_row_major_matrix(self) -> p3_matrix::dense::RowMajorMatrix<F>
    where
        Self: Sized,
        F: Clone,
    {
        let size = self.width() * self.height();
        let mut values = Vec::with_capacity(size);
        unsafe {
            std::ptr::copy(self.get_const_data_ptr(), values.as_mut_ptr(), size);
            values.set_len(size);
        }
        p3_matrix::dense::RowMajorMatrix::<F>::new(values, self.get_cols())
    }

    fn horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> (impl Iterator<Item = P>, impl Iterator<Item = F>)
    where
        P: PackedValue<Value = F>,
        F: Clone + 'a,
    {
        let buf = unsafe {
            std::slice::from_raw_parts(
                self.get_const_data_ptr()
                    .offset((r * self.width()) as isize),
                self.width(),
            )
        };
        let (packed, sfx) = P::pack_slice_with_suffix(buf);
        (packed.iter().cloned(), sfx.iter().cloned())
    }
}
