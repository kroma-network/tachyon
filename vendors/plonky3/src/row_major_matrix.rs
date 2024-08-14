#[cfg(test)]
mod test {
    use p3_baby_bear::BabyBear;
    use p3_field::{AbstractField, Field};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};

    use crate::baby_bear::RowMajorMatrix as TachyonRowMajorMatrix;

    #[test]
    fn test_row_major_matrix() {
        let mut data = (0..30)
            .map(|i| BabyBear::from_canonical_u32(i))
            .collect::<Vec<_>>();
        let tachyon_matrix: TachyonRowMajorMatrix<BabyBear> =
            TachyonRowMajorMatrix::new(data.as_mut_ptr(), 5, 6);
        let matrix = RowMajorMatrix::new(
            (0..30)
                .map(|i| BabyBear::from_canonical_u32(i))
                .collect::<Vec<_>>(),
            6,
        );
        assert_eq!(tachyon_matrix.width(), matrix.width());
        assert_eq!(tachyon_matrix.height(), matrix.height());
        assert_eq!(tachyon_matrix.get(3, 3), matrix.get(3, 3));

        assert_eq!(
            tachyon_matrix.row(3).collect::<Vec<_>>(),
            matrix.row(3).collect::<Vec<_>>()
        );

        assert_eq!(
            tachyon_matrix.clone().to_row_major_matrix(),
            matrix.clone().to_row_major_matrix()
        );

        let tachyon_row = tachyon_matrix.horizontally_packed_row::<<BabyBear as Field>::Packing>(3);
        let row = matrix.horizontally_packed_row::<<BabyBear as Field>::Packing>(3);
        assert_eq!(tachyon_row.0.collect::<Vec<_>>(), row.0.collect::<Vec<_>>());
        assert_eq!(tachyon_row.1.collect::<Vec<_>>(), row.1.collect::<Vec<_>>());

        let tachyon_rows = tachyon_matrix.split_rows(3);
        let rows = matrix.split_rows(3);
        assert_eq!(tachyon_rows.0, rows.0);
        assert_eq!(tachyon_rows.1, rows.1);
    }
}
