use super::LinearOperator;

/// A square or rectangular sparse matrix in compressed sparse row format.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix {
    n_rows: usize,
    n_cols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<f64>,
}

impl CsrMatrix {
    /// Builds a matrix from `(row, col, value)` triplets. Duplicate entries are summed;
    /// columns within each row end up sorted.
    ///
    /// # Panics
    /// If any index is out of range.
    pub fn from_triplets(n_rows: usize, n_cols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        let mut sorted: Vec<(usize, usize, f64)> = triplets.to_vec();
        for &(r, c, _) in &sorted {
            assert!(
                r < n_rows && c < n_cols,
                "triplet ({r}, {c}) outside {n_rows}x{n_cols}"
            );
        }
        sorted.sort_unstable_by_key(|&(r, c, _)| (r, c));

        let mut row_ptr = vec![0usize; n_rows + 1];
        let mut col_idx = Vec::with_capacity(sorted.len());
        let mut values = Vec::with_capacity(sorted.len());
        let mut last: Option<(usize, usize)> = None;
        for (r, c, v) in sorted {
            if last == Some((r, c)) {
                *values.last_mut().expect("entry exists") += v;
            } else {
                col_idx.push(c);
                values.push(v);
                row_ptr[r + 1] += 1;
                last = Some((r, c));
            }
        }
        for i in 0..n_rows {
            row_ptr[i + 1] += row_ptr[i];
        }
        Self {
            n_rows,
            n_cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Entry `(row, col)`, or 0 if it is not stored.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        let range = self.row_ptr[row]..self.row_ptr[row + 1];
        match self.col_idx[range.clone()].binary_search(&col) {
            Ok(k) => self.values[range.start + k],
            Err(_) => 0.0,
        }
    }

    /// Iterates over the stored `(col, value)` pairs of `row`.
    pub fn row(&self, row: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let range = self.row_ptr[row]..self.row_ptr[row + 1];
        self.col_idx[range.clone()]
            .iter()
            .copied()
            .zip(self.values[range].iter().copied())
    }

    /// Computes `y = A x` for a possibly rectangular matrix.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n_cols);
        assert_eq!(y.len(), self.n_rows);
        for (r, yr) in y.iter_mut().enumerate() {
            let (s, e) = (self.row_ptr[r], self.row_ptr[r + 1]);
            *yr = self.col_idx[s..e]
                .iter()
                .zip(&self.values[s..e])
                .map(|(&c, &v)| v * x[c])
                .sum();
        }
    }
}

impl LinearOperator for CsrMatrix {
    fn len(&self) -> usize {
        assert_eq!(self.n_rows, self.n_cols, "operator must be square");
        self.n_rows
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.matvec(x, y);
    }

    fn diagonal(&self, out: &mut [f64]) {
        for (r, d) in out.iter_mut().enumerate() {
            *d = self.get(r, r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicates_are_summed_and_columns_sorted() {
        let a = CsrMatrix::from_triplets(
            2,
            3,
            &[
                (1, 2, 1.0),
                (0, 1, 2.0),
                (1, 0, 3.0),
                (0, 1, 0.5),
                (1, 2, -4.0),
            ],
        );
        assert_eq!(a.nnz(), 3);
        assert_eq!(a.get(0, 1), 2.5);
        assert_eq!(a.get(1, 2), -3.0);
        assert_eq!(a.get(0, 0), 0.0);
        let cols: Vec<usize> = a.row(1).map(|(c, _)| c).collect();
        assert_eq!(cols, vec![0, 2]);
    }

    #[test]
    fn matvec_matches_dense_product() {
        let a = CsrMatrix::from_triplets(
            3,
            3,
            &[
                (0, 0, 4.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 4.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 4.0),
            ],
        );
        let mut y = [0.0; 3];
        a.matvec(&[1.0, 2.0, 3.0], &mut y);
        assert_eq!(y, [2.0, 4.0, 10.0]);
        let mut d = [0.0; 3];
        a.diagonal(&mut d);
        assert_eq!(d, [4.0; 3]);
    }

    #[test]
    fn empty_rows_are_handled() {
        let a = CsrMatrix::from_triplets(3, 3, &[(2, 0, 1.0)]);
        let mut y = [9.0; 3];
        a.matvec(&[5.0, 0.0, 0.0], &mut y);
        assert_eq!(y, [0.0, 0.0, 5.0]);
    }
}
