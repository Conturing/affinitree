/// The following test case has as subject the absolute value function.
///
/// The test case first manually constructs the function and then asserts
/// correctness on some inputs.
#[cfg(test)]
mod tests {
    use affinitree::core::afftree::AffTree;
    use affinitree::linalg::affine::PolyRepr;
    use affinitree::{aff, poly};

    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    pub fn test_abs() {
        let mut dd =
            AffTree::<2>::from_aff(poly!([[1, 1]] < [0]).convert_to(PolyRepr::MatrixBiasGeqZero)); // 0

        dd.add_decision(
            0,
            0,
            poly!([[-1, 1]] < [0]).convert_to(PolyRepr::MatrixBiasGeqZero),
        ); // 1
        dd.add_decision(
            0,
            1,
            poly!([[-1, 1]] < [0]).convert_to(PolyRepr::MatrixBiasGeqZero),
        ); // 2

        dd.add_terminal(1, 0, aff!([[0, 1]] + [0]));
        dd.add_terminal(1, 1, aff!([[1, 0]] + [0]));
        dd.add_terminal(2, 0, aff!([[-1, 0]] + [0]));
        dd.add_terminal(2, 1, aff!([[0, -1]] + [0]));

        assert_eq!(dd.len(), 7);
        assert_eq!(dd.num_terminals(), 4);

        assert_relative_eq!(dd.evaluate(&arr1(&[0., 0.])).unwrap()[0], 0.);
        assert_relative_eq!(dd.evaluate(&arr1(&[2., 0.])).unwrap()[0], 2.);
        assert_relative_eq!(dd.evaluate(&arr1(&[-3., 0.])).unwrap()[0], 3.);
        assert_relative_eq!(dd.evaluate(&arr1(&[0., -1.])).unwrap()[0], 1.);
        assert_relative_eq!(dd.evaluate(&arr1(&[0., 10.])).unwrap()[0], 10.);
    }
}
