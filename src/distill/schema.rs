//   Copyright 2025 affinitree developers
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

//! A collection of common piece-wise linear functions like activation functions

use ndarray::{Array1, Array2};

use crate::linalg::affine::AffFunc;
use crate::pwl::afftree::AffTree;

/// Creates an AffTree instance that corresponds to the ReLU function applied
/// to the specified ``row``.
///
/// Formally, it is defined as (partial_ReLU(x))_row = max {0, x_row}
#[allow(non_snake_case)]
pub fn partial_ReLU(dim: usize, row: usize) -> AffTree<2> {
    assert!(
        row < dim,
        "Expected row <= dim, got row={} <= dim={}",
        row,
        dim
    );

    // x_{row} <= 0
    let mut dd = AffTree::from_aff(AffFunc::unit(dim, row));

    let affine_false = AffFunc::identity(dim);
    let affine_true = AffFunc::zero_idx(dim, row);

    dd.add_child_node(0, 0, affine_false).unwrap();
    dd.add_child_node(0, 1, affine_true).unwrap();

    dd
}

/// Creates an AffTree instance that corresponds to the leaky ReLU function applied
/// to the specified ``row``.
///
/// Formally, it is defined as (partial_leaky_ReLU(x))_row = x_row if x_row > 0 else alpha * x_row
#[allow(non_snake_case)]
pub fn partial_leaky_ReLU(dim: usize, row: usize, alpha: f64) -> AffTree<2> {
    assert!(
        row < dim,
        "Expected row <= dim, got row={} <= dim={}",
        row,
        dim
    );

    // x_{row} <= 0
    let mut dd = AffTree::from_aff(AffFunc::unit(dim, row));

    let mut affine_true = AffFunc::zero_idx(dim, row);
    affine_true.mat[[row, row]] = alpha;
    let affine_false = AffFunc::identity(dim);

    dd.add_child_node(0, 0, affine_false).unwrap();
    dd.add_child_node(0, 1, affine_true).unwrap();

    dd
}

/// Creates an AffTree instance that corresponds to the hard hyperbolic tangent function applied
/// to the specified ``row``.
#[allow(non_snake_case)]
pub fn partial_hard_tanh(dim: usize, row: usize, min_val: f64, max_val: f64) -> AffTree<2> {
    assert!(
        row < dim,
        "Expected row <= dim, got row={} <= dim={}",
        row,
        dim
    );
    assert!(
        min_val <= max_val,
        "Expected min_val to be lower than or equal to max_val, but got {} > {}",
        min_val,
        max_val
    );

    let mut aff = AffFunc::unit(dim, row);
    aff.mat[[0, row]] = -1.0;
    aff.bias[0] = -max_val;
    let mut dd = AffTree::from_aff(aff);

    let mut aff = AffFunc::unit(dim, row);
    aff.mat[[0, row]] = 1.0;
    aff.bias[0] = min_val;

    let mut affine_max = AffFunc::zero_idx(dim, row);
    affine_max.bias[row] = max_val;

    let n = dd.add_child_node(0, 0, aff).unwrap();
    dd.add_child_node(0, 1, affine_max).unwrap();

    let affine_id = AffFunc::identity(dim);
    let mut affine_min = AffFunc::zero_idx(dim, row);
    affine_min.bias[row] = min_val;

    dd.add_child_node(n, 0, affine_id).unwrap();
    dd.add_child_node(n, 1, affine_min).unwrap();

    dd
}

/// Creates an AffTree instance that corresponds to the hard shrink function applied
/// to the specified ``row``.
#[allow(non_snake_case)]
pub fn partial_hard_shrink(dim: usize, row: usize, lambda: f64) -> AffTree<2> {
    assert!(
        row < dim,
        "Expected row <= dim, got row={} <= dim={}",
        row,
        dim
    );

    let mut aff = AffFunc::unit(dim, row);
    aff.mat[[0, row]] = -1.0;
    aff.bias[0] = -lambda;
    let mut dd = AffTree::from_aff(aff);

    let mut aff = AffFunc::unit(dim, row);
    aff.mat[[0, row]] = 1.0;
    aff.bias[0] = -lambda;

    let affine_max = AffFunc::identity(dim);

    let n = dd.add_child_node(0, 0, aff).unwrap();
    dd.add_child_node(0, 1, affine_max).unwrap();

    let affine_zero = AffFunc::zero_idx(dim, row);
    let affine_min = AffFunc::identity(dim);

    dd.add_child_node(n, 0, affine_zero).unwrap();
    dd.add_child_node(n, 1, affine_min).unwrap();

    dd
}

/// Creates an AffTree instance that corresponds to the hard sigmoid function applied
/// to the specified ``row``.
#[allow(non_snake_case)]
pub fn partial_hard_sigmoid(dim: usize, row: usize) -> AffTree<2> {
    assert!(
        row < dim,
        "Expected row <= dim, got row={} <= dim={}",
        row,
        dim
    );

    let mut aff = AffFunc::unit(dim, row);
    aff.mat[[0, row]] = -1.0;
    aff.bias[0] = -3.;
    let mut dd = AffTree::from_aff(aff);

    let mut aff = AffFunc::unit(dim, row);
    aff.mat[[0, row]] = 1.0;
    aff.bias[0] = -3.;

    let mut affine_max = AffFunc::zero_idx(dim, row);
    affine_max.bias[row] = 1.;

    let n = dd.add_child_node(0, 0, aff).unwrap();
    dd.add_child_node(0, 1, affine_max).unwrap();

    let mut affine_id = AffFunc::identity(dim);
    affine_id.mat[[row, row]] = 1. / 6.;
    affine_id.bias[row] = 0.5;

    let mut affine_min = AffFunc::zero_idx(dim, row);
    affine_min.bias[row] = 0.;

    dd.add_child_node(n, 0, affine_id).unwrap();
    dd.add_child_node(n, 1, affine_min).unwrap();

    dd
}

/// Creates an AffTree instance that corresponds to the threshold function applied
/// to the specified ``row``.
#[allow(non_snake_case)]
pub fn partial_threshold(dim: usize, row: usize, threshold: f64, value: f64) -> AffTree<2> {
    assert!(
        row < dim,
        "Expected row <= dim, got row={} <= dim={}",
        row,
        dim
    );

    let mut aff = AffFunc::unit(dim, row);
    aff.bias[0] = threshold;
    let mut dd = AffTree::from_aff(aff);

    let mut affine_true = AffFunc::zero_idx(dim, row);
    affine_true.bias[row] = value;
    let affine_false = AffFunc::identity(dim);

    dd.add_child_node(0, 0, affine_false).unwrap();
    dd.add_child_node(0, 1, affine_true).unwrap();

    dd
}

/// Create an AffTree instance that corresponds to the argmax function.
/// That is, for an input vector x return the first index i such that x_i contains the maximal element.
pub fn argmax(dim: usize) -> AffTree<2> {
    let affine = AffFunc::subtraction(dim, 1, 0);
    let mut dd = AffTree::from_aff(affine);

    let mut stack = Vec::new();
    stack.push((0, 1, 0));

    while let Some((parent_idx, max_when_false, max_when_true)) = stack.pop() {
        if max_when_false < dim - 1 {
            let affine_false = AffFunc::subtraction(dim, max_when_false + 1, max_when_false);
            let affine_true = AffFunc::subtraction(dim, max_when_false + 1, max_when_true);

            let node_false = dd.add_child_node(parent_idx, 0, affine_false).unwrap();
            let node_true = dd.add_child_node(parent_idx, 1, affine_true).unwrap();

            stack.push((node_false, max_when_false + 1, max_when_false));
            stack.push((node_true, max_when_false + 1, max_when_true));
        } else {
            let affine_false = AffFunc::constant(dim, max_when_false as f64);
            let affine_true = AffFunc::constant(dim, max_when_true as f64);

            dd.add_child_node(parent_idx, 0, affine_false).unwrap();
            dd.add_child_node(parent_idx, 1, affine_true).unwrap();
        }
    }

    dd
}

/// Creates an AffTree instance that corresponds to the class characterization.
/// That is, an indicator function that shows if the argmax of its input vector
/// coincides with the specified ``clazz``, i.e., if the value of the input at
/// position ``clazz`` is maximal.
pub fn class_characterization(dim: usize, clazz: usize) -> AffTree<2> {
    assert!(
        clazz < dim,
        "Class lies outside bounds, class={} and dim={}",
        clazz,
        dim
    );
    assert!(
        dim >= 2,
        "Class characterization can only be applied at two or more dimensions, got {}",
        dim
    );

    let mut iter = (0..dim).filter(|x| *x != clazz);

    let affine = AffFunc::subtraction(dim, iter.next().unwrap(), clazz);
    let mut dd = AffTree::from_aff(affine);
    let mut last_node = dd.tree.get_root_idx();

    for idx in iter {
        dd.add_child_node(last_node, 0, AffFunc::constant(dim, 0.))
            .unwrap();
        let new_node = dd
            .add_child_node(last_node, 1, AffFunc::subtraction(dim, idx, clazz))
            .unwrap();

        last_node = new_node;
    }

    dd.add_child_node(last_node, 0, AffFunc::constant(dim, 0.))
        .unwrap();
    dd.add_child_node(last_node, 1, AffFunc::constant(dim, 1.))
        .unwrap();

    dd
}

/// Creates an AffTree instance that tests whether an input has an infinity norm
/// bounded by ``minimum`` and ``maximum``, if specified.
///
/// If true, a constant 1 is returned, otherwise a constant 0.
pub fn inf_norm(dim: usize, minimum: Option<f64>, maximum: Option<f64>) -> AffTree<2> {
    let min_aff =
        minimum.map(|min| AffFunc::from_mats(-Array2::eye(dim), -Array1::from_elem(dim, min)));
    let max_aff =
        maximum.map(|max| AffFunc::from_mats(Array2::eye(dim), Array1::from_elem(dim, max)));

    let (first, second) = match (min_aff, max_aff) {
        (Some(a), Some(b)) => (a, Some(b)),
        (Some(a), None) => (a, None),
        (None, Some(b)) => (b, None),
        (None, None) => panic!("One of minimum and maximum must be specified"),
    };

    let mut row_iter = first.row_iter();

    let mut dd = AffTree::from_aff(row_iter.next().unwrap().to_owned());
    let mut last_idx = dd.tree.get_root_idx();

    for aff in row_iter {
        dd.add_child_node(last_idx, 0, AffFunc::constant(dim, 0.))
            .unwrap();
        last_idx = dd.add_child_node(last_idx, 1, aff.to_owned()).unwrap();
    }

    if let Some(aff) = second {
        for aff in aff.row_iter() {
            dd.add_child_node(last_idx, 0, AffFunc::constant(dim, 0.))
                .unwrap();
            last_idx = dd.add_child_node(last_idx, 1, aff.to_owned()).unwrap();
        }
    }

    dd.add_child_node(last_idx, 0, AffFunc::constant(dim, 0.))
        .unwrap();
    dd.add_child_node(last_idx, 1, AffFunc::constant(dim, 1.))
        .unwrap();

    dd
}

#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;
    use ndarray::arr1;

    use super::*;

    #[test]
    pub fn test_partial_relu_4() {
        let relu_dd = partial_ReLU(4, 1);

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[-1., -100., -2., 1000.])).unwrap(),
            arr1(&[-1., 0., -2., 1000.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[1., -3e-03, -2., 1.])).unwrap(),
            arr1(&[1., 0., -2., 1.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[1.4, 1e-03, 0.3, 4.])).unwrap(),
            arr1(&[1.4, 1e-03, 0.3, 4.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[1.5, -1., 0., 4.])).unwrap(),
            arr1(&[1.5, 0., 0., 4.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[-1.6, 11., 2., 1.])).unwrap(),
            arr1(&[-1.6, 11., 2., 1.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );
    }

    #[test]
    pub fn test_partial_leaky_relu_4() {
        let relu_dd = partial_leaky_ReLU(4, 1, 0.1);

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[-1., -100., -2., 1000.])).unwrap(),
            arr1(&[-1., -10., -2., 1000.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[1., -3e-03, -2., 1.])).unwrap(),
            arr1(&[1., -3e-04, -2., 1.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[1.4, 1e-03, 0.3, 4.])).unwrap(),
            arr1(&[1.4, 1e-03, 0.3, 4.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[1.5, -1., 0., 4.])).unwrap(),
            arr1(&[1.5, -0.1, 0., 4.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[-1.6, 11., 2., 1.])).unwrap(),
            arr1(&[-1.6, 11., 2., 1.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );
    }

    #[test]
    pub fn test_partial_hard_tanh_4() {
        let tanh_tree = partial_hard_tanh(4, 1, -5.0, 7.0);

        assert_relative_eq!(
            tanh_tree
                .evaluate(&arr1(&[-1., -100., -2., 1000.]))
                .unwrap(),
            arr1(&[-1., -5., -2., 1000.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            tanh_tree.evaluate(&arr1(&[1., -3e-03, -2., 1.])).unwrap(),
            arr1(&[1., -3e-03, -2., 1.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            tanh_tree.evaluate(&arr1(&[1.4, 1e-03, 0.3, 4.])).unwrap(),
            arr1(&[1.4, 1e-03, 0.3, 4.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            tanh_tree.evaluate(&arr1(&[1.5, -1., 0., 4.])).unwrap(),
            arr1(&[1.5, -1.0, 0., 4.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            tanh_tree.evaluate(&arr1(&[-1.6, 11., 2., 1.])).unwrap(),
            arr1(&[-1.6, 7., 2., 1.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );
    }

    #[test]
    pub fn test_partial_hard_shrink_4() {
        let shrink_tree = partial_hard_shrink(4, 1, 0.5);

        assert_relative_eq!(
            shrink_tree
                .evaluate(&arr1(&[-1., -100., -2., 1000.]))
                .unwrap(),
            arr1(&[-1., -100., -2., 1000.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            shrink_tree.evaluate(&arr1(&[1., -3e-03, -2., 1.])).unwrap(),
            arr1(&[1., 0., -2., 1.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            shrink_tree.evaluate(&arr1(&[1.4, 1e-03, 0.3, 4.])).unwrap(),
            arr1(&[1.4, 0., 0.3, 4.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            shrink_tree.evaluate(&arr1(&[1.5, -1., 0., 4.])).unwrap(),
            arr1(&[1.5, -1., 0., 4.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            shrink_tree.evaluate(&arr1(&[-1.6, 11., 2., 1.])).unwrap(),
            arr1(&[-1.6, 11., 2., 1.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );
    }

    #[test]
    pub fn test_partial_hard_sigmoid_4() {
        let sigmoid_tree = partial_hard_sigmoid(4, 1);

        assert_relative_eq!(
            sigmoid_tree
                .evaluate(&arr1(&[-1., -100., -2., 1000.]))
                .unwrap(),
            arr1(&[-1., 0., -2., 1000.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            sigmoid_tree
                .evaluate(&arr1(&[1., -3e-03, -2., 1.]))
                .unwrap(),
            arr1(&[1., 0.4995, -2., 1.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            sigmoid_tree
                .evaluate(&arr1(&[1.4, 6e-03, 0.3, 4.]))
                .unwrap(),
            arr1(&[1.4, 0.501, 0.3, 4.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            sigmoid_tree.evaluate(&arr1(&[1.5, -1.2, 0., 4.])).unwrap(),
            arr1(&[1.5, 0.3, 0., 4.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            sigmoid_tree.evaluate(&arr1(&[-1.6, 11., 2., 1.])).unwrap(),
            arr1(&[-1.6, 1., 2., 1.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );
    }

    #[test]
    pub fn test_partial_threshold_4() {
        let threshold_tree = partial_threshold(4, 1, -0.5, -5.);

        assert_relative_eq!(
            threshold_tree
                .evaluate(&arr1(&[-1., -100., -2., 1000.]))
                .unwrap(),
            arr1(&[-1., -5., -2., 1000.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            threshold_tree
                .evaluate(&arr1(&[1., -3e-03, -2., 1.]))
                .unwrap(),
            arr1(&[1., -3e-03, -2., 1.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            threshold_tree
                .evaluate(&arr1(&[1.4, 1e-03, 0.3, 4.]))
                .unwrap(),
            arr1(&[1.4, 1e-03, 0.3, 4.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            threshold_tree.evaluate(&arr1(&[1.5, -1., 0., 4.])).unwrap(),
            arr1(&[1.5, -5., 0., 4.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            threshold_tree
                .evaluate(&arr1(&[-1.6, 11., 2., 1.]))
                .unwrap(),
            arr1(&[-1.6, 11., 2., 1.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );
    }

    #[test]
    pub fn test_argmax() {
        let argmax_tree = argmax(4);

        assert_relative_eq!(
            argmax_tree.evaluate(&arr1(&[1., 2., -2., 1.])).unwrap()[0],
            1.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            argmax_tree.evaluate(&arr1(&[1., 1., 1., 1.])).unwrap()[0],
            0.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            argmax_tree.evaluate(&arr1(&[1., 0., 0., 4.])).unwrap()[0],
            3.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            argmax_tree
                .evaluate(&arr1(&[100., 400., 100000., 7000.]))
                .unwrap()[0],
            2.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            argmax_tree
                .evaluate(&arr1(&[1e-5, 1e-4, 1e-2, 1e-3]))
                .unwrap()[0],
            2.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );
    }

    #[test]
    pub fn test_class_characterization_1() {
        let class_dd = class_characterization(4, 1);

        assert_relative_eq!(
            class_dd.evaluate(&arr1(&[1., 2., -2., 1.])).unwrap()[0],
            1.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            class_dd.evaluate(&arr1(&[1., 1., 1., 1.])).unwrap()[0],
            1.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            class_dd.evaluate(&arr1(&[4., 0., 0., 1.])).unwrap()[0],
            0.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            class_dd
                .evaluate(&arr1(&[100., 400., 100000., 7000.]))
                .unwrap()[0],
            0.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            class_dd.evaluate(&arr1(&[1e-5, 1e-2, 1e-4, 1e-3])).unwrap()[0],
            1.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );
    }

    #[test]
    pub fn test_class_characterization_0() {
        let class_dd = class_characterization(4, 0);

        assert_relative_eq!(
            class_dd.evaluate(&arr1(&[1., 2., -2., 1.])).unwrap()[0],
            0.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            class_dd.evaluate(&arr1(&[1., 1., 1., 1.])).unwrap()[0],
            1.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            class_dd.evaluate(&arr1(&[4., 0., 0., 1.])).unwrap()[0],
            1.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            class_dd
                .evaluate(&arr1(&[100., 400., 100000., 7000.]))
                .unwrap()[0],
            0.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            class_dd.evaluate(&arr1(&[1e-5, 1e-2, 1e-4, 1e-3])).unwrap()[0],
            0.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );
    }

    #[test]
    pub fn test_inf_norm() {
        let inf_dd = inf_norm(4, Some(-2.), Some(5.));

        assert_relative_eq!(
            inf_dd.evaluate(&arr1(&[1., 2., -1.9, 1.])).unwrap()[0],
            1.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            inf_dd.evaluate(&arr1(&[1., 2., -2.5, 1.])).unwrap()[0],
            0.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            inf_dd.evaluate(&arr1(&[1., 1., 1., 1.])).unwrap()[0],
            1.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            inf_dd.evaluate(&arr1(&[4., 0., 0., 1.])).unwrap()[0],
            1.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            inf_dd.evaluate(&arr1(&[5.5, 0., 0., 1.])).unwrap()[0],
            0.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            inf_dd
                .evaluate(&arr1(&[100., 400., 100000., 7000.]))
                .unwrap()[0],
            0.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            inf_dd.evaluate(&arr1(&[1e-5, 1e-2, 1e-4, 1e-3])).unwrap()[0],
            1.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );
    }
}
