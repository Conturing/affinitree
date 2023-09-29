//   Copyright 2023 affinitree developers
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

use crate::core::afftree::AffTree;
use crate::linalg::affine::AffFunc;

use ndarray::{Array1, Array2};

#[allow(non_snake_case)]
pub fn ReLU(dim: usize) -> AffTree<2> {
    let mut dd = AffTree::new(dim);
    let mut stack = Vec::new();
    stack.push((0, 0));

    while let Some((row, parent)) = stack.pop() {
        let predicate = dd.tree.tree_node(parent).unwrap().value.aff.clone();

        dd.update_node(parent, predicate.row(row));

        let affine_true = predicate.clone();
        let mut affine_false = predicate;
        affine_false.reset_row(row);

        let child_true_idx = dd.add_child_node(parent, 1, affine_true);
        let child_false_idx = dd.add_child_node(parent, 0, affine_false);

        if row < dim - 1 {
            stack.push((row + 1, child_true_idx));
            stack.push((row + 1, child_false_idx));
        }
    }

    dd
}

#[allow(non_snake_case)]
pub fn partial_ReLU(dim: usize, row: usize) -> AffTree<2> {
    assert!(
        row < dim,
        "Expected row <= dim, got row={} <= dim={}",
        row,
        dim
    );

    let mut dd = AffTree::from_aff(AffFunc::unit(dim, row));

    let affine_true = AffFunc::identity(dim);
    let affine_false = AffFunc::zero_idx(dim, row);

    dd.add_child_node(0, 1, affine_true);
    dd.add_child_node(0, 0, affine_false);

    dd
}

/// Build an AffTree that implements the argmax function.
/// That is, for an input vector x return the first index that contains the maximal element.
pub fn argmax(dim: usize) -> AffTree<2> {
    let affine = AffFunc::subtraction(dim, 0, 1);
    let mut dd = AffTree::from_aff(affine);

    let mut stack = Vec::new();
    stack.push((0, 0, 1));

    while let Some((parent_idx, left, right)) = stack.pop() {
        if right < dim - 1 {
            let affine_true = AffFunc::subtraction(dim, left, right + 1);
            let affine_false = AffFunc::subtraction(dim, right, right + 1);

            let node_true = dd.add_child_node(parent_idx, 1, affine_true);
            let node_false = dd.add_child_node(parent_idx, 0, affine_false);

            stack.push((node_true, left, right + 1));
            stack.push((node_false, right, right + 1));
        } else {
            let affine_true = AffFunc::constant(dim, left as f64);
            let affine_false = AffFunc::constant(dim, right as f64);

            dd.add_child_node(parent_idx, 1, affine_true);
            dd.add_child_node(parent_idx, 0, affine_false);
        }
    }

    dd
}

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

    let affine = AffFunc::subtraction(dim, clazz, iter.next().unwrap());
    let mut dd = AffTree::from_aff(affine);
    let mut last_node = dd.tree.get_root_idx();

    for idx in iter {
        let new_node = dd.add_child_node(last_node, 1, AffFunc::subtraction(dim, clazz, idx));
        dd.add_child_node(last_node, 0, AffFunc::constant(dim, 0.));

        last_node = new_node;
    }

    dd.add_child_node(last_node, 1, AffFunc::constant(dim, 1.));
    dd.add_child_node(last_node, 0, AffFunc::constant(dim, 0.));

    dd
}

pub fn inf_norm(dim: usize, minimum: Option<f64>, maximum: Option<f64>) -> AffTree<2> {
    let min_aff =
        minimum.map(|min| AffFunc::from_mats(Array2::eye(dim), -Array1::from_elem(dim, min)));
    let max_aff =
        maximum.map(|max| AffFunc::from_mats(-Array2::eye(dim), Array1::from_elem(dim, max)));

    let (first, second) = if min_aff.is_some() {
        (min_aff.unwrap(), max_aff)
    } else {
        (
            max_aff.expect("One of minimum and maximum must be specified"),
            None,
        )
    };

    let mut row_iter = first.row_iter();

    let mut dd = AffTree::from_aff(row_iter.next().unwrap());
    let mut last_idx = dd.tree.get_root_idx();

    for aff in row_iter {
        dd.add_child_node(last_idx, 0, AffFunc::constant(dim, 0.));
        last_idx = dd.add_child_node(last_idx, 1, aff);
    }

    if let Some(aff) = second {
        for aff in aff.row_iter() {
            dd.add_child_node(last_idx, 0, AffFunc::constant(dim, 0.));
            last_idx = dd.add_child_node(last_idx, 1, aff);
        }
    }

    dd.add_child_node(last_idx, 0, AffFunc::constant(dim, 0.));
    dd.add_child_node(last_idx, 1, AffFunc::constant(dim, 1.));

    dd
}

#[cfg(test)]
mod tests {

    use crate::core::schema::{class_characterization, inf_norm, partial_ReLU};

    use super::{argmax, ReLU};
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    pub fn test_relu_4() {
        let relu_dd = ReLU(4);

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[-1., -100., -2., 1000.])).unwrap(),
            arr1(&[0., 0., 0., 1000.]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[1., -3e-03, -2., 1.])).unwrap(),
            arr1(&[1., 0., 0., 1.]),
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
    }

    #[test]
    pub fn test_relu_7() {
        let relu_dd = ReLU(7);

        assert_relative_eq!(
            relu_dd
                .evaluate(&arr1(&[-1., -100., -2., 1000., -3e-03, 1.4, 0.1]))
                .unwrap(),
            arr1(&[0., 0., 0., 1000., 0., 1.4, 0.1]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );
    }

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
    }

    #[test]
    pub fn test_argmax() {
        let relu_dd = argmax(4);

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[1., 2., -2., 1.])).unwrap()[0],
            1.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[1., 1., 1., 1.])).unwrap()[0],
            0.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[1., 0., 0., 4.])).unwrap()[0],
            3.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd
                .evaluate(&arr1(&[100., 400., 100000., 7000.]))
                .unwrap()[0],
            2.,
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            relu_dd.evaluate(&arr1(&[1e-5, 1e-4, 1e-2, 1e-3])).unwrap()[0],
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

        println!("{}", inf_dd);

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
