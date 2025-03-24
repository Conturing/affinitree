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

use std::ops::{Add, Div, Mul, Neg, Sub};

use itertools::Itertools;
use log::debug;

use super::afftree::*;
use super::impl_composition::{CompositionSchema, NoOpVis};
use crate::linalg::affine::*;
use crate::tree::graph::*;

// Implementation of arithmetic and logic operators for AffTree

// Implementation of macros inspired by ndarray

macro_rules! forward_trait_impl(
    ($trt:ident, $op:ident, $doc:expr) => (

    /// Performs the binary operation
    #[doc=$doc]
    /// on self and rhs.
    /// This operation is the result of lifting [`AffFunc::
    #[doc=stringify!($op)]
    /// `] to AffTree.
    ///
    /// In this variant, new memory is allocated.
    impl<const K: usize> $trt<AffTree<K>> for &AffTree<K> {
        type Output = AffTree<K>;

        fn $op(self, rhs: AffTree<K>) -> Self::Output {
            self.$op(&rhs)
        }
    }

    /// Performs the binary operation
    #[doc=$doc]
    /// on self and rhs.
    /// This operation is the result of lifting [`AffFunc::
    #[doc=stringify!($op)]
    /// `] to AffTree.
    ///
    /// In this variant, new memory is allocated.
    impl<const K: usize> $trt<&AffTree<K>> for &AffTree<K> {
        type Output = AffTree<K>;

        fn $op(self, rhs: &AffTree<K>) -> Self::Output {
            self.clone().$op(rhs)
        }
    }

    /// Performs the binary operation
    #[doc=$doc]
    /// on self and rhs.
    /// This operation is the result of lifting [`AffFunc::
    #[doc=stringify!($op)]
    /// `] to AffTree.
    ///
    /// In this variant, self is moved and the operations is performed in place.
    impl<const K: usize> $trt<AffTree<K>> for AffTree<K> {
        type Output = AffTree<K>;

        fn $op(self, rhs: AffTree<K>) -> Self::Output {
            self.$op(&rhs)
        }
    }

    )
);

macro_rules! generate_infeasible {
    ("infeasible") => {
        fn explore<const K: usize>(
            context: &AffTree<K>,
            parent: TreeIndex,
            child: TreeIndex,
        ) -> bool {
            context.is_edge_feasible(parent, child)
        }
    };
    ("true") => {
        fn explore<const K: usize>(_: &AffTree<K>, _: TreeIndex, _: TreeIndex) -> bool {
            true
        }
    };
}

macro_rules! impl_op_schema(
    ($trt:ident, $op:ident, $name:ident, $doc:expr, $inf:tt) => (

    struct $name {}

    impl CompositionSchema for $name {
        fn update_decision(original: &AffFunc, _: &AffFunc) -> AffFunc {
            original.to_owned()
        }

        fn update_terminal(original: &AffFunc, context: &AffFunc) -> AffFunc {
            debug!("applying {} to {} and {}", stringify!($op), context, original);
            context.clone().$op(original)
        }

        generate_infeasible!($inf);
    }

    forward_trait_impl!($trt, $op, $doc);

    /// Performs the binary operation
    #[doc=$doc]
    /// on self and rhs.
    /// This operation is the result of lifting [`AffFunc::
    #[doc=stringify!($op)]
    /// `] to AffTree.
    ///
    /// In this variant, self is moved and the operations is performed in place.
    impl<const K: usize> $trt<&AffTree<K>> for AffTree<K> {
        type Output = AffTree<K>;

        fn $op(mut self, rhs: &AffTree<K>) -> Self::Output {
            let terminals = self.tree.terminal_indices().collect_vec();
            AffTree::<K>::generic_composition_inplace(
                rhs,
                &mut self,
                terminals,
                $name {},
                NoOpVis {}
            );
            self
        }
    }
    )
);

impl_op_schema!(Add, add, AdditionSchema, "addition", "infeasible");
impl_op_schema!(Sub, sub, SubtractionSchema, "subtraction", "infeasible");
#[rustfmt::skip]
impl_op_schema!(Mul, mul, MultiplicationSchema, "multiplication", "infeasible");
impl_op_schema!(Div, div, DivisionSchema, "division", "infeasible");

impl<const K: usize> AffTree<K> {
    /// Applies the given unary operation ``op`` to all terminals of this tree.
    pub fn unary_op_inplace<F>(&mut self, op: F) -> &mut Self
    where
        F: Fn(AffFunc) -> AffFunc,
    {
        for terminal in self.tree.terminals_mut() {
            take_mut::take(&mut terminal.value.aff, &op);
        }
        self
    }

    /// Applies the given unary operation ``op`` to all terminals of this tree.
    pub fn unary_op_into<F>(mut self, op: F) -> Self
    where
        F: Fn(AffFunc) -> AffFunc,
    {
        self.unary_op_inplace(op);
        self
    }
}

impl<const K: usize> Neg for AffTree<K> {
    type Output = AffTree<K>;

    fn neg(self) -> Self::Output {
        self.unary_op_into(|node| node.neg())
    }
}

macro_rules! impl_scalar_op(
    ($trt:ident, $mth:ident) => (

    impl<const K: usize> $trt<AffFunc> for AffTree<K> {
        type Output = AffTree<K>;

        fn $mth(self, rhs: AffFunc) -> Self::Output {
            self.$mth(&rhs)
        }
    }

    impl<const K: usize> $trt<&AffFunc> for AffTree<K> {
        type Output = AffTree<K>;

        fn $mth(self, rhs: &AffFunc) -> Self::Output {
            self.unary_op_into(|node| node.$mth(rhs))
        }
    }

    impl<const K: usize> $trt<AffTree<K>> for AffFunc {
        type Output = AffTree<K>;

        fn $mth(self, rhs: AffTree<K>) -> Self::Output {
            (&self).$mth(rhs)
        }
    }

    impl<const K: usize> $trt<AffTree<K>> for &AffFunc {
        type Output = AffTree<K>;

        fn $mth(self, rhs: AffTree<K>) -> Self::Output {
            rhs.unary_op_into(|node| self.clone().$mth(node))
        }
    }

    )
);

impl_scalar_op!(Add, add);
impl_scalar_op!(Sub, sub);
impl_scalar_op!(Div, div);
impl_scalar_op!(Mul, mul);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distill::schema;
    use crate::{aff, path};

    #[test]
    fn test_add() {
        let mut dd0 = AffTree::<2>::from_aff(aff!([-1., 2.] + -1.));
        dd0.compose::<false, false>(&schema::partial_ReLU(1, 0));

        let mut dd1 = AffTree::<2>::from_aff(aff!([-2., -1.] + 3.));
        dd1.compose::<false, false>(&schema::partial_ReLU(1, 0));

        let dd = dd0 + dd1;

        assert_eq!(
            dd.tree.node_value(path!(dd.tree, 1, 1)).unwrap().aff,
            aff!([0., 0.] + 0.)
        );
        assert_eq!(
            dd.tree.node_value(path!(dd.tree, 1, 0)).unwrap().aff,
            aff!([-2., -1.] + 3.)
        );
        assert_eq!(
            dd.tree.node_value(path!(dd.tree, 0, 1)).unwrap().aff,
            aff!([-1., 2.] + -1.)
        );
        assert_eq!(
            dd.tree.node_value(path!(dd.tree, 0, 0)).unwrap().aff,
            aff!([-3., 1.] + 2.)
        );
    }

    #[test]
    fn test_add2() {
        let mut dd0 = AffTree::<2>::from_aff(aff!([-1., 2.] + -1.));
        dd0.compose::<false, false>(&schema::partial_ReLU(1, 0));

        let mut dd1 = AffTree::<2>::from_aff(aff!([-2., -1.] + 3.));
        dd1.compose::<false, false>(&schema::partial_ReLU(1, 0));

        // allocates new space
        let dd2 = &dd0 + &dd1;
        assert_ne!(dd2.len(), dd0.len());

        // moves dd0 and performs addition inplace
        let dd3 = dd0 + &dd1;
        assert_eq!(dd3.len(), dd2.len());
    }

    #[test]
    fn test_sub() {
        let mut dd0 = AffTree::<2>::from_aff(aff!([-1., 2.] + -1.));
        dd0.compose::<false, false>(&schema::partial_ReLU(1, 0));

        let mut dd1 = AffTree::<2>::from_aff(aff!([-2., -1.] + 3.));
        dd1.compose::<false, false>(&schema::partial_ReLU(1, 0));

        let dd = dd0 - dd1;

        assert_eq!(
            dd.tree.node_value(path!(dd.tree, 1, 1)).unwrap().aff,
            aff!([0., 0.] + 0.)
        );
        assert_eq!(
            dd.tree.node_value(path!(dd.tree, 1, 0)).unwrap().aff,
            aff!([2., 1.] + -3.)
        );
        assert_eq!(
            dd.tree.node_value(path!(dd.tree, 0, 1)).unwrap().aff,
            aff!([-1., 2.] + -1.)
        );
        assert_eq!(
            dd.tree.node_value(path!(dd.tree, 0, 0)).unwrap().aff,
            aff!([1., 3.] + -4.)
        );
    }

    #[test]
    fn test_neg() {
        let mut dd0 = AffTree::<2>::from_aff(aff!([-1., 2.] + -1.));
        dd0.compose::<false, false>(&schema::partial_ReLU(1, 0));

        let dd = -dd0;

        assert_eq!(
            dd.tree.node_value(path!(dd.tree, 1)).unwrap().aff,
            aff!([0., 0.] + 0.)
        );
        assert_eq!(
            dd.tree.node_value(path!(dd.tree, 0)).unwrap().aff,
            aff!([1., -2.] + 1.)
        );
    }

    #[test]
    fn test_scalar_add() {
        let mut dd = AffTree::<2>::from_aff(aff!([[-1., 2.], [-2., -1.]] + [-1., 3.]));
        dd.compose::<false, false>(&schema::partial_ReLU(2, 0));
        dd.compose::<false, false>(&schema::partial_ReLU(2, 1));
        let f = aff!([[-1., 1.], [-2., 2.]] + [4., -4.]);

        let g = dd + &f;

        assert_eq!(g.tree.node_value(path!(g.tree, 1, 1)).unwrap().aff, f);
        assert_eq!(
            g.tree.node_value(path!(g.tree, 1, 0)).unwrap().aff,
            aff!([[-1., 1.], [-4., 1.]] + [4., -1.])
        );
        assert_eq!(
            g.tree.node_value(path!(g.tree, 0, 1)).unwrap().aff,
            aff!([[-2., 3.], [-2., 2.]] + [3., -4.])
        );
        assert_eq!(
            g.tree.node_value(path!(g.tree, 0, 0)).unwrap().aff,
            aff!([[-2., 3.], [-4., 1.]] + [3., -1.])
        );
    }

    #[test]
    fn test_scalar_sub() {
        let mut dd = AffTree::<2>::from_aff(aff!([[-2., 3.], [-4., 1.]] + [-7., 2.]));
        dd.compose::<false, false>(&schema::partial_ReLU(2, 0));
        dd.compose::<false, false>(&schema::partial_ReLU(2, 1));
        let f = aff!([[-2., 2.], [-5., 5.]] + [1., -1.]);

        let g = dd.clone() - &f;

        assert_eq!(
            g.tree.node_value(path!(g.tree, 1, 1)).unwrap().aff,
            aff!([[2., -2.], [5., -5.]] + [-1., 1.])
        );
        assert_eq!(
            g.tree.node_value(path!(g.tree, 1, 0)).unwrap().aff,
            aff!([[2., -2.], [1., -4.]] + [-1., 3.])
        );
        assert_eq!(
            g.tree.node_value(path!(g.tree, 0, 1)).unwrap().aff,
            aff!([[0., 1.], [5., -5.]] + [-8., 1.])
        );
        assert_eq!(
            g.tree.node_value(path!(g.tree, 0, 0)).unwrap().aff,
            aff!([[0., 1.], [1., -4.]] + [-8., 3.])
        );

        assert_eq!(
            g.tree.node_value(path!(g.tree, 0)).unwrap().aff,
            dd.tree.node_value(path!(dd.tree, 0)).unwrap().aff
        );
        assert_eq!(
            g.tree.node_value(path!(g.tree, 1)).unwrap().aff,
            dd.tree.node_value(path!(dd.tree, 1)).unwrap().aff
        );
    }

    #[test]
    fn test_scalar_sub_rev() {
        let mut dd = AffTree::<2>::from_aff(aff!([[-2., 3.], [-4., 1.]] + [-7., 2.]));
        dd.compose::<false, false>(&schema::partial_ReLU(2, 0));
        dd.compose::<false, false>(&schema::partial_ReLU(2, 1));

        let f = aff!([[-2., 2.], [-5., 5.]] + [1., -1.]);
        let g = &f - dd.clone();

        assert_eq!(
            g.tree.node_value(path!(g.tree, 1, 1)).unwrap().aff,
            aff!([[-2., 2.], [-5., 5.]] + [1., -1.])
        );
        assert_eq!(
            g.tree.node_value(path!(g.tree, 1, 0)).unwrap().aff,
            aff!([[-2., 2.], [-1., 4.]] + [1., -3.])
        );
        assert_eq!(
            g.tree.node_value(path!(g.tree, 0, 1)).unwrap().aff,
            aff!([[0., -1.], [-5., 5.]] + [8., -1.])
        );
        assert_eq!(
            g.tree.node_value(path!(g.tree, 0, 0)).unwrap().aff,
            aff!([[0., -1.], [-1., 4.]] + [8., -3.])
        );

        assert_eq!(
            g.tree.node_value(path!(g.tree, 0)).unwrap().aff,
            dd.tree.node_value(path!(dd.tree, 0)).unwrap().aff
        );
        assert_eq!(
            g.tree.node_value(path!(g.tree, 1)).unwrap().aff,
            dd.tree.node_value(path!(dd.tree, 1)).unwrap().aff
        );
    }
}
