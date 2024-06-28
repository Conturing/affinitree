//   Copyright 2024 affinitree developers
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

use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use ndarray::{Data, DataMut, DataOwned, OwnedRepr};
use num_traits::Float;

use super::affine::*;

// Inspired by ndarray
macro_rules! impl_ops (
    ($trt:ident, $mth:ident, $doc:expr) => (

    /// Performs elementwise
    #[doc=$doc]
    /// between self and rhs.
    ///
    /// In this variant, new memory is allocated.
    impl<S: Data<Elem = A>, A: Float, S2: Data<Elem = A>> $trt<&AffFuncBase<FunctionT, S2>> for &AffFuncBase<FunctionT, S> {
        type Output = AffFuncBase<FunctionT, OwnedRepr<A>>;

        fn $mth(self, rhs: &AffFuncBase<FunctionT, S2>) -> Self::Output {
            let mat = self.mat.$mth(&rhs.mat);
            let bias = self.bias.$mth(&rhs.bias);
            AffFuncBase::from_mats(mat, bias)
        }
    }

    /// Performs elementwise
    #[doc=$doc]
    /// between self and rhs.
    ///
    /// In this variant, self is moved and the operations is performed in place.
    impl<S: DataOwned<Elem = A> + DataMut, A: Float, S2: Data<Elem = A>> $trt<AffFuncBase<FunctionT, S2>> for AffFuncBase<FunctionT, S> {
        type Output = AffFuncBase<FunctionT, S>;

        fn $mth(self, rhs: AffFuncBase<FunctionT, S2>) -> Self::Output {
            self.$mth(&rhs)
        }
    }

    /// Performs elementwise
    #[doc=$doc]
    /// between self and rhs.
    ///
    /// In this variant, self is moved and the operations is performed in place.
    impl<S: DataOwned<Elem = A> + DataMut, A: Float, S2: Data<Elem = A>> $trt<&AffFuncBase<FunctionT, S2>> for AffFuncBase<FunctionT, S> {
        type Output = AffFuncBase<FunctionT, S>;

        fn $mth(self, rhs: &AffFuncBase<FunctionT, S2>) -> Self::Output {
            let mat = self.mat.$mth(&rhs.mat);
            let bias = self.bias.$mth(&rhs.bias);
            AffFuncBase::from_mats(mat, bias)
        }
    }

    )
);

impl_ops!(Add, add, "addition");
impl_ops!(Sub, sub, "subtraction");
impl_ops!(Mul, mul, "multiplication");
impl_ops!(Div, div, "division");
impl_ops!(Rem, rem, "remainder");

impl<S: DataOwned<Elem = A> + DataMut, A: Float> Neg for AffFuncBase<FunctionT, S> {
    type Output = AffFuncBase<FunctionT, S>;

    fn neg(self) -> Self::Output {
        let mat = self.mat.neg();
        let bias = self.bias.neg();
        AffFuncBase::from_mats(mat, bias)
    }
}

impl<'a, S, A> Neg for &'a AffFuncBase<FunctionT, S>
where
    S: Data<Elem = A>,
    A: Float,
    &'a A: 'a + Neg<Output = A>,
{
    type Output = AffFuncBase<FunctionT, OwnedRepr<A>>;

    fn neg(self) -> Self::Output {
        let mat = self.mat.neg();
        let bias = self.bias.neg();
        AffFuncBase::from_mats(mat, bias)
    }
}

#[cfg(test)]
mod tests {
    use approx::*;
    use ndarray::arr1;

    use super::*;
    use crate::aff;

    #[test]
    pub fn test_add() {
        let f1 = AffFunc::unit(6, 4);
        let f2 = AffFunc::constant(6, 10.);
        let g1 = f1.clone() + f2.clone();
        let g2 = f1.clone() + f2.view();
        let g3 = f1.clone() + &f2;
        let g4 = f1.clone() + &f2.view();
        let g5 = &f1 + &f2.view();
        let g6 = &f1.view() + &f2.view();

        drop(f1);

        assert_eq!(
            g1.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[310.])
        );
        assert_eq!(
            g2.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[310.])
        );
        assert_eq!(
            g3.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[310.])
        );
        assert_eq!(
            g4.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[310.])
        );
        assert_eq!(
            g5.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[310.])
        );
        assert_eq!(
            g6.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[310.])
        );
    }

    #[test]
    pub fn test_sub() {
        let f1 = aff!([[-1., 3.], [4., -6.], [0.3, 0.2]] + [2., 0., -2.]);
        let f2 = aff!([[2., -1.], [2., 3.], [3., 2.]] + [3., 5., 7.]);
        let diff = aff!([[-3., 4.], [2., -9.], [-2.7, -1.8]] + [-1, -5., -9.]);
        let g1 = f1.clone() - f2.clone();
        let g2 = f1.clone() - f2.view();
        let g3 = f1.clone() - &f2;
        let g4 = f1.clone() - &f2.view();
        let g5 = &f1 - &f2.view();
        let g6 = &f1.view() - &f2.view();

        assert_relative_eq!(g1, diff);
        assert_relative_eq!(g2, diff);
        assert_relative_eq!(g3, diff);
        assert_relative_eq!(g4, diff);
        assert_relative_eq!(g5, diff);
        assert_relative_eq!(g6, diff);
    }

    #[test]
    pub fn test_neg() {
        let f1 = AffFunc::unit(6, 4);
        let g1 = -f1.clone();
        let g2 = -&f1;
        let g3 = -&f1.view();

        drop(f1);

        assert_eq!(
            g1.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[-300.])
        );
        assert_eq!(
            g2.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[-300.])
        );
        assert_eq!(
            g3.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[-300.])
        );
    }
}
