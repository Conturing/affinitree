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

//! Structs to store linear functions and polytopes

use core::fmt;
use std::fmt::Display;
use std::iter::zip;
use std::marker::PhantomData;
use std::ops::{self, Add, Mul};

use itertools::enumerate;
use ndarray::{self, arr1, Axis};
use ndarray::{
    concatenate, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2, RawDataClone,
};

use crate::linalg::vis::{write_aff, write_polytope};

// wrap ndarray data types
pub type Owned = ndarray::OwnedRepr<f64>;
pub type ViewRepr<'a> = ndarray::ViewRepr<&'a f64>;

pub trait Ownership: Data<Elem = f64> + RawDataClone<Elem = f64> {}

impl Ownership for Owned {}
impl<'a> Ownership for ViewRepr<'a> {}

#[derive(Clone, Default, Debug)]
pub struct FunctionT;

#[derive(Clone, Default, Debug)]
pub struct PolytopeT;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AffFuncBase<I, D: Ownership> {
    pub mat: ArrayBase<D, Ix2>,
    pub bias: ArrayBase<D, Ix1>,
    pub _phantom: PhantomData<I>,
}

pub type AffFunc = AffFuncBase<FunctionT, Owned>;
pub type AffFuncView<'a> = AffFuncBase<FunctionT, ViewRepr<'a>>;
pub type Polytope = AffFuncBase<PolytopeT, Owned>;
pub type PolytopeView<'a> = AffFuncBase<PolytopeT, ViewRepr<'a>>;

// General constructor
impl<I, D: Ownership> AffFuncBase<I, D> {
    /// Create a new instance of an affine combination consisting of a matrix mat: R^{m x n} and a vector bias: R^m.
    ///
    /// When interpreted as a function it is equivalent to f(x) = mat @ x + bias.
    /// When interpreted as a polytope it describes the set P = {x | mat @ x <= bias}
    #[inline(always)]
    pub fn from_mats(mat: ArrayBase<D, Ix2>, bias: ArrayBase<D, Ix1>) -> AffFuncBase<I, D> {
        assert_eq!(
            mat.len_of(Axis(0)),
            bias.len_of(Axis(0)),
            "Dimensions mismatch of matrix and bias: {} x {} and {}",
            mat.len_of(Axis(0)),
            mat.len_of(Axis(1)),
            bias.len_of(Axis(0))
        );
        debug_assert!(
            mat.iter().all(|x| x.is_normal() || *x == 0f64),
            "Non-normal floats are deprecated"
        );

        AffFuncBase {
            mat: mat,
            bias: bias,
            _phantom: PhantomData,
        }
    }

    #[cfg(test)]
    pub fn random_affine(dim_out: usize, dim_in: usize) -> AffFuncBase<I, Owned> {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let mut mat = Array2::zeros((dim_out, dim_in));
        let mut bias = Array1::zeros(dim_out);
        for i in 0..dim_out {
            for j in 0..dim_in {
                mat[[i, j]] = rng.gen()
            }
            bias[i] = rng.gen();
        }

        AffFuncBase::<I, Owned> {
            mat: mat,
            bias: bias,
            _phantom: PhantomData,
        }
    }
}

// Function specific constructors
impl<D: Ownership> AffFuncBase<FunctionT, D> {
    /// Returns the affine function that implements the identity function f(x)=x.
    #[inline(always)]
    pub fn identity(dim: usize) -> AffFunc {
        AffFunc {
            mat: Array2::from_diag_elem(dim, 1f64),
            bias: Array1::zeros(dim),
            _phantom: PhantomData,
        }
    }

    /// Returns the affine function that always returns the specified value.
    #[inline(always)]
    pub fn constant(dim: usize, value: f64) -> AffFunc {
        let bias = arr1(&[value]);
        AffFunc {
            mat: Array2::zeros((1, dim)),
            bias: bias,
            _phantom: PhantomData,
        }
    }

    /// Returns the affine function that returns the element in the given column.
    #[inline(always)]
    pub fn unit(dim: usize, column: usize) -> AffFunc {
        let mut mat = Array2::zeros((1, dim));
        mat[[0, column]] = 1.;
        AffFunc {
            mat: mat,
            bias: Array1::zeros(1),
            _phantom: PhantomData,
        }
    }

    /// Returns the affine function that sets the element at index to zero and leaves all other elements unchanged.
    #[inline(always)]
    pub fn zero_idx(dim: usize, index: usize) -> AffFunc {
        let mut mat = Array2::eye(dim);
        mat[[index, index]] = 0.;
        AffFunc {
            mat: mat,
            bias: Array1::zeros(dim),
            _phantom: PhantomData,
        }
    }

    /// Returns the affine function R^dim -> R that sums all inputs.
    #[inline(always)]
    pub fn sum(dim: usize) -> AffFunc {
        AffFunc {
            mat: Array2::ones((1, dim)),
            bias: Array1::zeros(1),
            _phantom: PhantomData,
        }
    }

    /// Returns the affine function that subtracts the right index from the left index.
    #[inline(always)]
    pub fn subtraction(dim: usize, left: usize, right: usize) -> AffFunc {
        let mut matrix = Array2::zeros((1, dim));
        matrix[[0, left]] = 1.;
        matrix[[0, right]] = -1.;
        let bias = Array1::zeros(1);
        AffFunc {
            mat: matrix,
            bias: bias,
            _phantom: PhantomData,
        }
    }
}

// Polytope specific constructors
impl<D: Ownership> AffFuncBase<PolytopeT, D> {
    #[inline(always)]
    pub fn unrestricted(dim: usize) -> Polytope {
        Polytope::from_mats(Array2::zeros((1, dim)), Array1::ones(1))
    }

    /// Create a polytope from a set of halfspaces described by normal vectors and points in the plane (hesse normal form).
    #[inline(always)]
    pub fn from_normal(normal_vectors: Array2<f64>, distance_vectors: Array2<f64>) -> Polytope {
        let bias = (&normal_vectors).mul(&distance_vectors).sum_axis(Axis(1));
        Polytope::from_mats(-normal_vectors, -bias)
    }

    /// Constructs a dim dimensional hypercube centered at the origin.
    /// In each dimension two hyperplanes are placed with distance +/- radius from the origin.
    pub fn hypercube(dim: usize, radius: f64) -> Polytope {
        let mat: Array2<f64> = concatenate![Axis(0), Array2::eye(dim), -Array2::eye(dim)];
        let bias: Array1<f64> = radius * Array1::<f64>::ones(2 * dim);
        Polytope::from_mats(mat, bias)
    }

    pub fn hyperrectangle(dim: usize, intervals: &[(f64, f64)]) -> Polytope {
        let mat: Array2<f64> = concatenate![Axis(0), Array2::eye(dim), -Array2::eye(dim)];
        let mut bias: Array1<f64> = Array1::<f64>::zeros(2 * dim);
        for idx in 0..dim {
            bias[idx] = intervals[idx].1;
            bias[idx + dim] = -intervals[idx].0;
        }
        Polytope::from_mats(mat, bias)
    }
}

// General methods
impl<I, D: Ownership> AffFuncBase<I, D> {
    /// Returns the dimension of the input space.
    #[inline(always)]
    pub fn indim(&self) -> usize {
        self.mat.shape()[1]
    }

    /// Returns the dimension of the image space.
    #[inline(always)]
    pub fn outdim(&self) -> usize {
        self.mat.shape()[0]
    }

    #[inline(always)]
    pub fn get_matrix(&self) -> ArrayView2<f64> {
        self.mat.view()
    }

    #[inline(always)]
    pub fn get_bias(&self) -> ArrayView1<f64> {
        self.bias.view()
    }

    pub fn row(&self, row: usize) -> AffFuncBase<I, Owned> {
        assert!(
            row < self.outdim(),
            "Row outside range: got {} but only {} rows exist",
            row,
            self.outdim()
        );
        AffFuncBase::<I, Owned> {
            mat: self.mat.row(row).to_owned().insert_axis(Axis(0)),
            bias: Array1::from_elem(1, self.bias[row]),
            _phantom: PhantomData,
        }
    }

    /// Iterate over the rows of the matrix and bias.
    pub fn row_iter(&self) -> impl Iterator<Item = AffFuncBase<I, Owned>> + '_ {
        zip(self.mat.outer_iter(), self.bias.outer_iter()).map(|(row, bias)| {
            AffFuncBase::<I, Owned>::from_mats(
                row.to_owned().insert_axis(Axis(0)),
                bias.to_owned().insert_axis(Axis(0)),
            )
        })
    }
}

impl<D: Ownership> AffFuncBase<PolytopeT, D> {
    pub fn n_constraints(&self) -> usize {
        self.mat.shape()[0]
    }
}

impl<I> AffFuncBase<I, Owned> {
    pub fn normalize(mut self) -> AffFuncBase<I, Owned> {
        for (mut row, mut bias) in zip(self.mat.outer_iter_mut(), self.bias.outer_iter_mut()) {
            let norm: f64 = row.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
            row.map_inplace(|x| *x /= norm);
            bias /= norm;
        }
        self
    }
}

impl<D: Ownership> AffFuncBase<FunctionT, D> {
    /// Evaluate this function under the given input.
    /// Mathematically, this corresponds to calculating mat @ input + bias
    pub fn apply<S: Data<Elem = f64>>(&self, input: &ArrayBase<S, Ix1>) -> Array1<f64> {
        self.mat.dot(input) + &self.bias
    }

    /// Evaluate the transposed of this function under the given input.
    /// For orthogonal functions this corresponds to the inverse.
    /// Mathematically, this corresponds to calculating mat.T @ (input - bias)
    pub fn apply_transpose<S: Data<Elem = f64>>(&self, input: &ArrayBase<S, Ix1>) -> Array1<f64> {
        self.mat.t().dot(&(input - &self.bias))
    }
}

impl<D: Ownership> AffFuncBase<PolytopeT, D> {
    /// *Notice*: Distance is not normalized. Use with caution.
    #[inline]
    pub fn distance_raw(&self, point: &Array1<f64>) -> Array1<f64> {
        &self.bias - self.mat.dot(point)
    }

    /// Return the (normalized) distance from point to all hyperplanes of this polytope.
    /// Precisely, it returns a vector where each element is the distance from the given point to the hyperplane in order.
    /// Distance is positive if the point is inside the halfspace of that inequality and negative otherwise.
    /// Returns f64::INFINITY if the corresponding halfspaces includes all points.
    pub fn distance(&self, point: &Array1<f64>) -> Array1<f64> {
        let mut raw_dist = self.distance_raw(point);
        for (row, mut dist) in zip(self.mat.outer_iter(), raw_dist.outer_iter_mut()) {
            let norm: f64 = row.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
            dist /= norm;
        }
        raw_dist
    }

    #[inline]
    pub fn contains(&self, point: &Array1<f64>) -> bool {
        self.distance_raw(point).into_iter().all(|x| x >= 0f64)
    }
}

// Combine existing AffFuncs to new ones
impl<D: Ownership> AffFuncBase<FunctionT, D> {
    /// Compose self with other. The resulting function will have the same effect as first applying other and then self.
    ///
    /// # Example
    ///
    /// ``` rust
    /// use affinitree::linalg::affine::AffFunc;
    /// use ndarray::arr1;
    ///
    /// let f1 = AffFunc::sum(6);
    /// let f2 = AffFunc::zero_idx(6, 5);
    ///
    /// assert_eq!(
    ///     f1.compose(&f2).apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
    ///     f1.apply(&f2.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])))
    /// );
    /// ```
    pub fn compose(&self, other: &AffFuncBase<FunctionT, D>) -> AffFunc {
        assert!(
            self.indim() == other.outdim(),
            "Dimensions of functions mismatch for composition: {} to {}",
            self.indim(),
            other.outdim()
        );
        AffFunc {
            mat: self.mat.dot(&other.mat),
            bias: self.apply(&other.bias),
            _phantom: PhantomData,
        }
    }

    pub fn stack(&self, other: &AffFuncBase<FunctionT, D>) -> AffFunc {
        assert!(
            self.indim() == other.indim(),
            "Dimensions of functions mismatch for stacking: {} to {}",
            self.indim(),
            other.indim()
        );
        AffFunc {
            mat: concatenate![Axis(0), self.mat, other.mat],
            bias: concatenate![Axis(0), self.bias, other.bias],
            _phantom: PhantomData,
        }
    }

    pub fn add(self, other: &AffFuncBase<FunctionT, D>) -> AffFunc {
        assert!(
            self.indim() == other.indim() && self.outdim() == other.outdim(),
            "Dimensions of functions mismatch for adding: {} -> {} and {} -> {}",
            self.indim(),
            self.outdim(),
            other.indim(),
            other.outdim()
        );
        AffFunc {
            mat: self.mat.add(&other.mat),
            bias: self.bias.add(&other.bias),
            _phantom: PhantomData,
        }
    }
}

impl AffFunc {
    pub fn negate(self) -> AffFunc {
        AffFunc {
            mat: -self.mat.clone(),
            bias: -self.bias,
            _phantom: PhantomData,
        }
    }
}

// Combine existing Polytopes to new ones
impl<D: Ownership> AffFuncBase<PolytopeT, D> {
    pub fn translate(&self, direction: &Array1<f64>) -> Polytope {
        Polytope::from_mats(self.mat.to_owned(), &self.bias + self.mat.dot(direction))
    }

    pub fn intersection(&self, other: &AffFuncBase<PolytopeT, D>) -> Polytope {
        assert!(self.indim() == other.indim());

        let mat = concatenate![Axis(0), self.mat, other.mat];
        let bias = concatenate![Axis(0), self.bias, other.bias];

        Polytope::from_mats(mat, bias)
    }

    /// Constructs a new polytope as the intersection of polys.
    /// That is, the resulting polytope contains all points that are contained in each polytope of polys.
    pub fn intersection_n(dim: usize, polys: &[AffFuncBase<PolytopeT, D>]) -> Polytope {
        if polys.is_empty() {
            return Polytope::unrestricted(dim);
        }

        let mat_view: Vec<ArrayView2<f64>> = polys.iter().map(|poly| poly.mat.view()).collect();
        let mat_concat = ndarray::concatenate(Axis(0), mat_view.as_slice());
        let mat = match mat_concat {
            Ok(result) => result,
            Err(error) => panic!("Error when concatenating matrices, probably caused by a mismatch in dimensions: {:?}", error),
        };

        let bias_view: Vec<ArrayView1<f64>> = polys.iter().map(|poly| poly.bias.view()).collect();
        let bias_concat = ndarray::concatenate(Axis(0), bias_view.as_slice());
        let bias = match bias_concat {
            Ok(result) => result,
            Err(error) => panic!(
                "Error when concatenating bias, probably caused by a mismatch in dimensions: {:?}",
                error
            ),
        };

        Polytope::from_mats(mat, bias)
    }
}

// Switch between ownership
impl<I> AffFuncBase<I, Owned> {
    #[inline]
    pub fn view<'a>(&'a self) -> AffFuncBase<I, ViewRepr<'a>> {
        AffFuncBase::<I, ViewRepr<'a>> {
            mat: self.mat.view(),
            bias: self.bias.view(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, I> AffFuncBase<I, ViewRepr<'a>> {
    #[inline]
    pub fn to_owned(&self) -> AffFuncBase<I, Owned> {
        AffFuncBase::<I, Owned> {
            mat: self.mat.to_owned(),
            bias: self.bias.to_owned(),
            _phantom: PhantomData,
        }
    }
}

// Switch between types
impl<D: Ownership> AffFuncBase<FunctionT, D> {
    #[inline]
    pub fn as_polytope(&self) -> AffFuncBase<PolytopeT, D> {
        AffFuncBase::<PolytopeT, D> {
            mat: self.mat.clone(),
            bias: self.bias.clone(),
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PolyRepr {
    MatrixLeqBias,
    MatrixBiasLeqZero,
    MatrixGeqBias,
    MatrixBiasGeqZero,
}

impl<D: Ownership> AffFuncBase<PolytopeT, D> {
    #[inline]
    pub fn as_function(&self) -> AffFuncBase<FunctionT, D> {
        AffFuncBase::<FunctionT, D> {
            mat: self.mat.clone(),
            bias: self.bias.clone(),
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn new(aff: AffFuncBase<FunctionT, D>) -> AffFuncBase<PolytopeT, D> {
        AffFuncBase::<PolytopeT, D> {
            mat: aff.mat,
            bias: aff.bias,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn convert_to(&self, repr: PolyRepr) -> AffFuncBase<FunctionT, Owned> {
        match repr {
            PolyRepr::MatrixLeqBias => AffFuncBase::<FunctionT, Owned> {
                mat: self.mat.to_owned(),
                bias: self.bias.to_owned(),
                _phantom: PhantomData,
            },
            PolyRepr::MatrixBiasLeqZero => AffFuncBase::<FunctionT, Owned> {
                mat: self.mat.to_owned(),
                bias: -self.bias.to_owned(),
                _phantom: PhantomData,
            },
            PolyRepr::MatrixGeqBias => AffFuncBase::<FunctionT, Owned> {
                mat: -self.mat.to_owned(),
                bias: -self.bias.to_owned(),
                _phantom: PhantomData,
            },
            PolyRepr::MatrixBiasGeqZero => AffFuncBase::<FunctionT, Owned> {
                mat: -self.mat.to_owned(),
                bias: self.bias.to_owned(),
                _phantom: PhantomData,
            },
        }
    }
}

impl AffFunc {
    pub fn reset_row(&mut self, row: usize) {
        self.mat.row_mut(row).fill(0.);
        self.bias[row] = 0.;
    }
}

impl<D: Ownership> AffFuncBase<PolytopeT, D> {
    /// Returns the linear program that encodes the chebyshev center of this polytope.
    /// That is, the resulting polytope contains all points that are
    ///
    /// 1. contained in this polytope
    /// 2. whose last dimension is less than or equal to the minimal distance to the hyperplanes of this polytope
    ///
    /// The returned array encodes the coefficients of the cost function.
    /// Minimizing this function over the region of the returned polytope gives the center point and radius of the largest enclosed sphere inside this polytope.
    pub fn chebyshev_center(&self) -> (Polytope, Array1<f64>) {
        // distance to each hyperplane
        let mut norm = Array2::<f64>::zeros((self.mat.len_of(Axis(0)), 1));

        for (idx, row) in enumerate(self.mat.outer_iter()) {
            norm[[idx, 0]] = row.map(|x: &f64| x.powi(2)).sum().sqrt();
        }
        let amod = concatenate![Axis(1), self.mat, norm];

        // radius must be positive
        let mut rad = Array2::<f64>::zeros((1, amod.len_of(Axis(1))));
        rad[[0, amod.len_of(Axis(1)) - 1]] = -1.;
        let amod = concatenate![Axis(0), amod, rad];
        let bmod = concatenate![Axis(0), self.bias.clone(), Array1::<f64>::zeros(1)];

        (
            Polytope::from_mats(amod, bmod),
            Array1::from_iter(rad.iter().cloned()),
        )
    }
}

impl<I, D> core::cmp::PartialEq for AffFuncBase<I, D>
where
    D: Ownership,
{
    fn eq(&self, other: &Self) -> bool {
        let (
            AffFuncBase {
                mat: mat_self,
                bias: bias_self,
                _phantom: _,
            },
            AffFuncBase {
                mat: mat_other,
                bias: bias_other,
                _phantom: _,
            },
        ) = (self, other);
        mat_self.eq(mat_other) && bias_self.eq(bias_other)
    }
}

impl Display for AffFunc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write_aff(f, self, true)
    }
}

impl Display for Polytope {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write_polytope(f, self, true)
    }
}

impl ops::Add<&AffFunc> for AffFunc {
    type Output = AffFunc;

    fn add(self, other: &AffFunc) -> AffFunc {
        self.add(other)
    }
}

impl ops::Sub<AffFunc> for AffFunc {
    type Output = AffFunc;

    fn sub(self, other: AffFunc) -> AffFunc {
        self.add(&other.negate())
    }
}

impl ops::Neg for AffFunc {
    type Output = AffFunc;

    fn neg(self) -> AffFunc {
        self.negate()
    }
}

// see also ndarray's array! macro
#[macro_export]
macro_rules! aff {
    ([ $([$($x:expr),* $(,)*]),+ $(,)* ] + [ $($y:expr),* $(,)* ]) => {{
        $crate::linalg::affine::AffFunc::from_mats(
           ndarray::Array2::<f64>::from(vec![$( [ $( ($x as f64), )* ], )*]),
           ndarray::Array1::<f64>::from(vec![$($y as f64,)*])
        )
    }};
    ([ $($x:expr),* $(,)* ] + $y:expr) => {{
        $crate::linalg::affine::AffFunc::from_mats(
           ndarray::Array2::<f64>::from(vec![ [ $( ($x as f64), )* ]]),
           ndarray::Array1::<f64>::from(vec![$y as f64])
        )
    }};
}

#[macro_export]
macro_rules! poly {
    ([ $([$($x:expr),* $(,)*]),+ $(,)* ] < [ $($y:expr),* $(,)* ]) => {{
        $crate::linalg::affine::Polytope::from_mats(
           ndarray::Array2::<f64>::from(vec![$( [ $( ($x as f64), )* ], )*]),
           ndarray::Array1::<f64>::from(vec![$($y as f64,)*])
        )
    }};
    ([ $([$($x:expr),* $(,)*]),+ $(,)* ] + [ $($y:expr),* $(,)* ] < 0) => {{
        $crate::linalg::affine::Polytope::from_mats(
           ndarray::Array2::<f64>::from(vec![$( [ $( ($x as f64), )* ], )*]),
           ndarray::Array1::<f64>::from(vec![$(-$y as f64,)*])
        )
    }};
    ([ $([$($x:expr),* $(,)*]),+ $(,)* ] > [ $($y:expr),* $(,)* ]) => {{
        $crate::linalg::affine::Polytope::from_mats(
           ndarray::Array2::<f64>::from(vec![$( [ $( (-$x as f64), )* ], )*]),
           ndarray::Array1::<f64>::from(vec![$(-$y as f64,)*])
        )
    }};
    ([ $([$($x:expr),* $(,)*]),+ $(,)* ] + [ $($y:expr),* $(,)* ] > 0) => {{
        $crate::linalg::affine::Polytope::from_mats(
           ndarray::Array2::<f64>::from(vec![$( [ $( (-$x as f64), )* ], )*]),
           ndarray::Array1::<f64>::from(vec![$($y as f64,)*])
        )
    }};
}

#[cfg(test)]
mod tests {
    use crate::linalg::affine::{AffFunc, Polytope};

    use itertools::Itertools;
    use ndarray::{arr1, arr2, array, s, Array2};

    use approx::assert_relative_eq;

    fn init_logger() {
        // minilp has a bug if logging is enabled
        // match fast_log::init(Config::new().console().chan_len(Some(100000))) {
        //     Ok(_) => (),
        //     Err(err) => println!("Error occurred while configuring logger: {:?}", err),
        // }
    }

    /* AffFunc Tests */

    #[test]
    pub fn test_from_mats() {
        let mat = Array2::from_diag(&arr1(&[1., 7., -2., 1e-4, 0.3]));
        let bias = arr1(&[0.1, -7., 1e-2, 1e+4, 0.3]);
        let f = AffFunc::from_mats(mat.clone(), bias.clone());

        assert_eq!(f.mat, mat);
        assert_eq!(f.bias, bias);
    }

    #[test]
    #[should_panic]
    pub fn test_from_mats_nan() {
        let mat = Array2::from_diag(&arr1(&[1., 7., f64::NAN, 1e-4, 0.3]));
        let bias = arr1(&[0.1, -7., 1e-2, 1e+4, 0.3]);
        AffFunc::from_mats(mat, bias);
    }

    #[test]
    pub fn test_identity() {
        let f = AffFunc::identity(6);

        assert_eq!(
            f.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[0.3, -0.2, 0., -20., 300., -4000.])
        );
    }

    #[test]
    pub fn test_constant() {
        let f = AffFunc::constant(6, 10.);

        assert_eq!(
            f.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[10.])
        );
    }

    #[test]
    pub fn test_unit() {
        let f = AffFunc::unit(6, 1);

        assert_eq!(
            f.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[-0.2])
        );
    }

    #[test]
    pub fn test_zero() {
        let f = AffFunc::zero_idx(6, 1);

        assert_eq!(
            f.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[0.3, 0., 0., -20., 300., -4000.])
        );
    }

    #[test]
    pub fn test_sum() {
        let f = AffFunc::sum(6);

        assert_eq!(
            f.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[-3719.9])
        );
    }

    #[test]
    pub fn test_subtraction() {
        let f = AffFunc::subtraction(6, 1, 4);

        assert_eq!(
            f.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[-300.2])
        );
    }

    #[test]
    pub fn test_dim() {
        let f = aff!([[1, 2, 5, 7], [-2, -9, 7, 8]] + [1, -1]);

        assert_eq!(f.indim(), 4);
        assert_eq!(f.outdim(), 2);
    }

    #[test]
    pub fn test_row() {
        let f = aff!([[1, 2, 5, 7], [-2, -9, 7, 8]] + [1, -1]);

        assert_eq!(f.row(0), aff!([[1, 2, 5, 7]] + [1]));
        assert_eq!(f.row(1), aff!([[-2, -9, 7, 8]] + [-1]));
    }

    #[test]
    pub fn test_row_iter() {
        let f = AffFunc::from_mats(Array2::eye(4), arr1(&[1., 2., 3., 4.]));

        let fs = f.row_iter().collect_vec();

        assert_eq!(
            fs,
            vec![
                AffFunc::from_mats(arr2(&[[1., 0., 0., 0.]]), arr1(&[1.])),
                AffFunc::from_mats(arr2(&[[0., 1., 0., 0.]]), arr1(&[2.])),
                AffFunc::from_mats(arr2(&[[0., 0., 1., 0.]]), arr1(&[3.])),
                AffFunc::from_mats(arr2(&[[0., 0., 0., 1.]]), arr1(&[4.]))
            ]
        )
    }

    #[test]
    pub fn test_normalize() {
        let f = aff!([[1, 0, -1], [0, 1, 0]] + [2, 5]);
        let sqrt2 = 2.0f64.sqrt();

        assert_eq!(
            f.normalize(),
            aff!([[1.0 / sqrt2, 0, -1.0 / sqrt2], [0, 1, 0]] + [2. / sqrt2, 5])
        );
    }

    #[test]
    pub fn test_compose() {
        let f1 = AffFunc::zero_idx(6, 5);
        let f2 = AffFunc::sum(6);
        let f = f2.compose(&f1);

        assert_eq!(
            f.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[280.1])
        );
    }

    #[test]
    pub fn test_stack() {
        let f1 = AffFunc::unit(6, 5);
        let f2 = AffFunc::unit(6, 1);
        let f = f1.stack(&f2);

        assert_eq!(
            f.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[-4000., -0.2])
        );
    }

    #[test]
    pub fn test_add() {
        let f1 = AffFunc::unit(6, 4);
        let f2 = AffFunc::constant(6, 10.);
        let f = f1 + &f2;

        assert_eq!(
            f.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[310.])
        );
    }

    #[test]
    pub fn test_sub() {
        let f1 = AffFunc::unit(6, 3);
        let f2 = AffFunc::unit(6, 0);
        let f = f1 - f2;

        assert_eq!(
            f.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[-20.3])
        );
    }

    #[test]
    pub fn test_neg() {
        let f1 = AffFunc::unit(6, 4);
        let f = -f1;

        assert_eq!(
            f.apply(&arr1(&[0.3, -0.2, 0., -20., 300., -4000.])),
            arr1(&[-300.])
        );
    }

    #[test]
    pub fn test_aff_macro() {
        assert_eq!(
            aff!([1, 0, 1] + 2),
            AffFunc::from_mats(arr2(&[[1., 0., 1.]]), arr1(&[2.]))
        );
        assert_eq!(
            aff!([[1, 0, 1]] + [2]),
            AffFunc::from_mats(arr2(&[[1., 0., 1.]]), arr1(&[2.]))
        );
        assert_eq!(
            aff!([[-2, 0, 3], [4, -5, 9.3]] + [2, -1]),
            AffFunc::from_mats(arr2(&[[-2., 0., 3.], [4., -5., 9.3]]), arr1(&[2., -1.]))
        );
    }

    #[test]
    pub fn test_affine() {
        let a = AffFunc::random_affine(4, 3);
        let b = AffFunc::identity(4);
        let c = b.compose(&a);
        assert!(c.indim() == 3);
        assert!(c.outdim() == 4);
    }

    /* Polytope Tests */

    #[test]
    pub fn test_unrestricted() {
        let poly = Polytope::unrestricted(4);

        assert_eq!(poly.indim(), 4);
    }

    #[test]
    pub fn test_hyperrectangle() {
        let ival = [(1., 2.), (-1., 1.)];

        let poly1 = Polytope::hyperrectangle(2, &ival);
        let poly2 = Polytope::from_mats(
            array![[1., 0.], [0., 1.], [-1., 0.], [0., -1.]],
            array![2., 1., -1., 1.],
        );

        assert_eq!(poly1, poly2);
    }

    #[test]
    pub fn test_poly_normal_constructor() {
        init_logger();

        let normals = arr2(&[
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
        ]);

        let points = arr2(&[
            [0.0, 0.0],
            [0.0, 0.0],
            [4.0, 4.0],
            [4.0, 4.0],
            [1.0, 0.0],
            [3.0, 0.0],
        ]);

        let a = Polytope::from_normal(normals.to_owned(), points);

        let b = Polytope::from_mats(-normals, arr1(&[-0.0, -0.0, 4.0, 4.0, -1.0, 3.0]));

        assert_eq!(a, b);
    }

    #[test]
    pub fn test_distance() {
        let poly = poly!([[2., 0., 0.]] < [2.]);

        assert_eq!(poly.distance(&arr1(&[1., 0., 0.])), arr1(&[0.]));
        assert_eq!(poly.distance(&arr1(&[-7., 0., 0.])), arr1(&[8.]));
        assert_eq!(poly.distance(&arr1(&[7., 0., 0.])), arr1(&[-6.]));
    }

    #[test]
    pub fn test_distance_unrestricted() {
        let poly = Polytope::unrestricted(4);

        assert_eq!(
            poly.distance(&arr1(&[0., 1., 0., 0.])),
            arr1(&[f64::INFINITY])
        );
    }

    #[test]
    pub fn test_contains() {
        init_logger();

        let weights = arr2(&[
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-2.0, -1.0],
        ]);
        let bias = arr1(&[-1.0, -1.0, 4.0, 4.0, -3.0, 6.0, -2.0, -8.0]);
        let poly = Polytope::from_mats(weights, bias);

        // extreme points
        assert!(poly.contains(&arr1(&[3.5, 1.0])));
        assert!(poly.contains(&arr1(&[4.0, 1.0])));
        assert!(poly.contains(&arr1(&[4.0, 2.0])));
        assert!(poly.contains(&arr1(&[3.34, 1.33])));

        // inner point
        assert!(poly.contains(&arr1(&[3.75, 1.25])));

        // outer points
        assert!(!poly.contains(&arr1(&[3.25, 1.15])));
        assert!(!poly.contains(&arr1(&[3.5, 2.0])));
        assert!(!poly.contains(&arr1(&[4.5, 1.5])));
        assert!(!poly.contains(&arr1(&[4.0, 0.0])));
    }

    #[test]
    pub fn test_translate() {
        let poly = Polytope::hypercube(3, 0.5);

        assert!(!poly.contains(&arr1(&[4.0, 7.0, 0.0])));

        let poly = poly.translate(&arr1(&[4.0, 6.8, 0.4]));

        assert!(poly.contains(&arr1(&[4.0, 7.0, 0.0])));

        // depends on the implementation of hypercube
        assert_relative_eq!(
            poly.distance(&arr1(&[4.0, 7.0, 0.0])),
            arr1(&[0.5, 0.3, 0.9, 0.5, 0.7, 0.1])
        );
    }

    #[test]
    pub fn test_intersection() {
        let poly1 = Polytope::hypercube(2, 0.5).translate(&arr1(&[-0.2, -0.2]));
        let poly2 = Polytope::hypercube(2, 0.5).translate(&arr1(&[0.2, 0.2]));

        let poly = poly1.intersection(&poly2);

        assert!(poly.contains(&arr1(&[0.2, 0.2])));
        assert!(poly.contains(&arr1(&[0.2, -0.2])));
        assert!(poly.contains(&arr1(&[-0.2, 0.2])));
        assert!(poly.contains(&arr1(&[-0.2, -0.2])));

        assert!(!poly.contains(&arr1(&[0.4, 0.4])));
        assert!(!poly.contains(&arr1(&[-0.4, -0.4])));
    }

    #[test]
    pub fn test_chebyshev_box() {
        let poly = Polytope::from_mats(
            array![[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
            array![1., 1., 1., 1.],
        );

        let (p, _) = poly.chebyshev_center();

        assert!(p.contains(&arr1(&[0.0, 0.0, 0.9])));
        assert!(!p.contains(&arr1(&[0.0, 0.0, 1.1])));
    }

    #[test]
    pub fn test_chebyshev_box_2() {
        let poly = Polytope::from_mats(
            array![[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
            array![2., 1., 1., 1.],
        );

        let (p, _) = poly.chebyshev_center();

        assert!(p.contains(&arr1(&[0.1, 0.0, 0.9])));
        assert!(p.contains(&arr1(&[0.9, 0.0, 0.9])));
    }

    #[test]
    pub fn test_chebyshev_triangle() {
        let poly = Polytope::from_mats(array![[1., 1.], [-1., 1.], [0., -1.]], array![0., 0., 2.4]);

        let (p, _) = poly.chebyshev_center();

        assert!(p.contains(&arr1(&[0., -1.414, 0.9])));
    }

    #[test]
    pub fn test_intersection_0() {
        init_logger();

        let weights = arr2(&[
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-2.0, -1.0],
        ]);
        let bias = arr1(&[-1.0, -1.0, 4.0, 4.0, -3.0, 6.0, -2.0, -11.0]);
        let poly = Polytope::from_mats(weights.clone(), bias.clone());

        let poly0 = Polytope::from_mats(
            weights.slice(s![..4, ..]).to_owned(),
            bias.slice(s![..4]).to_owned(),
        );
        let poly1 = Polytope::from_mats(
            weights.slice(s![4.., ..]).to_owned(),
            bias.slice(s![4..]).to_owned(),
        );

        assert_eq!(poly, poly0.intersection(&poly1));
    }

    #[test]
    pub fn test_intersection_1() {
        init_logger();

        let weights = arr2(&[
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-2.0, -1.0],
        ]);
        let bias = arr1(&[-1.0, -1.0, 4.0, 4.0, -3.0, 6.0, -2.0, -11.0]);
        let poly = Polytope::from_mats(weights.clone(), bias.clone());

        let poly0 = Polytope::from_mats(
            weights.slice(s![..4, ..]).to_owned(),
            bias.slice(s![..4]).to_owned(),
        );
        let poly1 = Polytope::from_mats(
            weights.slice(s![5.., ..]).to_owned(),
            bias.slice(s![5..]).to_owned(),
        );

        assert_ne!(poly, poly0.intersection(&poly1));
    }

    #[test]
    pub fn test_intersection_n_0() {
        init_logger();

        let weights = arr2(&[
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-2.0, -1.0],
        ]);
        let bias = arr1(&[-1.0, -1.0, 4.0, 4.0, -3.0, 6.0, -2.0, -11.0]);
        let poly = Polytope::from_mats(weights.clone(), bias.clone());

        let poly0 = Polytope::from_mats(
            weights.slice(s![..2, ..]).to_owned(),
            bias.slice(s![..2]).to_owned(),
        );
        let poly1 = Polytope::from_mats(
            weights.slice(s![2..4, ..]).to_owned(),
            bias.slice(s![2..4]).to_owned(),
        );
        let poly2 = Polytope::from_mats(
            weights.slice(s![4..6, ..]).to_owned(),
            bias.slice(s![4..6]).to_owned(),
        );
        let poly3 = Polytope::from_mats(
            weights.slice(s![6.., ..]).to_owned(),
            bias.slice(s![6..]).to_owned(),
        );

        assert_eq!(
            poly,
            Polytope::intersection_n(
                0,
                &[
                    poly0.to_owned(),
                    poly1.to_owned(),
                    poly2.to_owned(),
                    poly3.to_owned()
                ]
            )
        );

        assert_ne!(
            poly,
            Polytope::intersection_n(
                0,
                &[
                    poly0.to_owned(),
                    poly2.to_owned(),
                    poly1.to_owned(),
                    poly3.to_owned()
                ]
            )
        );
        assert_ne!(poly, Polytope::intersection_n(0, &[poly0, poly1, poly2]));
    }

    #[test]
    pub fn test_poly_macro() {
        assert_eq!(
            poly!([[1, 0, 1]] < [2]),
            Polytope::from_mats(arr2(&[[1., 0., 1.]]), arr1(&[2.]))
        );
        assert_eq!(
            poly!([[-2, 0, 3], [4, -5, 9.3]] < [2, -1]),
            Polytope::from_mats(arr2(&[[-2., 0., 3.], [4., -5., 9.3]]), arr1(&[2., -1.]))
        );
    }
}
