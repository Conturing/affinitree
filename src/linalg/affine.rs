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

//! Structs to store linear functions and polytopes

use core::fmt;
use std::fmt::Debug;
use std::iter::{Sum, zip};
use std::marker::PhantomData;
use std::ops::{BitAnd, DivAssign, Mul, Neg};

use approx::{AbsDiffEq, RelativeEq};
use itertools::{Itertools, enumerate};
use ndarray::{
    self, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, DataMut, DataOwned, Ix1,
    Ix2, LinalgScalar, OwnedRepr, RawDataClone, ViewRepr, Zip, arr1, concatenate, s, stack,
};
use num_traits::float::Float;

pub struct AffFuncBase<T, S>
where
    S: Data,
    S::Elem: Float,
{
    pub mat: ArrayBase<S, Ix2>,
    pub bias: ArrayBase<S, Ix1>,
    pub _phantom: PhantomData<T>,
}

#[derive(Clone, Default, Debug)]
pub struct FunctionT;

#[derive(Clone, Default, Debug)]
pub struct PolytopeT;

// types going forward
type AffFuncG<A> = AffFuncBase<FunctionT, OwnedRepr<A>>;
type AffFuncViewG<'a, A> = AffFuncBase<FunctionT, ViewRepr<&'a A>>;
type PolytopeG<A> = AffFuncBase<PolytopeT, OwnedRepr<A>>;
type PolytopeViewG<'a, A> = AffFuncBase<PolytopeT, ViewRepr<&'a A>>;

// for compatibility
pub type AffFunc = AffFuncG<f64>;
pub type AffFuncView<'a> = AffFuncViewG<'a, f64>;
pub type Polytope = PolytopeG<f64>;
pub type PolytopeView<'a> = PolytopeViewG<'a, f64>;

impl<I, D: Data<Elem = A>, A: Float + Debug> Debug for AffFuncBase<I, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_tuple("AffFuncBase")
            .field(&self.mat)
            .field(&self.bias)
            .finish()
    }
}

impl<I, D: Data<Elem = A> + RawDataClone, A: Float + Clone> Clone for AffFuncBase<I, D> {
    fn clone(&self) -> Self {
        AffFuncBase {
            mat: self.mat.clone(),
            bias: self.bias.clone(),
            _phantom: self._phantom,
        }
    }
}

/// # General constructor
impl<I, D: Data<Elem = A>, A: Float> AffFuncBase<I, D> {
    /// Create a new instance of an affine combination consisting of a matrix mat: R^{m x n} and a vector bias: R^m.
    ///
    /// When interpreted as a function, it is equivalent to f(x) = mat @ x + bias.
    /// When interpreted as a polytope, it encodes the set P = {x | mat @ x <= bias}
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
            mat.iter().all(|x| x.is_normal() || x.is_zero()),
            "Non-normal floats are deprecated"
        );
        debug_assert!(
            bias.iter().all(|x| x.is_normal() || x.is_zero()),
            "Non-normal floats are deprecated"
        );

        AffFuncBase {
            mat,
            bias,
            _phantom: PhantomData,
        }
    }

    #[cfg(test)]
    pub fn from_random(dim_out: usize, dim_in: usize) -> AffFuncBase<I, OwnedRepr<f64>> {
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

        AffFuncBase::<I, OwnedRepr<f64>> {
            mat,
            bias,
            _phantom: PhantomData,
        }
    }
}

impl<I, A: Float> AffFuncBase<I, OwnedRepr<A>> {
    pub fn from_row_iter<'a, D, Iter>(
        indim: usize,
        outdim: usize,
        rows: Iter,
    ) -> AffFuncBase<I, OwnedRepr<A>>
    where
        D: Data<Elem = A>,
        Iter: IntoIterator<Item = (ArrayBase<D, Ix1>, &'a A)>,
        A: 'a,
    {
        let mut mat = Array2::zeros((outdim, indim));
        let mut bias = Array1::zeros(outdim);

        let mut iter = rows.into_iter();

        Zip::from(mat.axis_iter_mut(Axis(0)))
            .and(&mut bias)
            .for_each(|mut row, value| {
                let (x, y) = iter.next().unwrap_or_else(|| {
                    panic!(
                        "Invalid number of elements in iterator: expected at least {}",
                        outdim
                    )
                });

                row.assign(&x);
                *value = *y;
            });

        AffFuncBase::<I, OwnedRepr<A>>::from_mats(mat, bias)
    }
}

/// # AffFunc specific constructors
impl<A: Float> AffFuncG<A> {
    /// Creates an affine function that implements the identity function f(x)=x.
    #[inline(always)]
    #[rustfmt::skip]
    pub fn identity(dim: usize) -> AffFuncG<A> {
        AffFuncG::<A>::from_mats(
            Array2::eye(dim),
            Array1::zeros(dim)
        )
    }

    /// Creates an affine function that implements the zero function f(x)=0.
    #[inline(always)]
    #[rustfmt::skip]
    pub fn zeros(dim: usize) -> AffFuncG<A> {
        AffFuncG::<A>::from_mats(
            Array2::zeros((dim, dim)),
            Array1::zeros(dim)
        )
    }

    /// Creates an affine function that always returns the specified value.
    #[inline(always)]
    #[rustfmt::skip]
    pub fn constant(dim: usize, value: A) -> AffFuncG<A> {
        AffFuncG::<A>::from_mats(
            Array2::zeros((1, dim)),
            arr1(&[value])
        )
    }

    /// Creates an affine function that returns the element in the given index of its input.
    #[inline(always)]
    #[rustfmt::skip]
    pub fn unit(dim: usize, index: usize) -> AffFuncG<A> {
        let mut mat = Array2::zeros((1, dim));
        mat[[0, index]] = A::one();

        AffFuncG::<A>::from_mats(
            mat,
            Array1::zeros(1)
        )
    }

    /// Creates an affine function that sets the element at the given index to zero
    /// and leaves all other elements unchanged.
    #[inline(always)]
    #[rustfmt::skip]
    pub fn zero_idx(dim: usize, index: usize) -> AffFuncG<A> {
        let mut mat = Array2::eye(dim);
        mat[[index, index]] = A::zero();

        AffFuncG::<A>::from_mats(
            mat,
            Array1::zeros(dim)
        )
    }

    /// Creates an affine function R^dim -> R that returns the sum
    /// over all its inputs.
    #[inline(always)]
    #[rustfmt::skip]
    pub fn sum(dim: usize) -> AffFuncG<A> {
        AffFuncG::<A>::from_mats(
            Array2::ones((1, dim)),
            Array1::zeros(1)
        )
    }

    /// Creates an affine function that subtracts the right index from the left index.
    #[inline(always)]
    #[rustfmt::skip]
    pub fn subtraction(dim: usize, left: usize, right: usize) -> AffFuncG<A> {
        let mut matrix = Array2::zeros((1, dim));
        matrix[[0, left]] = A::one();
        matrix[[0, right]] = -A::one();
        let bias = Array1::zeros(1);

        AffFuncG::<A>::from_mats(
            matrix,
            bias
        )
    }

    /// Creates an affine function that rotates the space as specified
    /// by the given orthogonal matrix ``rotator``.
    #[inline(always)]
    #[rustfmt::skip]
    pub fn rotation(rotator: Array2<A>) -> AffFuncG<A> {
        assert_eq!(rotator.shape()[0], rotator.shape()[1]);
        let dim = rotator.shape()[0];

        AffFuncG::<A>::from_mats(
            rotator,
            Array1::zeros(dim)
        )
    }

    /// Creates an affine function that scales vectors uniformly along
    /// all axis.
    #[inline(always)]
    pub fn uniform_scaling(dim: usize, scalar: A) -> AffFuncG<A> {
        AffFuncG::<A>::scaling(&Array1::from_elem(dim, scalar))
    }

    /// Creates an affine function that scales vectors.
    #[inline(always)]
    pub fn scaling(scalars: &Array1<A>) -> AffFuncG<A> {
        AffFuncG::<A>::from_mats(
            Array2::from_diag(scalars),
            Array1::zeros(scalars.shape()[0]),
        )
    }

    /// Creates an affine function that slices inputs along specified axes.
    /// For each axis were ``reference_point`` is NaN, the corresponding axis will be kept.
    /// For each other axis, the axis is fixed with the value specified in ``reference_point``.
    #[inline(always)]
    pub fn slice(reference_point: &Array1<A>) -> AffFuncG<A> {
        let input_mask = reference_point.map(|x| if x.is_nan() { A::one() } else { A::zero() });
        let fixed_values = reference_point.map(|x| if x.is_nan() { A::zero() } else { *x });

        AffFuncG::<A>::from_mats(Array2::from_diag(&input_mask), fixed_values)
    }

    /// Creates an affine function that translates vectors by the given ``offset``.
    #[inline(always)]
    #[rustfmt::skip]
    pub fn translation(dim: usize, offset: Array1<A>) -> AffFuncG<A> {
        AffFuncG::<A>::from_mats(
            Array2::zeros((offset.shape()[0], dim)),
            offset
        )
    }
}

/// # Polytope specific constructors
impl<A: Float> PolytopeG<A> {
    /// Creates a polytope that contains the complete `dim`-dimensional ambient space.
    #[inline(always)]
    pub fn unbounded(dim: usize) -> PolytopeG<A> {
        PolytopeG::<A>::from_mats(Array2::zeros((1, dim)), Array1::ones(1))
    }

    /// Creates a `dim`-dimensional polytope that contains no point of the ambient space.
    #[inline(always)]
    pub fn empty(dim: usize) -> PolytopeG<A> {
        PolytopeG::<A>::from_mats(Array2::zeros((1, dim)), -Array1::ones(1))
    }

    /// Creates a polytope from a set of halfspaces described by ``normal_vectors`` and ``points`` in the plane (hesse normal form).
    #[inline(always)]
    pub fn from_normal(normal_vectors: Array2<A>, points: Array2<A>) -> PolytopeG<A> {
        let bias = (&normal_vectors).mul(&points).sum_axis(Axis(1));
        PolytopeG::<A>::from_mats(-normal_vectors, -bias)
    }

    /// Creates a ``dim``-dimensional hypercube centered at the origin.
    /// In each dimension two hyperplanes are placed with distance +/- ``radius`` from the origin.
    pub fn hypercube(dim: usize, radius: A) -> PolytopeG<A> {
        let mat: Array2<A> = concatenate![Axis(0), Array2::eye(dim), -Array2::eye(dim)];
        let bias: Array1<A> = Array1::from_elem(2 * dim, radius);
        PolytopeG::<A>::from_mats(mat, bias)
    }

    /// Creates a ``dim``-dimensional hyperrectangle centered at the origin.
    /// The distances from the origin to the faces of the rectangle are given by ``intervals`` in order of the axes.
    /// For interpretation of axis bounds see also [``Self::axis_bounds``].
    pub fn hyperrectangle(intervals: &[(A, A)]) -> PolytopeG<A> {
        let dim = intervals.len();
        let mut mat: Array2<A> = Array2::zeros((2 * dim, dim));
        let mut bias: Array1<A> = Array1::zeros(2 * dim);
        for (idx, (lower, upper)) in intervals.iter().enumerate() {
            Self::place_axis_bounds(2 * idx, &mut mat, &mut bias, idx, *lower, *upper);
        }
        PolytopeG::<A>::from_mats(mat, bias)
    }

    /// Creates a ``dim``-dimensional regular simplex with edge length sqrt(2) containing
    /// the origin.
    pub fn simplex(dim: usize) -> PolytopeG<A> {
        let mut mat = Array2::ones((dim + 1, dim));
        let bias = Array1::ones(dim + 1);

        let dist = -(A::one() + (A::from(dim).unwrap() + A::one()).sqrt() + A::from(dim).unwrap());

        for idx in 0..dim {
            mat[[idx, idx]] = mat[[idx, idx]] + dist;
        }

        PolytopeG::<A>::from_mats(mat, bias)
    }

    /// Creates a ``dim``-dimensional cross polytope centered at the origin.
    /// It generalizes the octahedron in that all its vertices are of the form
    /// +/- e_i where e_i is the i-th unit vector.
    pub fn cross_polytope(dim: usize) -> PolytopeG<A> {
        let rows = usize::pow(2, dim as u32);

        let mut mat = Array2::ones((rows, dim));
        let bias = Array1::ones(rows);

        for i in 0..rows {
            for j in 0..dim {
                if i.bitand(1 << j) != 0 {
                    mat[[i, j]] = -A::one();
                }
            }
        }

        PolytopeG::<A>::from_mats(mat, bias)
    }

    /// Creates a polytope that restricts the value of the specified `axis` to be bounded by
    /// the values of `lower_bound` and `upper_bound` (inclusive).
    /// A value of `neg_inf` (resp. `inf`) results in an unbounded lower (resp. upper) bound.
    /// The value of `lower_bound` must be less than or equal to `upper_bound`.
    #[inline(always)]
    pub fn axis_bounds(dim: usize, axis: usize, lower_bound: A, upper_bound: A) -> PolytopeG<A> {
        assert!(
            axis < dim,
            "Invalid axis received: axis {} does not exist in a {}-dimensional space",
            axis,
            dim
        );

        let mut mat = Array2::zeros((2, dim));
        let mut bias = Array1::zeros(2);
        Self::place_axis_bounds(0, &mut mat, &mut bias, axis, lower_bound, upper_bound);
        PolytopeG::<A>::from_mats(mat, bias)
    }

    fn place_axis_bounds<B: Float>(
        idx: usize,
        mat: &mut Array2<B>,
        bias: &mut Array1<B>,
        axis: usize,
        lower: B,
        upper: B,
    ) {
        assert!(lower <= upper);
        if lower.is_infinite() {
            bias[idx] = B::one();
        } else {
            mat[[idx, axis]] = -B::one();
            bias[idx] = -lower;
        }
        if upper.is_infinite() {
            bias[idx + 1] = B::one();
        } else {
            mat[[idx + 1, axis]] = B::one();
            bias[idx + 1] = upper;
        }
    }
}

/// # General methods
impl<I, D: Data<Elem = A>, A: Float> AffFuncBase<I, D> {
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
    pub fn matrix_view(&self) -> ArrayView2<D::Elem> {
        self.mat.view()
    }

    #[inline(always)]
    pub fn bias_view(&self) -> ArrayView1<D::Elem> {
        self.bias.view()
    }

    /// Returns the affine function that acts on component ``row``, which is
    /// the ``row``-th row of this function.
    pub fn row<'a>(&'a self, row: usize) -> AffFuncBase<I, ViewRepr<&'a A>> {
        assert!(
            row < self.outdim(),
            "Row outside range: got {} but only {} rows exist",
            row,
            self.outdim()
        );
        AffFuncBase::<I, ViewRepr<&'a A>>::from_mats(
            self.mat.row(row).insert_axis(Axis(0)),
            self.bias.slice(s![row]).insert_axis(Axis(0)),
        )
    }

    /// Returns an iterator over the rows of this AffFuncBase instance.
    /// Elements are returned as views.
    pub fn row_iter<'a>(&'a self) -> impl Iterator<Item = AffFuncBase<I, ViewRepr<&'a A>>> + 'a
    where
        A: 'a,
    {
        self.mat
            .outer_iter()
            .zip(self.bias.outer_iter())
            .map(|(row, bias)| {
                AffFuncBase::<I, ViewRepr<&'a A>>::from_mats(
                    row.insert_axis(Axis(0)),
                    bias.insert_axis(Axis(0)),
                )
            })
    }

    // /// Iterate over the columns of this AffFuncBase instance.
    // /// Elements are returned as views.
    // pub fn column_iter(&self) -> impl Iterator<Item = AffFuncBase<I, ViewRepr>> + '_ {
    //     self.mat.axis_iter(Axis(1))
    //         .map(|column| {
    //             AffFuncBase::<I, ViewRepr>::from_mats(
    //                 column.insert_axis(Axis(1)),
    //                 ArrayView1::zeros(1),
    //             )
    //         })
    // }

    /// Removes the rows given by ``indices``.
    /// The iterator must return the rows to remove in ascending order.
    pub fn remove_rows<Iter: IntoIterator<Item = usize>>(
        &self,
        indices: Iter,
    ) -> AffFuncBase<I, OwnedRepr<A>> {
        let mut indices = indices.into_iter();
        let mut index = indices.next();

        let rows = self
            .mat
            .axis_iter(Axis(0))
            .zip(self.bias.iter())
            .enumerate()
            .filter(|(idx, _)| {
                if let Some(val) = index {
                    if *idx == val {
                        index = indices.next();
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            })
            .map(|(_, data)| data)
            .collect_vec();

        if index.is_some() {
            panic!(
                "iterator of indices for removal should have been completely consumed after checking each row"
            );
        }

        AffFuncBase::<I, OwnedRepr<A>>::from_row_iter(self.indim(), rows.len(), rows)
    }

    pub fn remove_zero_rows(&self) -> AffFuncBase<I, OwnedRepr<A>> {
        let rows = self
            .mat
            .axis_iter(Axis(0))
            .zip(self.bias.iter())
            .filter(|(r, &v)| r.iter().any(|&x| x != A::zero()) || v != A::zero())
            .collect_vec();

        AffFuncBase::<I, OwnedRepr<A>>::from_row_iter(self.indim(), rows.len(), rows)
    }

    pub fn remove_zero_columns(&self) -> AffFuncBase<I, OwnedRepr<A>> {
        let rows = self
            .mat
            .axis_iter(Axis(1))
            .filter(|r| r.iter().any(|&x| x != A::zero()))
            .collect_vec();

        AffFuncBase::<I, OwnedRepr<A>>::from_mats(
            stack(Axis(1), rows.as_slice()).unwrap(),
            self.bias.to_owned(),
        )
    }
}

impl<D: Data<Elem = A>, A: Float> AffFuncBase<PolytopeT, D> {
    /// Returns the number of inequalities (constraints) of this polytope.
    pub fn n_constraints(&self) -> usize {
        self.mat.shape()[0]
    }
}

impl<I, A: Float + DivAssign + Sum> AffFuncBase<I, OwnedRepr<A>> {
    /// Normalizes every row of this affine function / polytope with respect to Euclidean norm (l2).
    pub fn normalize(mut self) -> AffFuncBase<I, OwnedRepr<A>> {
        for (mut row, mut bias) in zip(self.mat.outer_iter_mut(), self.bias.outer_iter_mut()) {
            let norm: A = row.iter().map(|&x| x.powi(2)).sum::<A>().sqrt();
            if norm > A::epsilon() {
                row.map_inplace(|x| *x /= norm);
                bias.map_inplace(|x| *x /= norm);
            }
        }
        self
    }
}

impl<A: Float> AffFuncBase<PolytopeT, OwnedRepr<A>> {
    /// Removes row constraints which are always satisfied on their own.
    /// When a row constraint is encountered that is always false the
    /// whole polytope is replaced by [``PolytopeG::empty()``].
    pub fn remove_tautologies(&self) -> AffFuncBase<PolytopeT, OwnedRepr<A>> {
        let rows: Option<Vec<_>> = self
            .mat
            .axis_iter(Axis(0))
            .zip(self.bias.iter())
            .filter_map(|(r, v)| {
                if r.iter().all(|&x| x == A::zero()) {
                    if *v >= A::zero() {
                        // superfluous bound
                        None
                    } else {
                        // infeasible
                        Some(None)
                    }
                } else {
                    Some(Some((r, v)))
                }
            })
            .collect();

        let rows = match rows {
            Some(rows) => rows,
            None => return PolytopeG::<A>::empty(self.indim()),
        };

        if rows.is_empty() {
            Self::unbounded(self.indim())
        } else {
            PolytopeG::<A>::from_row_iter(self.indim(), rows.len(), rows)
        }
    }
}

impl<A: Float + DivAssign + Sum + RelativeEq<A, Epsilon: Clone>>
    AffFuncBase<PolytopeT, OwnedRepr<A>>
{
    /// Removes all duplicate rows of this polytope from back to front.
    /// Time complexity is O(n m^2) where n is the number of columns and m the
    /// number of rows.
    pub fn remove_duplicate_rows(&self) -> AffFuncBase<PolytopeT, OwnedRepr<A>> {
        let normal = self.clone().normalize();

        let mut dups: Vec<usize> = Vec::with_capacity(self.n_constraints());
        for i in (0..self.n_constraints()).rev() {
            for j in (0..i).rev() {
                let mat_eq = ArrayView1::relative_eq(
                    &normal.mat.row(i),
                    &normal.mat.row(j),
                    A::default_epsilon(),
                    A::default_max_relative(),
                );
                let bias_eq = A::relative_eq(
                    &normal.bias[i],
                    &normal.bias[j],
                    A::default_epsilon(),
                    A::default_max_relative(),
                );
                if mat_eq && bias_eq {
                    dups.push(i);
                    break;
                }
            }
        }

        self.remove_rows(dups.into_iter().rev())
    }
}

/// # Evaluation on inputs
impl<D: Data<Elem = A>, A: Float + LinalgScalar> AffFuncBase<FunctionT, D> {
    /// Evaluates this function under the given input.
    /// Mathematically, this corresponds to calculating mat @ input + bias
    pub fn apply<S: Data<Elem = A>>(&self, input: &ArrayBase<S, Ix1>) -> Array1<A> {
        self.mat.dot(input) + &self.bias
    }

    /// Evaluates the transposed of this function under the given input.
    /// For orthogonal functions this corresponds to the inverse.
    /// Mathematically, this corresponds to calculating mat.T @ (input - bias)
    pub fn apply_transpose<S: Data<Elem = A>>(&self, input: &ArrayBase<S, Ix1>) -> Array1<A> {
        self.mat.t().dot(&(input - &self.bias))
    }
}

/// # Distances
impl<D: Data<Elem = A>, A: Float + LinalgScalar> AffFuncBase<PolytopeT, D> {
    /// Calculates the distance from a point to the hyperplanes defined by the
    /// rows of this polytope.
    ///
    /// # Warning
    ///
    /// Distance is not normalized.
    #[inline]
    pub fn distance_raw<S: Data<Elem = A>>(&self, point: &ArrayBase<S, Ix1>) -> Array1<A> {
        &self.bias - self.mat.dot(point)
    }

    /// Calculates the distances from multiple points to the hyperplanes defined by the
    /// rows of this polytope.
    ///
    /// # Warning
    ///
    /// Distance is not normalized.
    #[inline]
    pub fn distances_raw<S: Data<Elem = A>>(&self, point: &ArrayBase<S, Ix2>) -> Array2<A> {
        let b_bias = self.bias.broadcast(point.dim()).unwrap();
        &b_bias.t() - self.mat.dot(point)
    }
}

impl<D: Data<Elem = A>, A: Float + LinalgScalar + DivAssign + Sum> AffFuncBase<PolytopeT, D> {
    /// Calculates the (normalized) distance from a point to the hyperplanes defined by the
    /// rows of this polytope.
    ///
    /// Precisely, it returns a vector where each element is the distance from the given point to the hyperplane in order.
    /// Distance is positive if the point is inside the halfspace of that inequality and negative otherwise.
    /// Returns f64::INFINITY if the corresponding halfspace includes all points.
    pub fn distance<S: Data<Elem = A>>(&self, point: &ArrayBase<S, Ix1>) -> Array1<A> {
        let mut raw_dist = self.distance_raw(point);
        for (row, mut dist) in zip(self.mat.outer_iter(), raw_dist.outer_iter_mut()) {
            let norm: A = row.iter().map(|&x| x.powi(2)).sum::<A>().sqrt();
            dist.map_inplace(|x| *x /= norm);
        }
        raw_dist
    }
}

impl<D: Data<Elem = A>, A: Float + LinalgScalar> AffFuncBase<PolytopeT, D> {
    /// Tests whether the input ``point`` lies inside this polytope or not.
    #[inline]
    pub fn contains<S: Data<Elem = A>>(&self, point: &ArrayBase<S, Ix1>) -> bool {
        self.distance_raw(point)
            .into_iter()
            .all(|x| x >= A::from(-1e-8).unwrap())
    }
}

/// # Combination of two AffFunc instances
impl<D: Data<Elem = A>, A: Float + LinalgScalar> AffFuncBase<FunctionT, D> {
    /// Composes self with other. The resulting function will have the same effect as first applying other and then self.
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
    pub fn compose(
        &self,
        other: &AffFuncBase<FunctionT, D>,
    ) -> AffFuncBase<FunctionT, OwnedRepr<A>> {
        assert_eq!(
            self.indim(),
            other.outdim(),
            "Invalid shared dimensions for composition: {} and {}",
            self.indim(),
            other.outdim()
        );
        AffFuncBase::<FunctionT, OwnedRepr<A>>::from_mats(
            self.mat.dot(&other.mat),
            self.apply(&other.bias),
        )
    }

    /// Stacks this function on top of ``other`` vertically.
    pub fn stack(&self, other: &AffFuncBase<FunctionT, D>) -> AffFuncBase<FunctionT, OwnedRepr<A>> {
        assert_eq!(
            self.indim(),
            other.indim(),
            "Invalid input dimensions for stacking: {} and {}",
            self.indim(),
            other.indim()
        );
        AffFuncBase::<FunctionT, OwnedRepr<A>>::from_mats(
            concatenate![Axis(0), self.mat, other.mat],
            concatenate![Axis(0), self.bias, other.bias],
        )
    }
}

impl<D: Data<Elem = A> + DataOwned + RawDataClone + DataMut, A: Float + LinalgScalar + Neg>
    AffFuncBase<FunctionT, D>
{
    pub fn negate(self) -> AffFuncBase<FunctionT, D> {
        AffFuncBase::<FunctionT, D>::from_mats(-self.mat.clone(), -self.bias)
    }
}

/// # Combination of Polytopes
impl<D: Data<Elem = A>, A: Float + LinalgScalar> AffFuncBase<PolytopeT, D> {
    pub fn translate(&self, direction: &Array1<A>) -> PolytopeG<A> {
        PolytopeG::<A>::from_mats(self.mat.to_owned(), &self.bias + self.mat.dot(direction))
    }

    pub fn intersection(&self, other: &AffFuncBase<PolytopeT, D>) -> PolytopeG<A> {
        assert!(self.indim() == other.indim());

        let mat = concatenate![Axis(0), self.mat, other.mat];
        let bias = concatenate![Axis(0), self.bias, other.bias];

        PolytopeG::<A>::from_mats(mat, bias)
    }

    /// Constructs a new polytope representing the intersection of `polys`.
    /// That is, the resulting polytope contains all points that are contained in each polytope of `polys`.
    #[rustfmt::skip]
    pub fn intersection_n(dim: usize, polys: &[AffFuncBase<PolytopeT, D>]) -> PolytopeG<A> {
        if polys.is_empty() {
            return PolytopeG::<A>::unbounded(dim);
        }

        let mat_view: Vec<ArrayView2<A>> = polys.iter()
            .map(|poly| poly.mat.view())
            .collect();
        let mat_concat = ndarray::concatenate(Axis(0), mat_view.as_slice());
        let mat = match mat_concat {
            Ok(result) => result,
            Err(error) => panic!(
                "Error when concatenating matrices, probably caused by a mismatch in dimensions: {:?}",
                error
            )
        };

        let bias_view: Vec<ArrayView1<A>> = polys
            .iter()
            .map(|poly| poly.bias.view())
            .collect();
        let bias_concat = ndarray::concatenate(Axis(0), bias_view.as_slice());
        let bias = match bias_concat {
            Ok(result) => result,
            Err(error) => panic!(
                "Error when concatenating bias, probably caused by a mismatch in dimensions: {:?}",
                error
            )
        };

        PolytopeG::<A>::from_mats(mat, bias)
    }

    /// Applies the given function to the input space of this polytope.
    pub fn apply_pre<D2: Data<Elem = A>>(&self, func: &AffFuncBase<FunctionT, D2>) -> PolytopeG<A>
// where
    //     D2: Ownership
    {
        assert_eq!(
            self.indim(),
            func.outdim(),
            "Invalid shared dimensions for composition: {} and {}",
            self.indim(),
            func.outdim()
        );

        PolytopeG::<A>::from_mats(
            self.mat.dot(&func.mat),
            -self.mat.dot(&func.bias) + &self.bias,
        )
    }

    /// Applies the function f(x) = inverse_mat^-1 @ x + bias to the output space of this polytope.
    /// This method does not compute the inverse function. Instead, the inverse must be provided.
    pub fn apply_post<D2, D3>(
        &self,
        inverse_mat: &ArrayBase<D2, Ix2>,
        bias: &ArrayBase<D3, Ix1>,
    ) -> PolytopeG<A>
    where
        D2: Data<Elem = A>,
        D3: Data<Elem = A>,
    {
        assert_eq!(
            self.indim(),
            inverse_mat.shape()[0],
            "Invalid shared dimensions for composition: {} and {}",
            self.indim(),
            inverse_mat.shape()[0]
        );
        assert_eq!(
            inverse_mat.shape()[0],
            bias.shape()[0],
            "Invalid dimensions of bias: expected {} but got {}",
            inverse_mat.shape()[0],
            bias.shape()[0]
        );

        PolytopeG::<A>::from_mats(
            self.mat.dot(inverse_mat),
            self.mat.dot(&inverse_mat.dot(bias)) + &self.bias,
        )
    }

    /// Rotate this polytope (i.e., the points contained in it) by the given rotation matrix.
    /// The matrix must represent a rotation, i.e., it must be orthogonal.
    /// The center of rotation is the origin.
    pub fn rotate<D2>(&self, orthogonal_mat: &ArrayBase<D2, Ix2>) -> PolytopeG<A>
    where
        D2: Data<Elem = A>,
    {
        self.apply_post(&orthogonal_mat.t(), &Array1::zeros(self.indim()))
    }

    // pub fn slice<D2>(&self, reference_vec: &ArrayBase<D2, Ix1>, reduce_dim: bool, add_constraints: bool) -> AffFuncBase<PolytopeT, OwnedRepr<A>>
    // where
    //     D2: Data<Elem = A>
    // {
    //     let diag = reference_vec.map(|x| {
    //         if *x == A::zero() {
    //             A::from(1).unwrap()
    //         } else {
    //             A::zero()
    //         }
    //     });

    //     let mut poly = self.apply_pre(&AffFuncG::<A>::from_mats(Array2::from_diag(&diag), reference_vec.to_owned()));

    //     // remove any columns that are zero
    //     if reduce_dim {
    //         use itertools::Itertools;

    //         let nonzero_columns = reference_vec.iter()
    //             .zip(poly.mat.axis_iter(Axis(1)))
    //             .filter(|(&x, _)| x == A::zero())
    //             .map(|(_, column)| column.insert_axis(Axis(1)))
    //             .collect_vec();

    //         poly.mat = concatenate(Axis(1), nonzero_columns.as_slice()).unwrap();
    //     }

    //     if add_constraints {
    //         use itertools::Itertools;

    //         let mut constraints = reference_vec.iter()
    //             .enumerate()
    //             .filter(|(_, &x)| x != A::zero())
    //             .map(|(idx, val)| Polytope::axis_bounds(self.indim(), idx, val - 0.0001, val + 0.0001))
    //             .collect_vec();

    //         constraints.push(poly);

    //         poly = PolytopeG::<A>::intersection_n(self.indim(), constraints.as_slice());
    //     }

    //     poly
    // }
}

// Switch between ownership
impl<I, S: Data<Elem = A>, A: Float> AffFuncBase<I, S> {
    #[inline]
    pub fn view<'a>(&'a self) -> AffFuncBase<I, ViewRepr<&'a A>> {
        AffFuncBase::<I, ViewRepr<&'a A>> {
            mat: self.mat.view(),
            bias: self.bias.view(),
            _phantom: PhantomData,
        }
    }
}

impl<I, S: Data<Elem = A>, A: Float> AffFuncBase<I, S> {
    #[inline]
    pub fn to_owned(&self) -> AffFuncBase<I, OwnedRepr<A>> {
        AffFuncBase::<I, OwnedRepr<A>> {
            mat: self.mat.to_owned(),
            bias: self.bias.to_owned(),
            _phantom: PhantomData,
        }
    }
}

// Switch between types
impl<D: Data<Elem = A> + RawDataClone, A: Float> AffFuncBase<FunctionT, D> {
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

impl<D: Data<Elem = A> + RawDataClone, A: Float> AffFuncBase<PolytopeT, D> {
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
}

impl<A: Float> AffFuncBase<PolytopeT, OwnedRepr<A>> {
    #[inline]
    pub fn convert_to(self, repr: PolyRepr) -> AffFuncBase<FunctionT, OwnedRepr<A>> {
        match repr {
            PolyRepr::MatrixLeqBias => AffFuncBase::<FunctionT, OwnedRepr<A>> {
                mat: self.mat,
                bias: self.bias,
                _phantom: PhantomData,
            },
            PolyRepr::MatrixBiasLeqZero => AffFuncBase::<FunctionT, OwnedRepr<A>> {
                mat: self.mat,
                bias: -self.bias,
                _phantom: PhantomData,
            },
            PolyRepr::MatrixGeqBias => AffFuncBase::<FunctionT, OwnedRepr<A>> {
                mat: -self.mat,
                bias: -self.bias,
                _phantom: PhantomData,
            },
            PolyRepr::MatrixBiasGeqZero => AffFuncBase::<FunctionT, OwnedRepr<A>> {
                mat: -self.mat,
                bias: self.bias,
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

impl<D: Data<Elem = A> + RawDataClone, A: Float> AffFuncBase<PolytopeT, D> {
    /// Returns the linear program that encodes the chebyshev center of this polytope.
    /// That is, the resulting polytope contains all points that are
    ///
    /// 1. contained in this polytope
    /// 2. whose last dimension is less than or equal to the minimal distance to the hyperplanes of this polytope
    ///
    /// The returned array encodes the coefficients of the cost function.
    /// Minimizing this function over the region of the returned polytope gives the center point and radius of the largest enclosed sphere inside this polytope.
    pub fn chebyshev_center(&self) -> (PolytopeG<A>, Array1<A>) {
        // distance to each hyperplane
        let mut norm = Array2::<A>::zeros((self.mat.len_of(Axis(0)), 1));

        for (idx, row) in enumerate(self.mat.outer_iter()) {
            norm[[idx, 0]] = row.map(|x: &A| x.powi(2)).sum().sqrt();
        }
        let mat = concatenate![Axis(1), self.mat, norm];

        // radius must be positive
        let mut radius = Array2::<A>::zeros((1, mat.len_of(Axis(1))));
        radius[[0, mat.len_of(Axis(1)) - 1]] = -A::one();
        let mat = concatenate![Axis(0), mat, radius];
        let bias = concatenate![Axis(0), self.bias.clone(), Array1::<A>::zeros(1)];

        (
            PolytopeG::<A>::from_mats(mat, bias),
            Array1::from_iter(radius.iter().cloned()),
        )
    }
}

impl<I, D: Data<Elem = A>, A: Float> core::cmp::PartialEq for AffFuncBase<I, D> {
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

impl<I, S, A> AbsDiffEq for AffFuncBase<I, S>
where
    S: Data<Elem = A>,
    A: Float + AbsDiffEq,
    A::Epsilon: Clone,
{
    type Epsilon = A::Epsilon;

    fn default_epsilon() -> A::Epsilon {
        A::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: A::Epsilon) -> bool {
        <ArrayBase<S, Ix2> as AbsDiffEq<_>>::abs_diff_eq(&self.mat, &other.mat, epsilon.clone())
            && <ArrayBase<S, Ix1> as AbsDiffEq<_>>::abs_diff_eq(&self.bias, &other.bias, epsilon)
    }
}

impl<I, S, A> RelativeEq for AffFuncBase<I, S>
where
    S: Data<Elem = A>,
    A: Float + RelativeEq,
    A::Epsilon: Clone,
{
    fn default_max_relative() -> Self::Epsilon {
        A::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: A::Epsilon, max_relative: A::Epsilon) -> bool {
        <ArrayBase<S, Ix2> as RelativeEq<_>>::relative_eq(
            &self.mat,
            &other.mat,
            epsilon.clone(),
            max_relative.clone(),
        ) && <ArrayBase<S, Ix1> as RelativeEq<_>>::relative_eq(
            &self.bias,
            &other.bias,
            epsilon,
            max_relative,
        )
    }
}

/// Creates a new ``AffFunc`` from the given matrix and bias.
///
/// See also ndarray's ``array`` macro
///
/// # Examples
///
/// ```rust
/// use affinitree::aff;
///
/// let func = aff!([[1, 2, 5, 7], [-2, -9, 7, 8]] + [1, -1]);
/// ```
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

/// Creates a new ``Polytope`` from the given matrix and bias.
///
/// # Examples
///
/// ```rust
/// use affinitree::poly;
///
/// let poly = poly!([[1, 0], [0, 1]] < [2, 3]);
/// ```
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
    use approx::assert_relative_eq;
    use itertools::Itertools;
    use ndarray::{Array2, Axis, arr1, arr2, array, s};

    use super::*;

    fn init_logger() {
        use env_logger::Target;
        use log::LevelFilter;

        let _ = env_logger::builder()
            .is_test(true)
            .filter_module("minilp", LevelFilter::Error)
            .target(Target::Stdout)
            .filter_level(LevelFilter::Warn)
            .try_init();
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
    pub fn test_from_row_iter() {
        let mat = array![[1., 2.], [-3., 0.5], [0.3, 1e+4]];
        let bias = arr1(&[0.1, -7., 1e-2]);

        let f = AffFunc::from_row_iter(2, 3, mat.axis_iter(Axis(0)).zip(bias.iter()));

        assert_eq!(mat, f.mat);
        assert_eq!(bias, f.bias);
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
    fn test_slice_afffunc() {
        let res = AffFunc::slice(&arr1(&[2., f64::NAN, -4.]));

        assert_eq!(res.apply(&arr1(&[5., -3., 7.])), arr1(&[2., -3., -4.]));

        assert_eq!(res.apply(&arr1(&[0., 9., -0.3])), arr1(&[2., 9., -4.]));
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

        assert_eq!(f.row(0).to_owned(), aff!([[1, 2, 5, 7]] + [1]));
        assert_eq!(f.row(1).to_owned(), aff!([[-2, -9, 7, 8]] + [-1]));
    }

    #[test]
    pub fn test_row_iter() {
        let f = AffFunc::from_mats(Array2::eye(4), arr1(&[1., 2., 3., 4.]));

        let fs = f.row_iter().map(|x| x.to_owned()).collect_vec();

        assert_eq!(
            fs,
            vec![
                aff!([1., 0., 0., 0.] + 1.),
                aff!([0., 1., 0., 0.] + 2.),
                aff!([0., 0., 1., 0.] + 3.),
                aff!([0., 0., 0., 1.] + 4.)
            ]
        )
    }

    // #[test]
    // pub fn test_column_iter() {
    //     let f = AffFunc::from_mats(Array2::eye(4), arr1(&[1., 2., 3., 4.]));

    //     let fs = f.column_iter().map(|x| x.to_owned()).collect_vec();

    //     assert_eq!(fs, vec![
    //         aff!([1., 0., 0., 0.] + 0.),
    //         aff!([0., 1., 0., 0.] + 0.),
    //         aff!([0., 0., 1., 0.] + 0.),
    //         aff!([0., 0., 0., 1.] + 0.)
    //     ])
    // }

    #[test]
    pub fn test_remove_rows() {
        let f = aff!([[1, 0, 2], [0, 3, -1], [2, 0.5, 0]] + [-7, 5, 1]);

        let g = f.remove_rows(vec![0, 2]);

        assert_eq!(g, aff!([[0, 3, -1]] + [5]));
    }

    #[test]
    pub fn test_remove_zero_rows() {
        let f = aff!([[1, 0, 2], [0, 0, 0], [0, 0, 0]] + [0, 0, 1]);

        let g = f.remove_zero_rows();

        assert_eq!(g, aff!([[1, 0, 2], [0, 0, 0]] + [0, 1]));
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

    #[rustfmt::skip]
    #[test]
    pub fn test_normalize_zero() {
        let f = aff!([[0, 0, 0], [0, 1, 0]] + [2, 5]);

        assert_eq!(
            f.normalize(),
            aff!([[0., 0, 0.], [0, 1, 0]] + [2, 5])
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
        let a = AffFunc::from_random(4, 3);
        let b = AffFunc::identity(4);
        let c = b.compose(&a);
        assert_eq!(c.indim(), 3);
        assert_eq!(c.outdim(), 4);
    }

    /* Polytope Tests */

    #[test]
    pub fn test_unbounded() {
        let poly = Polytope::unbounded(4);

        assert_eq!(poly.indim(), 4);
        assert!(poly.contains(&arr1(&[1.5, 0.0, -2.3, 1.0])));
        assert!(poly.contains(&arr1(&[1.0, 0.9, 0.2, 0.3])));
        assert!(poly.contains(&arr1(&[2.0, -1.0, -7.6, 2.6])));
    }

    #[test]
    pub fn test_hyperrectangle() {
        let ival = [(1., 2.), (-1., 1.)];

        let poly = Polytope::hyperrectangle(&ival);

        assert_eq!(poly.indim(), 2);
        assert!(poly.contains(&arr1(&[1.5, 0.0])));
        assert!(poly.contains(&arr1(&[1.0, 0.9])));
        assert!(poly.contains(&arr1(&[2.0, -1.0])));
        assert!(!poly.contains(&arr1(&[2.1, -0.2])));
        assert!(!poly.contains(&arr1(&[3.5, 4.0])));
        assert!(!poly.contains(&arr1(&[1.1, -2.0])));
        assert!(!poly.contains(&arr1(&[1.8, 1.2])));
    }

    #[test]
    pub fn test_simplex() {
        let poly = Polytope::simplex(2);

        assert_eq!(poly.indim(), 2);
        assert!(poly.contains(&arr1(&[1.0, 0.0])));
        assert!(poly.contains(&arr1(&[0.1, 0.9])));
        assert!(poly.contains(&arr1(&[-0.3, -0.3])));
        assert!(!poly.contains(&arr1(&[1.1, -0.2])));
        assert!(!poly.contains(&arr1(&[-0.4, 0.1])));
        assert!(!poly.contains(&arr1(&[-0.2, -0.5])));
    }

    #[test]
    pub fn test_cross_polytope() {
        let poly = Polytope::cross_polytope(2);

        assert_eq!(poly.indim(), 2);
        assert!(poly.contains(&arr1(&[1.0, 0.0])));
        assert!(poly.contains(&arr1(&[0.1, 0.9])));
        assert!(poly.contains(&arr1(&[-0.3, -0.3])));
        assert!(!poly.contains(&arr1(&[-1.1, -0.2])));
        assert!(!poly.contains(&arr1(&[-0.5, 0.6])));
        assert!(!poly.contains(&arr1(&[-0.2, -1.5])));
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
    pub fn test_remove_tautologies() {
        let poly = poly!([[1, 0, -1], [0, 0, 0], [0, 1, 0]] < [2, 1, 5]);

        assert_eq!(
            poly.remove_tautologies(),
            poly!([[1, 0, -1], [0, 1, 0]] < [2, 5])
        );
    }

    #[rustfmt::skip]
    #[test]
    pub fn test_remove_tautologies_zero() {
        let poly = poly!([[1, 0, -1, 0], [0, 0, 0, 0]] < [-7, 0]);

        assert_eq!(
            poly.remove_tautologies(),
            poly!([[1, 0, -1, 0]] < [-7])
        );
    }

    #[test]
    pub fn test_remove_tautologies_all_zero() {
        let poly = Polytope::from_mats(Array2::zeros((4, 10)), Array1::zeros(4));

        assert_eq!(
            poly.remove_tautologies(),
            Polytope::from_mats(Array2::zeros((1, 10)), Array1::ones(1))
        );
    }

    #[rustfmt::skip]
    #[test]
    pub fn test_remove_tautologies_infeasible() {
        let poly = poly!([[1, 0, -1], [0, 0, 0], [0, 1, 0]] < [2, -2, 5]);

        assert_eq!(
            poly.remove_tautologies(),
            poly!([[0, 0, 0]] < [-1])
        );
    }

    #[rustfmt::skip]
    #[test]
    pub fn test_remove_tautologies_only() {
        let poly = poly!([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] < [2, 7, 6]);

        assert_eq!(
            poly.remove_tautologies(),
            poly!([[0, 0, 0, 0, 0]] < [1])
        );
    }

    #[test]
    pub fn test_remove_duplicate_rows() {
        let poly = poly!([[1, 0, -1], [0, 2, 0], [1, 0, -1]] < [2, -2, 2]);

        assert_eq!(
            poly.remove_duplicate_rows(),
            poly!([[1, 0, -1], [0, 2, 0]] < [2, -2])
        );
    }

    #[test]
    pub fn test_remove_duplicate_rows_scaled() {
        let poly = poly!([[1, 0, -1], [-8, 2, 0], [1.5, 0, -1.5], [-4, 1, 0]] < [2, 10, 3, 5]);

        assert_eq!(
            poly.remove_duplicate_rows(),
            poly!([[1, 0, -1], [-8, 2, 0]] < [2, 10])
        );
    }

    #[test]
    pub fn test_remove_duplicate_rows_different_bias() {
        let poly = poly!([[1, 0, -1], [0, 2, 0], [1, 0, -1]] < [2, -2, 3]);

        assert_eq!(
            poly.remove_duplicate_rows(),
            poly!([[1, 0, -1], [0, 2, 0], [1, 0, -1]] < [2, -2, 3])
        );
    }

    #[test]
    pub fn test_remove_duplicate_rows_close() {
        let poly = poly!([[1, 0, -1], [0, 2, 0], [1, 0, -0.999]] < [2, -2, 2]);

        assert_eq!(
            poly.remove_duplicate_rows(),
            poly!([[1, 0, -1], [0, 2, 0], [1, 0, -0.999]] < [2, -2, 2])
        );
    }

    #[test]
    pub fn test_distance() {
        let poly = poly!([[2., 0., 0.]] < [2.]);

        assert_eq!(poly.distance(&arr1(&[1., 0., 0.])), arr1(&[0.]));
        assert_eq!(poly.distance(&arr1(&[-7., 0., 0.])), arr1(&[8.]));
        assert_eq!(poly.distance(&arr1(&[7., 0., 0.])), arr1(&[-6.]));
    }

    #[test]
    pub fn test_distance_unbounded() {
        let poly = Polytope::unbounded(4);

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
    fn test_rotate() {
        // define triangle with vertices (1, 1), (1, 2.87), (4.71, 1)
        let poly = poly!([[0, -1], [-1, 0], [0.45, 0.89]] < [-1, -1, 3]);

        // rotate by 45° counterclockwise
        let rot = array![[0.71, -0.71], [0.71, 0.71]];

        assert!(poly.contains(&array![1.1, 1.1]));
        assert!(poly.contains(&array![1.1, 2.6]));
        assert!(poly.contains(&array![2.0, 2.0]));
        assert!(poly.contains(&array![4.4, 1.1]));
        assert!(poly.contains(&array![3.0, 1.4]));

        let poly2 = poly.rotate(&rot);

        // rotated points
        assert!(poly2.contains(&array![0.0, 1.562]));
        assert!(poly2.contains(&array![-1.065, 2.627]));
        assert!(poly2.contains(&array![0.0, 2.84]));
        assert!(poly2.contains(&array![2.343, 3.905]));
        assert!(poly2.contains(&array![1.136, 3.124]));

        // points just outside the triangle
        assert!(!poly2.contains(&array![-1.7, 2.8]));
        assert!(!poly2.contains(&array![3.1, 4.3]));
        assert!(!poly2.contains(&array![0.2, 3.6]));
        assert!(!poly2.contains(&array![-1., 2.]));
        assert!(!poly2.contains(&array![1.1, 2.1]));
    }

    // #[test]
    // fn test_slice() {
    //     // define triangle with vertices (1, 1), (1, 2.87), (4.71, 1)
    //     let poly = poly!([[0, -1], [-1, 0], [0.45, 0.89]] < [-1, -1, 3]);

    //     let poly2 = poly.slice(&array![0., 1.5], false, false);

    //     // points just inside
    //     assert!(poly2.contains(&array![1.1, 0.]));
    //     assert!(poly2.contains(&array![3.6, 0.]));

    //     // points just outside
    //     assert!(!poly2.contains(&array![0.9, 0.]));
    //     assert!(!poly2.contains(&array![3.8, 0.]));
    // }

    // #[test]
    // fn test_slice_reduce() {
    //     // define triangle with vertices (1, 1), (1, 2.87), (4.71, 1)
    //     let poly = poly!([[0, -1], [-1, 0], [0.45, 0.89]] < [-1, -1, 3]);

    //     let poly2 = poly.slice(&array![0., 1.5], true, false);

    //     // points just inside
    //     assert!(poly2.contains(&array![1.1]));
    //     assert!(poly2.contains(&array![3.6]));

    //     // points just outside
    //     assert!(!poly2.contains(&array![0.9]));
    //     assert!(!poly2.contains(&array![3.8]));
    // }

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
