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

//! Format linear functions and polytopes

use core::fmt;
use core::iter::zip;
use std::fmt::{Debug, Display};
use std::ops::{Bound, RangeBounds};

use itertools::Itertools;
use itertools::Position::{First, Last, Middle, Only};
use ndarray::{ArrayView1, Data};

use super::affine::{AffFuncBase, AffFuncView, FunctionT, PolytopeT, PolytopeView};

#[derive(Debug, PartialEq, Clone)]
pub struct FormatOptions {
    /// Sort coefficients in descending order when the number of coefficients is bigger than or equal to ``sort_coefficients``
    pub sort_coefficients: usize,
    pub simplify_zero: bool,
    pub simplify_tautologies: bool,
    pub normalize: bool,
    pub skip_axes_n: usize,
    pub skip_axes: (Bound<i32>, Bound<i32>),
    pub skip_rows_n: usize,
    pub skip_rows: (Bound<i32>, Bound<i32>),
}

impl Default for FormatOptions {
    fn default() -> Self {
        FormatOptions {
            sort_coefficients: 0,
            simplify_zero: false,
            simplify_tautologies: false,
            normalize: false,
            skip_axes_n: 0,
            skip_axes: (Bound::Included(1), Bound::Excluded(0)),
            skip_rows_n: 0,
            skip_rows: (Bound::Included(1), Bound::Excluded(0)),
        }
    }
}

impl FormatOptions {
    pub fn default_func() -> FormatOptions {
        FormatOptions {
            sort_coefficients: 0,
            simplify_zero: true,
            simplify_tautologies: false,
            normalize: false,
            skip_axes_n: 0,
            skip_axes: (std::ops::Bound::Included(20), std::ops::Bound::Unbounded),
            skip_rows_n: 0,
            skip_rows: (std::ops::Bound::Included(5), std::ops::Bound::Unbounded),
        }
    }

    pub fn default_poly() -> FormatOptions {
        FormatOptions {
            sort_coefficients: 5,
            simplify_zero: false,
            simplify_tautologies: true,
            normalize: true,
            skip_axes_n: 0,
            skip_axes: (std::ops::Bound::Included(20), std::ops::Bound::Unbounded),
            skip_rows_n: 0,
            skip_rows: (std::ops::Bound::Included(5), std::ops::Bound::Unbounded),
        }
    }

    pub fn show_all_rows(mut self) -> Self {
        self.skip_rows = (std::ops::Bound::Included(1), std::ops::Bound::Excluded(0));
        self
    }

    pub fn show_all_axes(mut self) -> Self {
        self.skip_axes = (std::ops::Bound::Included(1), std::ops::Bound::Excluded(0));
        self
    }
}

// impl FormatOptions {
//     fn update_ranges(&mut self, indim: usize, outdim: usize) {
//         if self.skip_axes_n > indim {
//             self.skip_axes = (Bound::Included(1), Bound::Excluded(0));
//         } else {
//             self.skip_axes.1 = match self.skip_axes.1 {
//                 Bound::Excluded(val) => Bound::Excluded(if val.is_negative() { indim as i32 + val } else { val }),
//                 Bound::Included(val) => Bound::Included(if val.is_negative() { indim as i32 + val } else { val }),
//                 Bound::Unbounded => Bound::Unbounded,
//             };
//         }
//     }
// }

const TRUE: &str = "⊤";
const FALSE: &str = "⊥";
const PLUS: &str = "+";
const MINUS: &str = "−";
const LEQ: &str = "≤";
const ELLIPSIS: &str = "⋯";
const VERT_ELLIPSIS: &str = "⋮";

impl<S: Data<Elem = f64>> Display for AffFuncBase<FunctionT, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.display_with(FormatOptions::default_func()), f)
    }
}

impl<S: Data<Elem = f64>> Display for AffFuncBase<PolytopeT, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.display_with(FormatOptions::default_poly()), f)
    }
}

impl<'a, T, S: Data<Elem = f64>> AffFuncBase<T, S> {
    pub fn display_with(&'a self, options: FormatOptions) -> AffFuncBasePrinter<'a, T> {
        AffFuncBasePrinter::new(self.view(), options)
    }
}

#[derive(Clone)]
pub struct AffFuncBasePrinter<'a, T> {
    pub instance: AffFuncBase<T, ndarray::ViewRepr<&'a f64>>,
    pub options: FormatOptions,
}

impl<'a, T> AffFuncBasePrinter<'a, T> {
    pub fn new(
        instance: AffFuncBase<T, ndarray::ViewRepr<&'a f64>>,
        options: FormatOptions,
    ) -> Self {
        AffFuncBasePrinter::<'a, T> { instance, options }
    }
}

impl Display for AffFuncBasePrinter<'_, FunctionT> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write_func(f, self.instance.clone(), &self.options)
    }
}

impl Display for AffFuncBasePrinter<'_, PolytopeT> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write_poly(f, self.instance.clone(), &self.options)
    }
}

impl<T> Debug for AffFuncBasePrinter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.instance, f)
    }
}

/// Write a string representation of pred into f.
/// Here pred is interpreted as A @ x <= b
pub fn write_poly(
    f: &mut fmt::Formatter,
    pred: PolytopeView,
    options: &FormatOptions,
) -> std::fmt::Result {
    let mut first_skip = true;
    for (no, (row, (pos, bias))) in zip(
        pred.matrix_view().outer_iter(),
        pred.bias_view().iter().with_position(),
    )
    .enumerate()
    {
        if options.skip_rows.contains(&(no as i32)) {
            if first_skip {
                writeln!(f, " {}", VERT_ELLIPSIS)?;
                first_skip = false;
            }
            continue;
        }

        write_inequality(f, row, *bias, options)?;

        match pos {
            First | Middle => writeln!(f)?,
            _ => {}
        }
    }
    Ok(())
}

pub fn write_func(
    f: &mut fmt::Formatter,
    func: AffFuncView,
    options: &FormatOptions,
) -> std::fmt::Result {
    let mut first_skip = true;
    for (no, (row, (pos, bias))) in zip(
        func.matrix_view().outer_iter(),
        func.bias_view().iter().with_position(),
    )
    .enumerate()
    {
        if options.skip_rows.contains(&(no as i32)) {
            if first_skip {
                writeln!(f, " {}", VERT_ELLIPSIS)?;
                first_skip = false;
            }
            continue;
        }

        write_affcomb(f, row, *bias, options)?;

        match pos {
            First | Middle => writeln!(f)?,
            _ => {}
        }
    }
    Ok(())
}

pub fn write_inequality(
    f: &mut fmt::Formatter,
    row: ArrayView1<f64>,
    bias: f64,
    options: &FormatOptions,
) -> std::fmt::Result {
    let all_zero = row.iter().all(|x| *x == 0.0);
    if options.simplify_tautologies && all_zero {
        if bias >= 0.0 {
            return write!(f, "{}", TRUE);
        } else {
            return write!(f, "{}", FALSE);
        }
    }

    if options.normalize && !all_zero {
        let scale = row.iter().fold(0f64, |a, &b| a.max(b.abs()));
        let row = row.map(|x| *x / scale);
        let bias = bias / scale;

        write_lincomb(f, row.view(), options)?;
        write!(f, " {} ", LEQ)?;
        write_float(f, bias)
    } else {
        write_lincomb(f, row, options)?;
        write!(f, " {} ", LEQ)?;
        write_float(f, bias)
    }
}

pub fn write_affcomb(
    f: &mut fmt::Formatter,
    row: ArrayView1<f64>,
    bias: f64,
    options: &FormatOptions,
) -> std::fmt::Result {
    write_float(f, bias)?;
    write!(f, " ")?;

    if !(options.simplify_zero && row.iter().all(|x| *x == 0.0)) {
        write_lincomb(f, row, options)?;
    }

    Ok(())
}

pub fn write_lincomb(
    f: &mut fmt::Formatter,
    coefficients: ArrayView1<f64>,
    options: &FormatOptions,
) -> std::fmt::Result {
    let num = coefficients.shape()[0];
    let mut elements = coefficients.iter().copied().enumerate().collect_vec();

    if options.sort_coefficients != 0 && options.sort_coefficients <= num {
        elements.sort_unstable_by_key(|(_, value)| float_ord::FloatOrd(-value.abs()));
    }

    let mut first_skip = true;

    for (no, (pos, (idx, coeff))) in elements.into_iter().with_position().enumerate() {
        if options.skip_axes.contains(&(no as i32)) {
            if first_skip {
                write!(f, " {}", ELLIPSIS)?;
                first_skip = false;
            }
            continue;
        }

        match pos {
            First | Only => {}
            Middle | Last => {
                write!(f, " ")?;
            }
        }
        write_float(f, coeff)?;
        write!(f, " ${}", idx)?;
    }
    Ok(())
}

#[inline]
pub fn write_float(f: &mut fmt::Formatter, value: f64) -> std::fmt::Result {
    if value.is_sign_negative() {
        write!(f, "{}", MINUS)?;
    } else {
        write!(f, "{}", PLUS)?;
    }
    let precision = f.precision().unwrap_or(2);
    write!(f, "{:.*}", precision, value.abs())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, arr1};

    use super::*;
    use crate::linalg::affine::Polytope;
    use crate::poly;

    struct AffCombination {
        coefficients: Array1<f64>,
        bias: f64,
        options: FormatOptions,
    }

    impl Display for AffCombination {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write_affcomb(f, self.coefficients.view(), self.bias, &self.options)
        }
    }

    #[rustfmt::skip]
    #[test]
    fn test_affcomp() {
        let opt = FormatOptions::default();

        let actual = format!("{}", AffCombination { coefficients: arr1(&[1., -7., 2., 5.]), bias: 5., options: opt});

        let expected = "+5.00 +1.00 $0 −7.00 $1 +2.00 $2 +5.00 $3";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_affcomp_sorted() {
        let opt = FormatOptions { sort_coefficients: 1, ..Default::default() };

        let actual = format!("{}", AffCombination { coefficients: arr1(&[1., -7., 2., 5.]), bias: 5., options: opt});

        let expected = "+5.00 −7.00 $1 +5.00 $3 +2.00 $2 +1.00 $0";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_affcomp_skip() {
        let opt = FormatOptions { skip_axes: (Bound::Included(1), Bound::Excluded(3)), ..Default::default() };

        let actual = format!("{}", AffCombination { coefficients: arr1(&[1., -7., 2., 5.]), bias: 5., options: opt});

        let expected = "+5.00 +1.00 $0 ⋯ +5.00 $3";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_affcomp_skip_all() {
        let opt = FormatOptions { skip_axes: (Bound::Included(1), Bound::Unbounded), ..Default::default() };

        let actual = format!("{}", AffCombination { coefficients: arr1(&[1., -7., 2., 5.]), bias: 5., options: opt});

        let expected = "+5.00 +1.00 $0 ⋯";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_affcomp_skip_sorted() {
        let opt = FormatOptions { skip_axes: (Bound::Included(1), Bound::Excluded(3)), sort_coefficients: 1, ..Default::default() };

        let actual = format!("{}", AffCombination { coefficients: arr1(&[1., -7., 2., 5.]), bias: 5., options: opt});

        let expected = "+5.00 −7.00 $1 ⋯ +1.00 $0";

        assert_eq!(actual, expected);
    }

    struct Inequality {
        coefficients: Array1<f64>,
        bias: f64,
        options: FormatOptions,
    }

    impl Display for Inequality {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write_inequality(f, self.coefficients.view(), self.bias, &self.options)
        }
    }

    #[rustfmt::skip]
    #[test]
    fn test_ineq() {
        let opt = FormatOptions::default();

        let actual = format!("{}", Inequality { coefficients: arr1(&[1., -7., 2., 5.]), bias: 5., options: opt});

        let expected = "+1.00 $0 −7.00 $1 +2.00 $2 +5.00 $3 ≤ +5.00";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_ineq_sorted() {
        let opt = FormatOptions { sort_coefficients: 1, ..Default::default() };

        let actual = format!("{}", Inequality { coefficients: arr1(&[1., -7., 2., 5.]), bias: 5., options: opt});

        let expected = "−7.00 $1 +5.00 $3 +2.00 $2 +1.00 $0 ≤ +5.00";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_ineq_skip() {
        let opt =  FormatOptions { skip_axes: (Bound::Included(1), Bound::Excluded(3)), ..Default::default() };

        let actual = format!("{}", Inequality { coefficients: arr1(&[1., -7., 2., 5.]), bias: 5., options: opt});

        let expected = "+1.00 $0 ⋯ +5.00 $3 ≤ +5.00";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_ineq_skip_all() {
        let opt = FormatOptions { skip_axes: (Bound::Included(1), Bound::Unbounded), ..Default::default() };

        let actual = format!("{}", Inequality { coefficients: arr1(&[1., -7., 2., 5.]), bias: 5., options: opt});

        let expected = "+1.00 $0 ⋯ ≤ +5.00";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_ineq_skip_sorted() {
        let opt = FormatOptions { skip_axes: (Bound::Included(1), Bound::Excluded(3)), sort_coefficients: 1, ..Default::default() };

        let actual = format!("{}", Inequality { coefficients: arr1(&[1., -7., 2., 5.]), bias: 5., options: opt});

        let expected = "−7.00 $1 ⋯ +1.00 $0 ≤ +5.00";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_ineq_normalize() {
        let opt = FormatOptions { 
            normalize: true,
            simplify_tautologies: true,
            simplify_zero: true, 
            ..Default::default() 
        };

        let actual = format!("{}", Inequality { coefficients: arr1(&[1., -7., 2., 5.]), bias: 5., options: opt});

        let expected = "+0.14 $0 −1.00 $1 +0.29 $2 +0.71 $3 ≤ +0.71";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_ineq_tautology_true() {
        let opt = FormatOptions { 
            normalize: true,
            simplify_tautologies: true,
            simplify_zero: true, 
            ..Default::default() 
        };

        let actual = format!("{}", Inequality { coefficients: arr1(&[0., -0., 0., 0.]), bias: 5., options: opt});

        let expected = "⊤";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_ineq_tautology_false() {
        let opt = FormatOptions { 
            normalize: true,
            simplify_tautologies: true,
            simplify_zero: true, 
            ..Default::default() 
        };

        let actual = format!("{}", Inequality { coefficients: arr1(&[0., -0., 0., 0.]), bias: -5., options: opt});

        let expected = "⊥";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_ineq_zero() {
        let opt = FormatOptions { 
            normalize: true,
            simplify_zero: true, 
            ..Default::default() 
        };

        let actual = format!("{}", Inequality { coefficients: arr1(&[0., -0., 0., 0.]), bias: 5., options: opt});

        let expected = "+0.00 $0 −0.00 $1 +0.00 $2 +0.00 $3 ≤ +5.00";

        assert_eq!(actual, expected);
    }

    struct Poly {
        coefficients: Array2<f64>,
        bias: Array1<f64>,
        options: FormatOptions,
    }

    impl Display for Poly {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let poly = Polytope::from_mats(self.coefficients.to_owned(), self.bias.to_owned());
            write_poly(f, poly.view(), &self.options)
        }
    }

    #[rustfmt::skip]
    #[test]
    fn test_poly() {
        let opt = FormatOptions { 
            normalize: true,
            simplify_zero: true,
            skip_axes: (Bound::Included(2), Bound::Unbounded),
            skip_rows: (Bound::Included(2), Bound::Unbounded),
            ..Default::default() 
        };

        let actual = format!("{}", Poly { coefficients: Array2::from_elem((30, 30), 2.), bias: Array1::from_elem(30, -4.), options: opt });

        let expected = "+1.00 $0 +1.00 $1 ⋯ ≤ −2.00\n+1.00 $0 +1.00 $1 ⋯ ≤ −2.00\n ⋮\n";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_display_with() {
        let opt = FormatOptions { 
            normalize: true,
            simplify_zero: true,
            skip_axes: (Bound::Included(2), Bound::Unbounded),
            skip_rows: (Bound::Included(2), Bound::Unbounded),
            ..Default::default() 
        };

        let actual = format!("{}", poly!([[2, 2, 2], [2, 2, 2], [2, 2, 2]] < [-4, -4, -4]).display_with(opt));

        let expected = "+1.00 $0 +1.00 $1 ⋯ ≤ −2.00\n+1.00 $0 +1.00 $1 ⋯ ≤ −2.00\n ⋮\n";

        assert_eq!(actual, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_display_with_view() {
        let opt = FormatOptions { 
            normalize: true,
            simplify_zero: true,
            skip_axes: (Bound::Included(2), Bound::Unbounded),
            skip_rows: (Bound::Included(2), Bound::Unbounded),
            ..Default::default() 
        };

        let actual = format!("{}", poly!([[2, 2, 2], [2, 2, 2], [2, 2, 2]] < [-4, -4, -4]).view().display_with(opt));

        let expected = "+1.00 $0 +1.00 $1 ⋯ ≤ −2.00\n+1.00 $0 +1.00 $1 ⋯ ≤ −2.00\n ⋮\n";

        assert_eq!(actual, expected);
    }
}
