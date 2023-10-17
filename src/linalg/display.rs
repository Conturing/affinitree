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

//! Print linear functions and polytopes to the command line

use itertools::enumerate;
use ndarray::Array1;

use crate::linalg::affine::{AffFunc, Polytope};
use core::iter::zip;
use itertools::{
    Itertools,
    Position::{First, Last, Middle, Only},
};
use std::fmt::Write;

/// Write a string representation of pred into f.
/// Here pred is interpreted as A @ x <= b
pub fn write_polytope<T: Write>(
    f: &mut T,
    pred: &Polytope,
    pretty_print: bool,
) -> std::fmt::Result {
    for (row, bias) in zip(
        pred.get_matrix().outer_iter(),
        pred.get_bias().iter().with_position(),
    ) {
        write_inequality(
            f,
            row.to_owned(),
            *bias.into_inner(),
            pretty_print,
            pretty_print,
            pretty_print,
        )?;

        match bias {
            First(_) | Middle(_) => writeln!(f)?,
            _ => {}
        }
    }
    Ok(())
}

pub fn write_aff<T: Write>(f: &mut T, func: &AffFunc, pretty_print: bool) -> std::fmt::Result {
    for (row, bias) in zip(
        func.get_matrix().outer_iter(),
        func.get_bias().iter().with_position(),
    ) {
        write_aff_row(
            f,
            row.to_owned(),
            *bias.into_inner(),
            pretty_print,
            pretty_print,
        )?;

        match bias {
            First(_) | Middle(_) => writeln!(f)?,
            _ => {}
        }
    }
    Ok(())
}

pub fn write_inequality<T: Write>(
    f: &mut T,
    row: Array1<f64>,
    bias: f64,
    simplify_tautologies: bool,
    normalize: bool,
    truncate: bool,
) -> std::fmt::Result {
    if simplify_tautologies && row.iter().all(|x| *x == 0.0) {
        if bias >= 0.0 {
            return write!(f, "TRUE");
        } else {
            return write!(f, "FALSE");
        }
    }

    let (row, bias) = if normalize {
        let scale = row.iter().fold(0f64, |a, &b| a.max(b.abs()));
        (row.map(|x| *x / scale), bias / scale)
    } else {
        (row, bias)
    };

    write_lincomb(f, row, truncate)?;

    write!(f, " <= ")?;

    if truncate {
        write!(f, "{:.2}", bias)
    } else {
        write!(f, "{}", bias)
    }
}

pub fn write_aff_row<T: Write>(
    f: &mut T,
    row: Array1<f64>,
    bias: f64,
    simplify_zero: bool,
    truncate: bool,
) -> std::fmt::Result {
    if simplify_zero && row.iter().all(|x| *x == 0.0) {
        if truncate {
            return write!(f, "{:.2}", bias);
        } else {
            return write!(f, "{}", bias);
        }
    }

    write_lincomb(f, row, truncate)?;

    if bias.is_sign_negative() {
        write!(f, " − ")?
    } else {
        write!(f, " + ")?
    }

    if truncate {
        write!(f, "{:.2}", bias.abs())
    } else {
        write!(f, "{}", bias.abs())
    }
}

pub fn write_lincomb<T: Write>(
    f: &mut T,
    coefficients: Array1<f64>,
    truncate: bool,
) -> std::fmt::Result {
    for (idx, coeff) in enumerate(coefficients.iter().with_position()) {
        match coeff {
            First(val) | Only(val) => {
                if val.is_sign_negative() {
                    write!(f, "−${} ", idx)?
                } else {
                    write!(f, " ${} ", idx)?
                }
            }
            Middle(val) | Last(val) => {
                if val.is_sign_negative() {
                    write!(f, " −${} ", idx)?
                } else {
                    write!(f, " +${} ", idx)?
                }
            }
        }
        if truncate {
            write!(f, "{:.2}", coeff.into_inner().abs())?
        } else {
            write!(f, "{}", coeff.into_inner().abs())?
        }
    }
    Ok(())
}
