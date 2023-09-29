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

//! A collection of high-level methods to distill AffTree instances out of neural networks

use std::borrow::Borrow;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use console::style;
use indicatif::{HumanDuration, ProgressBar, ProgressStyle};
use itertools::Itertools;
use ndarray_npy::{NpzReader, ReadNpyError, ReadNpzError};
use regex::Regex;

use crate::core::afftree::AffTree;
use crate::core::schema::{
    argmax as argmax_dd, class_characterization as class_char_dd, partial_ReLU,
};
use crate::linalg::affine::AffFunc;

#[derive(Debug, Clone)]
pub enum Layer {
    Linear(AffFunc),
    ReLU,
    Argmax,
    ClassChar(usize),
}

// TODO docs
pub fn afftree_from_layers<I>(dim: usize, layers: I) -> AffTree<2>
where
    I: IntoIterator,
    I::Item: Borrow<Layer>,
{
    let mut dim = dim;

    let container = layers.into_iter().collect_vec();

    let mut n_splits = 0;
    let mut first_dim = 0;
    let mut last_dim = dim;

    for layer in &container {
        match layer.borrow() {
            Layer::Linear(func) => {
                last_dim = func.outdim();
                if first_dim == 0 {
                    first_dim = func.outdim();
                }
            }
            Layer::ReLU => n_splits += last_dim,
            Layer::ClassChar(_) => n_splits += 1,
            Layer::Argmax => n_splits += last_dim,
        }
    }

    // Rough underapproximation of the number of nodes
    // It is assumed that after the first layer each node has on average q children.
    let estimated_capacity = {
        let n = first_dim as i32;
        let m = n_splits.saturating_sub(first_dim) as i32;
        let q = 1.3f64;

        2.0f64.powi(n) * (2.0 + (q.powi(m + 1) - q) / (q - 1.0))
    } as usize;
    println!("Estimated number of layers: {}", n_splits + 1);
    println!("Estimated number of nodes: {}", estimated_capacity);

    let mut dd = AffTree::<2>::with_capacity(dim, estimated_capacity);

    let pb = ProgressBar::new(n_splits as u64);
    let sty = ProgressStyle::default_bar()
        .template(&format!(
            "{: >12} {}",
            style("Building").cyan().bold(),
            "[{bar:25}] {pos:>2}/{len:2} ({elapsed})"
        ))
        .unwrap()
        .progress_chars("=> ");
    pb.set_style(sty.clone());

    for layer in container.into_iter() {
        match layer.borrow() {
            Layer::Linear(aff) => {
                assert!(aff.indim() == dim);
                dd.apply_func(aff);
                dim = aff.outdim();
            }
            Layer::ReLU => {
                for idx in 0..dim {
                    let timer = Instant::now();
                    dd.compose::<true>(&partial_ReLU(dim, idx));
                    let duration = timer.elapsed();

                    pb.println(format!(
                        "{: >12} partial ReLU in {:#} ({} nodes, {} terminals)",
                        style("Finished").green().bold(),
                        HumanDuration(duration),
                        dd.tree.len(),
                        dd.tree.num_terminals()
                    ));
                    pb.inc(1);
                }
            }
            Layer::Argmax => dd.compose::<true>(&argmax_dd(dim)),
            Layer::ClassChar(clazz) => dd.compose::<true>(&class_char_dd(dim, *clazz)),
        }
    }
    pb.finish_and_clear();
    println!(
        "\n{: >12} constructing decision tree",
        style("Completed").green().bold()
    );
    dd
}

// TODO: docs
pub fn read_layers<P: AsRef<Path>>(path: &P) -> Result<Vec<Layer>, ReadNpzError> {
    let file = File::open(path).map_err(ReadNpyError::from)?;
    let mut npz = NpzReader::new(file)?;

    let mut names = npz.names()?;
    names.sort_unstable();

    let pattern = Regex::new(r"^(\d+)\.(linear.weights|relu).npy$").unwrap();

    let layers = names
        .iter()
        .filter_map(|x| pattern.captures(x))
        .map(|x| match x.get(2).unwrap().as_str() {
            "relu" => Ok(Layer::ReLU),
            "linear.weights" => Ok(Layer::Linear(AffFunc::from_mats(
                npz.by_name(&format!(
                    "{}.linear.weights.npy",
                    x.get(1).unwrap().as_str()
                ))?,
                npz.by_name(&format!("{}.linear.bias.npy", x.get(1).unwrap().as_str()))?,
            ))),
            _ => unreachable!(),
        })
        .collect::<Result<Vec<Layer>, ReadNpzError>>()?;

    Ok(layers)
}

#[cfg(test)]
mod tests {

    use crate::core::builder::{afftree_from_layers, read_layers};
    use approx::assert_relative_eq;
    use ndarray::arr1;
    use std::path::Path;

    #[test]
    pub fn test_read_npy() {
        let layers = read_layers(&Path::new("res/nn/ecoli.npz")).unwrap();

        let dd = afftree_from_layers(7, &layers);

        assert_relative_eq!(
            dd.evaluate(&arr1(&[0.0, 1.0, 6.0, 2.0, -100.0, 7.0, -1.0]))
                .unwrap(),
            arr1(&[-19.85087719, 79.9919784, 20.50838996, -114.81462218]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            dd.evaluate(&arr1(&[0.0, 1.0, 6.0, 2.0, 0.0, 7.0, -1.0]))
                .unwrap(),
            arr1(&[-6.84282267, 10.55842791, -1.14444066, -23.78016759]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            dd.evaluate(&arr1(&[0.0, 1.0, 6.0, 2.0, 0.0, -7.0, -1.0]))
                .unwrap(),
            arr1(&[4.25559725, -9.07232097, -16.83579659, -5.00034567]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );

        assert_relative_eq!(
            dd.evaluate(&arr1(&[5.0, -2.0, 3.0, 50.0, -5.0, -2.0, 8.0]))
                .unwrap(),
            arr1(&[4.59256626, 21.26106201, 26.68923948, -43.60172981]),
            epsilon = 1e-08,
            max_relative = 1e-05
        );
    }
}
