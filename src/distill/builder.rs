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
use std::time::{Duration, Instant};

use console::style;
use indicatif::{HumanDuration, ProgressBar, ProgressStyle};
use itertools::Itertools;
use ndarray_npy::{NpzReader, ReadNpyError, ReadNpzError};
use regex::Regex;

use crate::distill::schema::{
    argmax, class_characterization, partial_ReLU, partial_hard_sigmoid, partial_hard_tanh,
};
use crate::linalg::affine::AffFunc;
use crate::pwl::afftree::AffTree;

pub trait NodeEstimator {
    fn estimate_nodes<Item: Borrow<Layer>>(
        &self,
        dim: usize,
        current_depth: usize,
        layers: &[Item],
    ) -> (usize, usize);
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimpleNodeEstimator;

impl NodeEstimator for SimpleNodeEstimator {
    fn estimate_nodes<Item>(
        &self,
        dim: usize,
        _current_depth: usize,
        layers: &[Item],
    ) -> (usize, usize)
    where
        Item: Borrow<Layer>,
    {
        let mut n_splits = 0;
        let mut first_dim = 0;
        let mut last_dim = dim;

        for layer in layers {
            match layer.borrow() {
                Layer::Linear(func) => {
                    last_dim = func.outdim();
                    if first_dim == 0 {
                        first_dim = func.outdim();
                    }
                }
                Layer::ReLU(_) => n_splits += 1,
                Layer::HardTanh(_) => n_splits += 1,
                Layer::HardSigmoid(_) => n_splits += 1,
                Layer::ClassChar(_) => n_splits += 1,
                Layer::Argmax => n_splits += last_dim,
            }
        }

        // Rough underapproximation of the number of nodes
        // It is assumed that after the first layer each node has on average q children.
        let n = first_dim as i32;
        let m = n_splits.saturating_sub(first_dim) as i32;
        let q = 1.3f64;

        let estimated_capacity =
            (2.0f64.powi(n) * (2.0 + (q.powi(m + 1) - q) / (q - 1.0))) as usize;

        (n_splits, estimated_capacity)
    }
}

pub trait DistillVisitor {
    fn start_distill(&mut self, dim: usize, n_layers: usize, n_nodes: usize);
    fn start_layer(&mut self, layer: &Layer);
    fn finish_layer(
        &mut self,
        layer: &Layer,
        new_nodes: usize,
        decision_nodes: usize,
        terminal_nodes: usize,
    );
    fn finish_distill(&mut self, total_decisions: usize, total_terminals: usize);
}

#[derive(Clone, Debug)]
pub struct DistillConsole {
    pb: ProgressBar,
    timer: Instant,
    len: usize,
}

impl DistillConsole {
    pub fn new() -> DistillConsole {
        DistillConsole {
            pb: ProgressBar::hidden(),
            timer: Instant::now(),
            len: 0,
        }
    }
}

impl Default for DistillConsole {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillVisitor for DistillConsole {
    fn start_distill(&mut self, dim: usize, n_layers: usize, n_nodes: usize) {
        self.pb = ProgressBar::new(n_layers as u64);
        let sty = ProgressStyle::default_bar()
            .template(&format!(
                "{: >12} {}",
                style("Building").cyan().bold(),
                "[{bar:25}] {pos:>2}/{len:2} ({elapsed})"
            ))
            .unwrap()
            .progress_chars("=> ");
        self.pb.set_style(sty.clone());
        self.pb.enable_steady_tick(Duration::from_secs(5));

        println!("Input dim: {}", dim);
        println!("Estimated number of layers: {}", n_layers);
        println!("Estimated number of nodes: {}", n_nodes);

        self.timer = Instant::now();
        self.len = n_layers;
    }

    fn start_layer(&mut self, _layer: &Layer) {
        self.timer = Instant::now();
    }

    fn finish_layer(
        &mut self,
        layer: &Layer,
        _new_nodes: usize,
        decision_nodes: usize,
        terminal_nodes: usize,
    ) {
        let duration = self.timer.elapsed();
        match layer {
            Layer::Linear(_) => {}
            Layer::ReLU(_) => {
                self.pb.println(format!(
                    "{: >12} partial ReLU in {:#} ({} nodes, {} terminals)",
                    style("Finished").green().bold(),
                    HumanDuration(duration),
                    decision_nodes + terminal_nodes,
                    terminal_nodes
                ));
                self.pb.inc(1);
            }
            Layer::HardTanh(_) => {
                self.pb.println(format!(
                    "{: >12} partial hard tanh in {:#} ({} nodes, {} terminals)",
                    style("Finished").green().bold(),
                    HumanDuration(duration),
                    decision_nodes + terminal_nodes,
                    terminal_nodes
                ));
                self.pb.inc(1);
            }
            Layer::HardSigmoid(_) => {
                self.pb.println(format!(
                    "{: >12} partial hard sigmoid in {:#} ({} nodes, {} terminals)",
                    style("Finished").green().bold(),
                    HumanDuration(duration),
                    decision_nodes + terminal_nodes,
                    terminal_nodes
                ));
                self.pb.inc(1);
            }
            Layer::ClassChar(_) => {}
            Layer::Argmax => {}
        }
    }

    fn finish_distill(&mut self, total_decisions: usize, total_terminals: usize) {
        self.pb.finish_and_clear();
        println!(
            "\n{: >12} constructing decision tree ({} decisions, {} terminals)",
            style("Completed").green().bold(),
            total_decisions,
            total_terminals
        );
    }
}

#[derive(serde::Serialize)]
struct CsvRow {
    depth: usize,
    inner_nodes: usize,
    terminal_nodes: usize,
    time_ms: u128,
    in_dim: usize,
}

#[derive(Debug)]
pub struct DistillCsv {
    writer: csv::Writer<File>,
    timer: Instant,
    depth: usize,
    in_dim: usize,
    pb: ProgressBar,
}

impl DistillCsv {
    pub fn new<P: AsRef<Path>>(path: P) -> DistillCsv {
        DistillCsv {
            writer: csv::Writer::from_path(path).unwrap(),
            timer: Instant::now(),
            depth: 0,
            in_dim: 0,
            pb: ProgressBar::hidden(),
        }
    }
}

impl DistillVisitor for DistillCsv {
    fn start_distill(&mut self, dim: usize, n_layers: usize, _n_nodes: usize) {
        self.timer = Instant::now();
        self.depth = 0;
        self.in_dim = dim;
        self.pb = ProgressBar::new(n_layers as u64);
        let sty = ProgressStyle::default_bar()
            .template(&format!(
                "{: >12} {}",
                style("Building").cyan().bold(),
                "[{bar:20}] {pos:>2}/{len:2} ({elapsed})"
            ))
            .unwrap()
            .progress_chars("=> ");
        self.pb.set_style(sty.clone());
    }

    fn start_layer(&mut self, _layer: &Layer) {
        self.timer = Instant::now();
    }

    fn finish_layer(
        &mut self,
        layer: &Layer,
        _new_nodes: usize,
        decision_nodes: usize,
        terminal_nodes: usize,
    ) {
        let duration = self.timer.elapsed();
        match layer {
            Layer::Linear(_) => {}
            Layer::ReLU(_) | Layer::HardTanh(_) | Layer::HardSigmoid(_) => {
                self.depth += 1;
                self.writer
                    .serialize(CsvRow {
                        depth: self.depth,
                        inner_nodes: decision_nodes,
                        terminal_nodes,
                        time_ms: duration.as_millis(),
                        in_dim: self.in_dim,
                    })
                    .unwrap();
                self.writer.flush().unwrap();
                self.pb.inc(1);
            }
            Layer::ClassChar(_) => {}
            Layer::Argmax => {}
        }
    }

    fn finish_distill(&mut self, _total_decisions: usize, _total_terminals: usize) {
        self.writer.flush().unwrap();
        self.pb.finish_and_clear();
    }
}

#[derive(Clone, Debug)]
pub struct NoOpVis {}

impl DistillVisitor for NoOpVis {
    fn start_distill(&mut self, _: usize, _: usize, _: usize) {}
    fn start_layer(&mut self, _: &Layer) {}
    fn finish_layer(&mut self, _: &Layer, _: usize, _: usize, _: usize) {}
    fn finish_distill(&mut self, _: usize, _: usize) {}
}

/// A simple enum type to conveniently specify the layer structure of a neural network.
/// Each ``Layer`` corresponds to one piece-wise linear function.
#[derive(Debug, Clone)]
pub enum Layer {
    /// A fully connected linear layer
    Linear(AffFunc),
    /// The ReLU applied to the i-th component of the input
    ReLU(usize),
    /// The hard hyperbolic tangent applied to the i-th component of the input
    HardTanh(usize),
    /// The hard sigmoid applied to the i-th component of the input
    HardSigmoid(usize),
    /// The argmax function
    Argmax,
    /// A binary version of the argmax called class characterization.
    /// The result is a boolean indicating whether the input belongs to the specified class or not.
    ClassChar(usize),
}

pub fn afftree_from_layers<I>(dim: usize, layers: I, precondition: Option<AffTree<2>>) -> AffTree<2>
where
    I: IntoIterator,
    I::Item: Borrow<Layer>,
{
    afftree_from_layers_generic(
        dim,
        layers,
        precondition,
        &mut SimpleNodeEstimator {},
        &mut NoOpVis {},
    )
}

/// Specialization of [`afftree_from_layers_generic`] that logs the progress
/// after each layer to the console.
pub fn afftree_from_layers_verbose<I>(
    dim: usize,
    layers: I,
    precondition: Option<AffTree<2>>,
) -> AffTree<2>
where
    I: IntoIterator,
    I::Item: Borrow<Layer>,
{
    afftree_from_layers_generic(
        dim,
        layers,
        precondition,
        &mut SimpleNodeEstimator {},
        &mut DistillConsole::new(),
    )
}

/// Specialization of [`afftree_from_layers_generic`] that logs characteristics of the tree
/// after each layer to a csv file located at ``path``.
pub fn afftree_from_layers_csv<I, P: AsRef<Path>>(
    dim: usize,
    layers: I,
    path: P,
    precondition: Option<AffTree<2>>,
) -> AffTree<2>
where
    I: IntoIterator,
    I::Item: Borrow<Layer>,
{
    afftree_from_layers_generic(
        dim,
        layers,
        precondition,
        &mut SimpleNodeEstimator {},
        &mut DistillCsv::new(path),
    )
}

/// Generic implementation of the distillation process.
///
/// The provided sequence of ``layers`` is mapped to equivalent
/// ``AffTree`` instances based on [`schema`]. Then, this sequence is composed into a single
/// ``AffTree`` using [`AffTree::compose`]. In between each composition
/// step the tree is pruned using [`AffTree::infeasible_elimination`].
///
/// Behavior can be customized by providing an appropriate ``visitor``.
pub fn afftree_from_layers_generic<I, Estimator, Visitor>(
    dim: usize,
    layers: I,
    precondition: Option<AffTree<2>>,
    node_estimator: &mut Estimator,
    visitor: &mut Visitor,
) -> AffTree<2>
where
    I: IntoIterator,
    I::Item: Borrow<Layer>,
    Estimator: NodeEstimator,
    Visitor: DistillVisitor,
{
    let container = layers.into_iter().collect_vec();

    let (n_layers, n_nodes) = node_estimator.estimate_nodes(dim, 0, &container);

    let mut dd = if let Some(dd) = precondition {
        assert_eq!(dd.in_dim(), dim);
        dd
    } else {
        AffTree::<2>::with_capacity(dim, n_nodes)
    };

    dd.reserve(n_nodes);

    visitor.start_distill(dim, n_layers, n_nodes);

    let mut dim = dim;

    for layer in container.into_iter() {
        let layer = layer.borrow();
        let old_len = dd.len();
        visitor.start_layer(layer);
        match layer {
            Layer::Linear(aff) => {
                assert!(aff.indim() == dim);
                dd.apply_func(aff);
                dim = aff.outdim();
            }
            Layer::ReLU(row) => {
                dd.compose::<false>(&partial_ReLU(dim, *row));
                dd.infeasible_elimination();
            }
            Layer::HardTanh(row) => {
                dd.compose::<false>(&partial_hard_tanh(dim, *row, -1., 1.));
                dd.infeasible_elimination();
            }
            Layer::HardSigmoid(row) => {
                dd.compose::<false>(&partial_hard_sigmoid(dim, *row));
                dd.infeasible_elimination();
            }
            Layer::Argmax => {
                dd.compose::<true>(&argmax(dim));
            }
            Layer::ClassChar(clazz) => {
                dd.compose::<true>(&class_characterization(dim, *clazz));
            }
        }
        visitor.finish_layer(
            layer,
            dd.len() - old_len,
            dd.len() - dd.tree.num_terminals(),
            dd.tree.num_terminals(),
        );
    }
    visitor.finish_distill(dd.len() - dd.tree.num_terminals(), dd.tree.num_terminals());

    dd
}

pub fn read_layers<P: AsRef<Path>>(path: &P) -> Result<Vec<Layer>, ReadNpzError> {
    let file = File::open(path).map_err(ReadNpyError::from)?;
    let mut npz = NpzReader::new(file)?;

    let mut names = npz.names()?;
    names.sort_unstable();

    let pattern = Regex::new(r"^(\d+)\.(linear.weights|relu).npy$").unwrap();

    let mut layers = Vec::with_capacity(names.len());
    let mut dim: usize = 0;

    for layer_descr in names.iter() {
        let layer_name = pattern.captures(layer_descr);
        //    .expect(&format!("Layer description unknown: {}", layer_descr));
        if layer_name.is_none() {
            continue;
        }
        let layer_name = layer_name.unwrap();
        match layer_name.get(2).unwrap().as_str() {
            "relu" => {
                for idx in 0..dim {
                    layers.push(Layer::ReLU(idx));
                }
            }
            "hard_tanh" => {
                for idx in 0..dim {
                    layers.push(Layer::HardTanh(idx));
                }
            }
            "hard_sigmoid" => {
                for idx in 0..dim {
                    layers.push(Layer::HardSigmoid(idx));
                }
            }
            "linear.weights" => {
                let aff = AffFunc::from_mats(
                    npz.by_name(&format!(
                        "{}.linear.weights.npy",
                        layer_name.get(1).unwrap().as_str()
                    ))?,
                    npz.by_name(&format!(
                        "{}.linear.bias.npy",
                        layer_name.get(1).unwrap().as_str()
                    ))?,
                );
                dim = aff.outdim();
                layers.push(Layer::Linear(aff));
            }
            _ => unreachable!(),
        }
    }

    Ok(layers)
}

#[cfg(test)]
mod tests {

    use crate::distill::builder::{afftree_from_layers, read_layers};
    use approx::assert_relative_eq;
    use ndarray::arr1;
    use std::path::Path;

    #[test]
    pub fn test_read_npy() {
        let layers = read_layers(&Path::new("res/nn/ecoli.npz")).unwrap();

        let dd = afftree_from_layers(7, &layers, None);

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
