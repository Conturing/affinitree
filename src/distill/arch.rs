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

//! A collection of high-level methods to distill AffTree instances out of neural networks

use std::fmt::Display;
use std::mem;

use itertools::Itertools;
use thiserror::Error;

use super::builder::Layer;
use crate::linalg::affine::AffFunc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorShape {
    Flat { in_dim: usize },
}

impl TensorShape {
    /// Returns the number of components this shape supports.
    pub fn max_dim(&self) -> usize {
        match self {
            TensorShape::Flat { in_dim } => *in_dim,
        }
    }

    /// Checks if this given number of elements is compatible with this shape.
    pub fn compatible_dim(&self, dim: usize) -> Result<(), ShapeError> {
        match self {
            TensorShape::Flat { in_dim } => {
                if *in_dim == dim {
                    Ok(())
                } else {
                    Err(ShapeError::Dim {
                        expected: *in_dim,
                        got: dim,
                    })
                }
            }
        }
    }

    /// Validates if the given index is within bounds for this shape.
    pub fn valid_index(&self, idx: usize) -> Result<(), ShapeError> {
        match self {
            TensorShape::Flat { in_dim } => {
                if idx < *in_dim {
                    Ok(())
                } else {
                    Err(ShapeError::Index {
                        index: idx,
                        len: *in_dim,
                    })
                }
            }
        }
    }
}

impl Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorShape::Flat { in_dim } => {
                write!(f, "[{}]", in_dim)
            }
        }
    }
}

#[derive(Error, Clone, Debug)]

pub enum ShapeError {
    #[error("Shape mismatch: expected {expected}, got {got}")]
    Dim { expected: usize, got: usize },
    #[error("Index {index} is out of bounds (length: {len})")]
    Index { index: usize, len: usize },
    #[error("Input type is not compatible")]
    Type,
}

#[derive(Clone, Debug)]
pub struct Architecture {
    pub input_shape: TensorShape,
    pub current_shape: TensorShape,
    /// All queued up operators together with the shape after the operator
    pub operators: Vec<(Layer, TensorShape)>,
}

impl Architecture {
    /// Creates a new architecture with the specified initial shape.
    pub fn new(input_shape: TensorShape) -> Architecture {
        Architecture {
            input_shape,
            current_shape: input_shape,
            operators: Vec::new(),
        }
    }

    /// Adds a linear layer to this architecture.
    pub fn linear(&mut self, aff: AffFunc) -> Result<(), ShapeError> {
        self.current_shape.compatible_dim(aff.indim())?;
        let in_dim = aff.outdim();
        self.current_shape = TensorShape::Flat { in_dim };
        self.operators
            .push((Layer::Linear(aff), self.current_shape));

        Ok(())
    }

    /// Adds a ReLU activation for the specified neuron.
    pub fn partial_relu(&mut self, idx: usize) -> Result<(), ShapeError> {
        self.current_shape.valid_index(idx)?;
        self.operators.push((Layer::ReLU(idx), self.current_shape));
        Ok(())
    }

    /// Adds a ReLU activation layer to this architecture.
    pub fn relu(&mut self) -> Result<(), ShapeError> {
        for idx in 0..self.current_shape.max_dim() {
            self.partial_relu(idx)?;
        }
        Ok(())
    }

    /// Adds a Leaky ReLU activation for the specified neuron.
    pub fn partial_leaky_relu(&mut self, idx: usize, alpha: f64) -> Result<(), ShapeError> {
        self.current_shape.valid_index(idx)?;
        self.operators
            .push((Layer::LeakyReLU(idx, alpha), self.current_shape));
        Ok(())
    }

    /// Adds a Leaky ReLU activation layer to this architecture.
    pub fn leaky_relu(&mut self, alpha: f64) -> Result<(), ShapeError> {
        for idx in 0..self.current_shape.max_dim() {
            self.partial_leaky_relu(idx, alpha)?;
        }
        Ok(())
    }

    /// Adds a hard tanh activation for the specified neuron.
    pub fn partial_hard_tanh(&mut self, idx: usize) -> Result<(), ShapeError> {
        self.current_shape.valid_index(idx)?;
        self.operators
            .push((Layer::HardTanh(idx), self.current_shape));
        Ok(())
    }

    /// Adds a hard tanh activation layer to this architecture.
    pub fn hard_tanh(&mut self) -> Result<(), ShapeError> {
        for idx in 0..self.current_shape.max_dim() {
            self.partial_hard_tanh(idx)?;
        }
        Ok(())
    }

    /// Adds a hard sigmoid activation for the specified neuron.
    pub fn partial_hard_sigmoid(&mut self, idx: usize) -> Result<(), ShapeError> {
        self.current_shape.valid_index(idx)?;
        self.operators
            .push((Layer::HardSigmoid(idx), self.current_shape));
        Ok(())
    }

    /// Adds a hard sigmoid activation layer to this architecture.
    pub fn hard_sigmoid(&mut self) -> Result<(), ShapeError> {
        for idx in 0..self.current_shape.max_dim() {
            self.partial_hard_sigmoid(idx)?;
        }
        Ok(())
    }

    /// Adds an argmax layer to this architecture.
    pub fn argmax(&mut self) -> Result<(), ShapeError> {
        self.operators.push((Layer::Argmax, self.current_shape));
        Ok(())
    }

    /// Returns the layers of this architecture.
    #[rustfmt::skip]
    pub fn operators(&self) -> impl Iterator<Item = &Layer> {
        self.operators.iter()
            .map(|(op, _)| op)
    }

    /// Extracts a subnetwork for an arbitrary range of layers.
    pub fn extract_range(&self, start: usize, end: usize) -> Result<Architecture, ShapeError> {
        if start >= end || end > self.operators.len() {
            return Err(ShapeError::Index {
                index: end,
                len: self.operators.len(),
            });
        }

        let iter = self.operators.iter();

        let (iter_skip, input_shape) = if start == 0 {
            (iter.skip(0), self.input_shape)
        } else {
            let mut iter_skip = iter.skip(start - 1);
            let (_, input_shape) = iter_skip.next().ok_or(ShapeError::Index {
                index: start,
                len: self.operators.len(),
            })?;
            (iter_skip, *input_shape)
        };

        let mut sub_arch = Architecture::new(input_shape);
        sub_arch.operators.reserve(end - start);

        for (layer, shape) in iter_skip.take(end - start) {
            sub_arch.operators.push((layer.clone(), *shape));
            sub_arch.current_shape = *shape;
        }

        Ok(sub_arch)
    }
}

impl Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{:<30}   {:<5}   {:<16}", "Layer", "Count", "Shape")?;
        writeln!(f, "{:=<30}==={:=>5}==={:=>16}", "=", "=", "=")?;

        let chunks = self
            .operators
            .iter()
            .chunk_by(|(layer, _)| mem::discriminant(layer));
        for (_, mut layers) in chunks.into_iter() {
            // writeln!(f, "{}", shape)?;

            let (layer, shape) = layers.next().unwrap();

            let descr = match layer {
                Layer::Linear(aff) => {
                    format!("Linear ({}x{})", aff.indim(), aff.outdim())
                }
                Layer::ReLU(idx) => {
                    format!("ReLU (idx={})", idx)
                }
                Layer::LeakyReLU(idx, alpha) => {
                    format!("LeakyReLU (alpha={} idx={})", alpha, idx)
                }
                Layer::HardTanh(idx) => {
                    format!("HardTanh (idx={})", idx)
                }
                Layer::HardSigmoid(idx) => {
                    format!("HardSigmoid (idx={})", idx)
                }
                Layer::Argmax => "Argmax".to_string(),
                Layer::ClassChar(class) => {
                    format!("ClassChar (class={})", class)
                }
            };
            let count = layers.count() + 1;

            writeln!(f, "{:<30}   {:>5}   {:>16}", descr, count, shape)?;
        }
        Ok(())
    }
}

impl IntoIterator for Architecture {
    type IntoIter =
        std::iter::Map<std::vec::IntoIter<(Layer, TensorShape)>, fn((Layer, TensorShape)) -> Layer>;
    type Item = Layer;

    fn into_iter(self) -> Self::IntoIter {
        self.operators.into_iter().map(|(op, _)| op)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::*;

    #[test]
    fn test_init() {
        let mut arch = Architecture::new(TensorShape::Flat { in_dim: 8 });
        arch.linear(AffFunc::from_mats(Array2::ones((4, 8)), Array1::zeros(4)))
            .unwrap();

        assert_eq!(arch.operators.len(), 1);
        assert_eq!(arch.current_shape, TensorShape::Flat { in_dim: 4 });
    }

    #[test]
    fn test_shape_error() {
        let mut arch = Architecture::new(TensorShape::Flat { in_dim: 4 });
        assert!(
            arch.linear(AffFunc::from_mats(Array2::ones((4, 8)), Array1::zeros(4)))
                .is_err()
        );
    }

    #[test]
    fn test_relu_layer() {
        let mut arch = Architecture::new(TensorShape::Flat { in_dim: 4 });
        arch.relu().unwrap();

        assert_eq!(arch.operators.len(), 4);
    }

    #[test]
    fn test_extract_range_start() {
        let mut arch = Architecture::new(TensorShape::Flat { in_dim: 8 });
        arch.linear(AffFunc::from_mats(Array2::ones((4, 8)), Array1::zeros(4)))
            .unwrap();
        arch.relu().unwrap();
        arch.linear(AffFunc::from_mats(Array2::ones((2, 4)), Array1::zeros(2)))
            .unwrap();
        arch.relu().unwrap();

        let subnetwork = arch.extract_range(0, 7).unwrap();

        assert_eq!(subnetwork.operators.len(), 7);
        assert_eq!(subnetwork.input_shape, TensorShape::Flat { in_dim: 8 });
        assert_eq!(subnetwork.current_shape, TensorShape::Flat { in_dim: 2 });
    }

    #[test]
    fn test_extract_range() {
        let mut arch = Architecture::new(TensorShape::Flat { in_dim: 8 });
        arch.linear(AffFunc::from_mats(Array2::ones((4, 8)), Array1::zeros(4)))
            .unwrap();
        arch.relu().unwrap();
        arch.linear(AffFunc::from_mats(Array2::ones((2, 4)), Array1::zeros(2)))
            .unwrap();
        arch.relu().unwrap();

        let subnetwork = arch.extract_range(2, 7).unwrap();

        assert_eq!(subnetwork.operators.len(), 5);
        assert_eq!(subnetwork.input_shape, TensorShape::Flat { in_dim: 4 });
        assert_eq!(subnetwork.current_shape, TensorShape::Flat { in_dim: 2 });
    }
}
