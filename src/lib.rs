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

/*!
Faithful neural network distillation.

This crate provides methods to [distill](https://en.wikipedia.org/wiki/Knowledge_distillation)
faithful decision trees out of piece-wise linear neural networks.
The resulting decision tree stores in a compact manner the linear regions of the distilled
network. The term faithful refers to the property that the resulting tree is not an
approximation but a complete semantic replica of the network (this property is also referred to
as semantic-preserving or fidelitous).

`affinitree` supports the following operations:
 - distill a decision tree from a sequence of linear and ReLU layers
 - combine decision tree instances using composition
 - visualize a decision tree using Graphviz's DOT language
 - optimize decision trees using infeasible path elimination
 - manually construct a decision tree to represent any piece-wise linear function (such as custom activation functions)

A corner stone of this crate is a data structure called [`AffTree`](crate::core::afftree::AffTree) that can represent any
(continuous or non-continuous) [piece-wise linear function](https://en.wikipedia.org/wiki/Piecewise_linear_function).
This structure is based on oblique decision trees and BSP trees.

# Quick Start
**[`AffTree`s](crate::core::afftree::AffTree)** can be directly constructed from a sequence of linear functions and ReLUs.
As `affinitree` works on pretrained neural networks, it was designed to be compatible with
many neural network libraries. Therefore, the encoding of layers is based on a universal
format introduced by `numpy` to encode matrices.

For testing purposes some pretrained networks are provided in the resource folder (res/nn/).
The following example loads a pretrained network on MNIST with 4 hidden layers à 5 neurons
and distills it into an `AffTree` called dd.
```rust
use affinitree::core::builder::{read_layers, afftree_from_layers};

// load a sequence of pretrained layers from a numpy file
let layers = read_layers(&"res/nn/mnist-5-5.npz").unwrap();
// distill the sequence of layers with input dimension 7 into an AffTree
let dd = afftree_from_layers(7, &layers);
```

Alternatively, linear functions can be directly encoded in Rust using the [`AffFunc`](crate::linalg::affine::AffFunc) struct.
Internally, this struct uses `ndarray` to store the matrices.
The following example encodes a pretrained network on the IRIS data set with 1 hidden layer
á 4 neurons using the `aff!` macro.
```rust
use affinitree::{poly, aff};
use affinitree::core::builder::{Layer::{Linear, ReLU}, afftree_from_layers};
use affinitree::linalg::affine::AffFunc;

let l0 = aff!([[-0.09177965670824051, 0.8253487348556519, -0.8163803815841675, -0.9800696969032288],
    [0.5591527223587036, -0.3632337152957916, 1.3144720792770386, 0.2468724548816681],
    [0.18317964673042297, -0.3006826341152191, 0.1607706993818283, 1.8758670091629028],
    [0.6726926565170288, -0.3332176208496094, -0.9476901888847351, -0.20959123969078064]]
    + [0.6692222356796265, 1.2492079734802246, -0.49917441606521606, 0.6329305171966553]);
let l1 = aff!([[1.1643257141113281, -0.7534151673316956, 0.17711225152015686, -1.1624157428741455],
    [-1.2407400608062744, 0.9271628856658936, -1.2888133525848389, 0.23608165979385376],
    [-1.1691803932189941, 0.8739460110664368, 1.1971392631530762, -1.7638847827911377]]
    + [-0.37897831201553345, 0.9170833826065063, -0.7026672959327698]);

let layers = vec![Linear(l0), ReLU, Linear(l1)];
let dd = afftree_from_layers(4, &layers);
```

# Neural Network Distillation
Many of todays Neural Networks are [piece-wise linear (PWL) functions](https://en.wikipedia.org/wiki/Piecewise_linear_function).
In essence, this means that the input space of such networks can be divided into non-overlapping
regions such that the network is linear in each region. This structure emerges from the ReLU
activation function, which is itself piece-wise linear: For inputs greater than zero, it corresponds
to the linear identity function, and for inputs smaller than or equal to zero, it corresponds to the
(linear) constant function that maps every input to zero.

It was observed independently in multiple use cases that the structure introduced by the ReLU
activation function can be used to *decompose* PWL networks into a set of linear classifiers,
each of which corresponds to exactly one (sub-)input region. Thereby, problems like explainability,
expressibility, and robustness of neural networks can be reduced to the simpler cases concerning
only linear classifiers.

`Affinitree` provides a central data structure to store the results of such decomposition processes.
This data structure is optimized for efficient construction, which is necessary as the number of
regions grows exponentially in the number of neurons of the modelled neural network.
Further, the library leverages the layer structure of neural networks for a concise and modular API
that is easy to follow and allows straightforward extension to individual use cases.

# Customization
This library is built with customization in mind. This can be achieved generically
based on central algebraic properties of decision trees. From these mathematical properties
one can derive an essential lifiting pattern that allows to implement a wealth of other functions.
Lifting allows to lift any binary function over affine functions onto `AffTree` level. This can be
used to construct complex `AffTree` instances from simple ones.

The lifiting capabilities are already used internally by many functions, for example by composition,
addition, subtraction, scalar multiplication, and Cartesian product.
*/

#![warn(
    missing_debug_implementations,
    //missing_docs,
    rust_2021_compatibility,
    // unreachable_pub
)]

// #[cfg(doctest)]
// doc_comment::doctest!("../README.md");

pub mod core;
pub mod linalg;
pub mod tree;
