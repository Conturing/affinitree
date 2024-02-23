# affinitree

[![crates.io](https://img.shields.io/crates/v/affinitree.svg?logo=rust)](https://crates.io/crates/affinitree)
[![docs.rs](https://img.shields.io/badge/affinitree-darkseagreen?logo=rust&label=docs.rs)](https://docs.rs/affinitree/latest/affinitree)
[![github](https://img.shields.io/badge/affinitree-cadetblue?logo=github&label=GitHub)](https://github.com/Conturing/affinitree) 
[![build](https://github.com/Conturing/affinitree/actions/workflows/ci.yml/badge.svg)](https://github.com/Conturing/affinitree/actions) 

The ``affinitree`` crate provides data structures and algorithms to efficiently extract decision trees out of piece-wise linear neural networks.

## Features

Currently the following features are supported:
 - build a decision tree from a sequence of piece-wise linear layers (e.g., ReLU, leaky ReLU, hard tanh, hard sigmoid)
 - combine decision tree instances using composition
 - visualize decision trees using Graphviz's DOT language
 - optimize decision trees using infeasible path elimination
 - manually construct a decision tree to represent any piece-wise linear function (such as custom activation functions)

A short guide is provided below.

Please feel free to contribute new functionality!

## Using with Cargo

```toml
[dependencies]
affinitree = "0.21.0"
```

Supports Rust 1.64 and later.

## Technical Details

The crate is split into four parts:

1. *tree*: data structure and algorithms for decision trees
2. *linalg*: linear functions, polytopes, and linear programs
3. *pwl*: piece-wise linear functions stored as decision trees
4. *distill*: distillation of piece-wise linear neural networks into decision trees

This crate focuses on an efficient representation of piece-wise linear functions using decision trees.
The decision tree is implemented over an arena provided by the `slab` crate.
Elements of the tree have a unique index during their lifetime.
However, after deletion, the index can be reused.
The API of the tree is oriented at `petgraph`.

This crate requires basic linear algebra features like matrix storage and multiplication.
For that the crate `ndarray` is used.


## First Steps

To get started with affinitree, its best to first create a new `AffTree` instance.
`AffTree`s can represent any piece-wise linear function by storing them as a decision tree.
They are an essential part of this library and are used in many contexts.
To construct a basic `AffTree`, one can call one of the following constructors:
```rust
use affinitree::pwl::afftree::AffTree;

let dim: usize = 4;
// Crate a new AffTree instance representing the identity function with input dimension 4
let dd1 = AffTree::<2>::new(dim);
// Same as above, but also allocate space for 31 additional nodes in the tree
let dd2 = AffTree::<2>::with_capacity(dim, 32);
```

The resulting decision tree encodes simply the identity function $\R^{dim} \to \R^{dim}$.
Next, we want to update the decision tree.
For that, let us assume the following toy example:
We want to to introduce the hyperplane $x_1 - x_3 \leq 1$ as a discrimination rule to split the input space into two regions.
Let us encode such a layer by hand.

```rust
use ndarray::{arr1, arr2};
use affinitree::{aff, linalg::affine::AffFunc};

// Crate a new affine function
let func1 = AffFunc::from_mats(arr2(&[[1., 0., -1., 0.]]), arr1(&[1.]));
// Same as above, but using the aff macro for convenience
let func2 = aff!([[1., 0., -1., 0.]] + [1.]);

assert_eq!(func1, func2);
```

Now applying this function to our tree is straightforward.

```rust
dd1.apply_func(&func);
```

However, most use cases of neural networks include deeper architectures with non-linear activation functions.
To apply ReLU to our linear function, we first have to construct a decision tree that encodes the ReLU function as an `AffTree` instance.
Affinitree comes with a collection of predefined piece-wise linear functions, including ReLU.

```rust
use affinitree::distill::schema::ReLU;

let relu = ReLU(1);
dd.compose(&relu);
```

To construct deeper architectures, both methods (apply_func and compose) can be used in sequence.
Other piece-wise linear activation functions can be used in a similar fashion.
One only needs to encode the function in an `AffTree` instance, and then use the `compose` method to apply
the activation function to the tree.

Finally, as the manual construction of `AffTree` instances can get cumbersome for larger networks, a high-level convenience function is provided.
This function only requires a list of the layers of the neural network.
Such lists can be read from the file system using the `.npz` format or specified explicitly.
For its test cases `affinitree` comes with a handful of pre-trained networks stored in this format.
For example, the mnist.npz file contains a pre-trained network over the first seven principal components of the MNIST data set with the layer structure 7-5-5-5-10.

```rust
use affinitree::distill::builder::{read_layers, afftree_from_layers};

// Load a sequence of pretrained layers from a numpy file
let layers = read_layers(&"res/nn/mnist-5-5.npz").unwrap();
// Distill the sequence of layers with input dimension 7 into an AffTree without a precondition
let dd = afftree_from_layers(7, &layers, None);
```

For additional examples have a look at the [test cases](tests).

## License

Copyright 2022–2024 `affinitree` developers.

Conceived and developed by Maximilian Schlüter, Jan Feider, and Gerrit Nolte.

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE), or the [MIT
license](LICENSE-MIT), at your option. You may not use this project except in
compliance with those terms.

## Contributing

Please feel free to create issues, fork the project or submit pull requests.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## Conduct

Please follow the [Rust Code of Conduct].

[Rust Code of Conduct]: https://www.rust-lang.org/conduct.html