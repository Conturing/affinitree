<div align="center">
  <a href="https://crates.io/crates/affinitree">
    <img src="https://img.shields.io/crates/v/affinitree.svg"/>
  </a>
  <a href="https://github.com/Conturing/affinitree/actions">
    <img src="https://github.com/Conturing/affinitree/actions/workflows/ci.yml/badge.svg"/>
  </a>
</div>

<p align="center">
  <b>Documentation</b>:
  <a href="https://docs.rs/affinitree/latest/affinitree/">Rust</a>
</p>

# affinitree

This crate provides methods to extract decision trees out of piece-wise linear neural networks.

Currently the following features are supported:
 - build a decision tree from a sequence of linear and ReLU layers
 - combine decision tree instances using composition
 - visualize a decision tree using Graphviz's DOT language
 - optimize decision trees using infeasible path elimination
 - manually construct a decision tree to represent any piece-wise linear function (such as custom activation functions)

A short guide is provided below.

Please feel free to contribute new functionality!

## Using with Cargo

```toml
[dependencies]
affinitree = "0.20.0"
```

Supports Rust 1.64 and later.

## Technical Details

This crate focuses on an efficient representation of piece-wise linear functions using decision trees.
The decision tree is implemented over an arena provided by the `slab` crate.
It has a compile time branching factor `K` (in most cases a binary tree is sufficient, i.e., K=2).
Elements of the tree have a unique index during their lifetime.
However, after deletion, the index can be reused.
The API of the tree is oriented at `petgraph`.

This crate requires basic linear algebra features like matrix storage and multiplication.
For that the crate `ndarray` is used.


## First Steps

To get started, we must create a new `AffTree` instances.
This is the core data structure of this crate.
It stores piece-wise linear functions, like that of ReLU neural networks, using a decision tree.
Decisions are used to partition the input space according to the represented function.
Terminals store the actual linear functions using matrices.

An empty `AffTree` can be constructed by calling on of the constructors:
```rust
use affinitree::core::afftree::AffTree;

let dd1 = AffTree::<2>::new();
let dd2 = AffTree::<2>::with_capacity(32);
```

Next, we want to update the decision tree.
For simple data sets like `iris` it is sometimes sufficient to train neural networks consisting of only a single linear layer.
As a toy example we use the discrimination rule $x_1 \leq 1$.
Let us encode such a layer by hand.

```rust
use ndarray::{arr1, arr2};
use affinitree::linalg::affine::AffFunc;

let func = AffFunc::from_mats(arr2(&[[1., 0., 0., 0.]]), arr1(&[1.]));
```

Now applying this function to our empty tree is straightforward.

```rust
dd1.apply_func(&func);
```

However, most use cases of neural networks include deeper architectures with non-linear activation functions.
To apply ReLU to our linear function, we first have to construct a decision tree that encodes the ReLU function as an `AffTree` instance.
Luckily, these are already predefined.

```rust
use affinitree::core::schema::ReLU;

let relu = ReLU(1);
dd.compose(&relu);
```

To construct deeper architectures, both methods can be used in sequence.
Other piece-wise linear activation functions can be used in a similar fashion.
One only needs to encode the function in an `AffTree` instance, and then use the `compose` method to apply
the activation function to the tree.

Finally, as the manual construction of `AffTree` instances can get cumbersome for larger networks, a high-level convenience function is provided.
This function only requires a list of the layers of the neural network.
Such lists can be read from the file system using the `.npz` format or specified explicitly.
For its test cases `affinitree` comes with a handful of pre-trained networks stored in this format.
For example, the mnist.npz file contains a pre-trained network over the first seven principal components of the MNIST data set with the layer structure 7-5-5-5-10.

```rust
use affinitree::core::builder::{read_layers, afftree_from_layers};

// load a sequence of pretrained layers from a numpy file
let layers = read_layers(&"res/nn/mnist-5-5.npz").unwrap();
// distill the sequence of layers with input dimension 7 into an AffTree
let dd = afftree_from_layers(7, &layers);
```

For additional examples have a look at the [test cases](tests).

## License

Copyright 2022–2023 `affinitree` developers.

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