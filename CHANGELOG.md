# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.22.0] 2024-06-28

### Added

- Docstrings to ``builder.rs``, ``schema.rs``, ``affine.rs``, ``afftree.rs``, ``iter.rs``, ``node.rs``, ``graph.rs``, ``iter.rs``
- Improved formatting of ``AffFuncBase``, configurable through ``FormatOptions``
- Tests to ``builder.rs``, ``affine.rs``, ``afftree.rs``
- Section on development tools to ``README.md``
- LeakyReLU to ``Layer`` enum
- Compilation error if both feature flags ``minilp`` and ``highs`` are supplied
- License headers updated and added where missing
- Dependencies to ``Cargo.toml``
- Struct ``Dot`` for simpler formatting to graphviz's dot language
- Struct ``PerformanceCounter`` to keep track of infeasible elimination
- Struct ``Bfs``which implements breath-first traversal
- Trait ``CompositionSchema`` for implementing other compositional operations over ``AffTree``
- Trait ``CompositionVisitor``for tracing composition steps 
- Trait ``TraversalMut`` which allows modifications to the tree while iterating, but which can also be downgraded to a normal iterator
- Trait implementations ``AbsDiffEq`` and ``RelativeEq`` for ``AffFuncBase``
- Trait implementations ``Add``, ``Sub``, ``Mul``, ``Div``, and ``Rem`` for ``AffFuncBase`` and ``&AffFuncBase`` (covering owned data and views)
- Trait implementations ``Add``, ``Sub``, ``Mul``, and ``Div`` for ``AffTree`` and ``&AffTree``
- Trait implementations ``Add``, ``Sub``, ``Mul``, and ``Div`` for ``AffTree`` and ``&AffTree`` where the second argument is of type ``AffFuncBase`` (scalar / element-wise operations)
- Methods ``AffFuncBase::{from_row_iter, remove_zero_rows, remove_zero_columns, display_with}``
- Methods ``PolytopeG::{axis_bounds, distances_raw, apply_pre, apply_post, rotate}``
- Method ``AffTree::reduce`` for removing redundant decisions of the bottom of the tree (cascading upwards as long as possible)

### Changed

- Extract infeasible elimination and composition from ``afftree.rs`` into ``impl_infeasible_elim.rs`` and ``impl_composition``
- AffFuncBase now accepts any basis type for its underlying arrays as long as it implements ``num_traits::float::Float``
- Tree iterators now all implement ``TraversalMut`` 
- Use nightly toolchain for ``rust fmt``
- Bound maximal space reserved in advance for nodes in distillation
- Tests in ``polyhedron.rs`` reorganized an checked against scipy
- Renamed ``display.rs`` to ``impl_affineformat.rs``
- Rename ``AffFuncBase::get_marix`` to ``matrix_view``
- Rename ``AffFuncBase::get_bias`` to ``bias_view``
- Method ``PolytopeG::convert_to`` takes ownership of self
- Method ``AffTree::from_poly`` now accepts an optional node for paths not following the given polytope
- Method ``AffTree::evaluate_to_terminal`` renamed to ``find_terminal``, max_iter removed, panic behavior changed
- Method ``AffTree::evaluate_node`` renamed to ``evaluate_decision``

### Fixed

- Input dim of distillation now acts on precondition if one is supplied

### Security


## [0.21.1] 2024-03-05

### Added

- Docstrings to methods in ``afftree.rs`` and ``graph.rs``
- License information for binaries (including transitive dependencies) using ``cargo about``

### Removed

- ``Cargo.lock`` from versioning
- Unused dependencies from ``Cargo.toml``


## [0.21.0] 2024-02-23

### Added

- HardTanh and HardSigmoid activation functions to ``Layer``
- Highs LP solver
- Feature flags ``minilp`` and ``highs`` to globally switch LP solvers
- Mirror heuristic to infeasible elimination
- Enum ``NodeState`` to record the feasibility status of every node in ``AffTree`` 
- Visitors to the distillation process
- Benchmark for infeasibility
- Dependencies to ``Cargo.toml``
- Methods ``Tree::{num_nodes, terminals_mut,describe}``
- Method ``DfsPreIter::with_root``

### Changed

- Module ``core`` is split into ``distill`` and ``pwl``
- Infeasible elimination now requires only a single depth-first search
- Rust fmt is reduced to stable features
- Activation functions for distillation are now specified per neuron instead of per layer
- Readme updated and improved
- Method ``AffTree::from_precondition`` renamed to ``from_poly``
- Method ``AffTree::update_node`` no longer resets solution cache
- Method ``AffTree::is_edge_feasible`` no longer accepts a precondition
- Methods ``AffFuncBase::{row, row_iter}`` now return a view
- Method ``AffFuncBase::contains`` allows an imprecision of up to 1e-8
- Methods ``AffFuncBase::{distance_raw, distance, contains}`` can now also be called on views
- Method ``builder::afftree_from_layers`` now accepts a precondition 

### Deprecated

- Method ``AffTree::is_edge_feasible``


## [0.20.1] 2024-02-07

### Added

- Activation functions ``partial_leaky_ReLU``, ``partial_hard_tanh``, ``partial_hard_shrink``, ``partial_hard_sigmoid``, and ``partial_threshold`` to ``schema``
- Badges to ``README.md``
- Docstrings to ``afftree.rs``, ``schema.rs``, ``affine.rs``
- Integration tests based on the absolute value function (``abs.rs``), a DNN trained over Iris (``iris.rs``), and a DNN trained over MNIST (``mnist.rs``)
- Methods ``AffTree::{from_slice, is_empty, depth, reserve}``
- Methods ``AffFuncBase::{rotation, uniform_scaling, scaling, slice, translation}``
- Methods ``Tree::{reserve, try_remove_child, remove_all_descendants, contains}``

### Changed

- Console output of inequalities and linear combinations

### Fixed

- Method ``Tree::add_child_node`` now checks if parent exists in the tree before adding the node, as it could receive the node id of a missing parent


## [0.20.0] 2023-09-29

Initial release.