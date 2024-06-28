//   Copyright 2024 affinitree developers
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

//! Central data structure to store piece-wise linear functions

use core::fmt;
use std::cell::RefCell;
use std::fmt::Display;
use std::mem;

use itertools::Itertools;
use ndarray::{concatenate, Array1, Array2, Axis};

use super::iter::PolyhedraIter;
use super::node::{write_children, AffContent};
use crate::linalg::affine::{AffFunc, Polytope};
use crate::pwl::iter::PolyhedraGen;
use crate::pwl::node::{AffNode, NodeState};
use crate::tree::graph::{Label, Tree, TreeIndex, TreeNode};

/// A specialized decision tree to hold [piece-wise linear functions](https://en.wikipedia.org/wiki/Piecewise_linear_function).
///
/// This structure is based on an oblique decision tree:
/// * Its inner nodes form a decision structure that partitions the input space similar to a [BSP tree](https://en.wikipedia.org/wiki/Binary_space_partitioning) into convex polyhedral regions.
/// * Its terminal nodes associate with each such region a linear function.
///
/// Technically, each decision node divides the space into two halfspace which are separated by
/// a [hyperplane](https://en.wikipedia.org/wiki/Hyperplane). The outgoing edges of decision nodes
/// have labels that uniquely associate the edge with a halfspace. A path over multiple edges then
/// corresponds to the intersection of the associated halfspaces of each edge.
///
/// To encode a given piece-wise linear function as an AffTree, one first has to derive the decision
/// structure based on the regions of the function. That is, each path from root to a terminal
/// corresponds to exactly one region of the function. Then, the terminal stores the respective
/// linear function that is active in this region.
///
/// # Example
///
/// The piece-wise linear function f(x, y) = |max{x,y}| is represented as the `AffTree`
/// <p align="center">
///   <img alt="example afftree" height="200" src="afftree_example.svg"/>
/// </p>
///
/// This tree is constructed in `affinitree` as follows:
/// ```rust
/// use affinitree::{aff, poly, pwl::afftree::AffTree, pwl::dot::Dot, linalg::affine::PolyRepr};
///
/// // construct a new AffTree instance with the given inequality as its root.
/// let mut dd = AffTree::<2>::from_aff(poly!([[1, 1]] < [0]).convert_to(PolyRepr::MatrixBiasGeqZero)); // node index 0
///
/// // add two additional decision nodes from inequalities
/// dd.add_decision(0, 0, poly!([[-1, 1]] < [0]).convert_to(PolyRepr::MatrixBiasGeqZero)); // node index 1
/// dd.add_decision(0, 1, poly!([[-1, 1]] < [0]).convert_to(PolyRepr::MatrixBiasGeqZero)); // node index 2
///
/// // add the terminal nodes with the given functions
/// // first argument is the parent node, second the label (true / false), and third the affine function
/// dd.add_terminal(1, 0, aff!([[0, 1]] + [0]));
/// dd.add_terminal(1, 1, aff!([[1, 0]] + [0]));
/// dd.add_terminal(2, 0, aff!([[-1, 0]] + [0]));
/// dd.add_terminal(2, 1, aff!([[0, -1]] + [0]));
///
/// // export the tree into the DOT format of graphviz
/// println!("{}", Dot::from(&dd));
/// // digraph dd {
/// // bgcolor=transparent;
/// // concentrate=true;
/// // margin=0;
/// // n0 [label=" $0 1.00 +$1 1.00 <= 0.00", shape=box];
/// // n1 [label="−$0 1.00 +$1 1.00 <= 0.00", shape=box];
/// // n2 [label="−$0 1.00 +$1 1.00 <= 0.00", shape=box];
/// // n3 [label=" $0 0.00 +$1 1.00 + 0.00", shape=ellipse];
/// // n4 [label=" $0 1.00 +$1 0.00 + 0.00", shape=ellipse];
/// // n5 [label="−$0 1.00 +$1 0.00 + 0.00", shape=ellipse];
/// // n6 [label=" $0 0.00 −$1 1.00 + 0.00", shape=ellipse];
/// // n0 -> n1 [label=0, style=dashed];
/// // n0 -> n2 [label=1, style=solid];
/// // n1 -> n3 [label=0, style=dashed];
/// // n1 -> n4 [label=1, style=solid];
/// // n2 -> n5 [label=0, style=dashed];
/// // n2 -> n6 [label=1, style=solid];
/// // }
/// ```
///
/// # Technical Overview
/// `AffTree`s are implemented over an arena provided by the [`slab`] crate.
/// They have a compile time branching factor `K` (in most cases a binary tree is sufficient, i.e., K=2).
/// Each node of the tree has a unique index during its lifetime.
///
/// # Composition
/// `AffTree`s allow a modular construction by composition. The semantics of composition
/// follow directly from the mathematical definition of [function composition](https://en.wikipedia.org/wiki/Function_composition).
/// Roughly speaking, the composition of two AffTree instances corresponds to an sequential evaluation of the two trees, as demonstrated by the following example.
///
/// ```rust
/// use affinitree::{aff, pwl::afftree::AffTree};
/// use ndarray::arr1;
///
/// let mut tree0 = AffTree::<2>::from_aff(aff!([[1., 0.]] + [2.]));
/// tree0.add_child_node(0, 0, aff!([[2, 0], [0, 2]] + [1, 0]));
/// tree0.add_child_node(0, 1, aff!([[2, 0], [0, 2]] + [0, 1]));
///
/// let mut tree1 = AffTree::<2>::from_aff(aff!([[-0.5, 0.]] + [-1.]));
/// tree1.add_child_node(0, 0, aff!([[3, 0], [0, 3]] + [5, 0]));
/// tree1.add_child_node(0, 1, aff!([[-3, 0], [0, -3]] + [0, 5]));
///
/// let mut comp = tree0.clone();
/// comp.compose::<false>(&tree1);
///
/// // the sequential evaluation of tree0 and tree1 on the input vector (2, -7)
/// // yields the same result as evaluating the composition tree
/// assert_eq!(
///     tree1.evaluate(&tree0.evaluate(&arr1(&[2., -7.])).unwrap()).unwrap(),
///     comp.evaluate(&arr1(&[2., -7.])).unwrap()
/// );
/// ```
///
/// # Infeasible Path Elimination
/// In an [AffTree], the (linear) predicates along a path can exhibit semantic dependencies.
/// For example, when considering the path conditions "x > 5, y < 2, x < 2" one notices
/// that there cannot be a possible assignment to x and y that satisfies all three conditions
/// at the same time. In that case we say the path is *infeasible*.
///
/// In general, a path in an [AffTree] is infeasible when the polytope defined by the conjunction
/// of all path conditions is the empty set. This can be checked with LP solvers.
/// As infeasible paths cannot be taken by any input, they can be safely removed without altering
/// the represented piece-wise linear function.
///
/// ```rust
/// use affinitree::{aff, pwl::afftree::AffTree};
/// use ndarray::arr1;
///
/// let mut dd = AffTree::<2>::from_aff(aff!([[1, 0]] + [-5]));
/// dd.add_child_node(0, 1, aff!([[0, -1]] + [2]));
/// dd.add_child_node(1, 1, aff!([[-1, 0]] + [2]));
/// // feasible
/// dd.add_child_node(2, 0, aff!([[0, 0]] + [0]));
/// // infeasible
/// dd.add_child_node(2, 1, aff!([[0, 0]] + [1]));
///
/// dd.infeasible_elimination();
/// println!("{}", &dd);
///
/// assert!(dd.tree.contains(3));
/// assert!(!dd.tree.contains(4));
/// ```
///
/// # Arithmetic Operators
/// A wide variety of arithmetic operators are defined for piece-wise linear functions.
/// These are also available for [AffTree]s based on a general lifting principle for
/// decision trees. Concretely, [AffTree]s support addition, subtraction, multiplication,
/// division, and remainder (with inline operators +, -, *, /, %).
///
/// Other operators can be defined by implementing the [CompositionSchema] trait.
///
/// # Reduction
/// A decision is redundant whenever all its children are (semantically) equivalent.
/// Without enforcing some normal form, it is hard to detect semantic equivalence.
/// However, one particular form can be computed linearly, by checking if the all
/// children are syntactically equivalent. By performing a bottom-up sweep, [AffTree::reduce]
/// eliminates any decision that whose terminal nodes are equal.
#[derive(Clone)]
pub struct AffTree<const K: usize> {
    pub tree: Tree<AffContent, K>,
    pub in_dim: usize,
    pub(super) polytope_cache: RefCell<Vec<Polytope>>,
}

impl<const K: usize> AffTree<K> {
    /// Creates an AffTree instance which represents the identity function for the given ``dim``.
    #[inline]
    pub fn new(dim: usize) -> AffTree<K> {
        Self::with_capacity(dim, 0)
    }

    /// Creates an AffTree instance which represents the identity function for the given ``dim``.
    /// Allocates space for as many nodes as specified by ``capacity`` (minimum 1).
    #[inline]
    pub fn with_capacity(dim: usize, capacity: usize) -> AffTree<K> {
        AffTree {
            tree: Tree::with_root(AffContent::new(AffFunc::identity(dim)), capacity),
            in_dim: dim,
            polytope_cache: RefCell::new(Vec::new()),
        }
    }

    /// Creates an AffTree instance which represents the given affine ``func``.
    #[inline]
    pub fn from_aff(func: AffFunc) -> AffTree<K> {
        AffTree {
            in_dim: func.indim(),
            tree: Tree::with_root(AffContent::new(func), 1),
            polytope_cache: RefCell::new(Vec::new()),
        }
    }

    /// Crates an AffTree instance from the given polytope ``poly``.
    /// The resulting AffTree is a partial function that only accepts inputs inside
    /// the given ``poly``, which are then mapped using the given ``func_true``.
    /// Inputs outside of ``poly`` are undefined when ``func_false`` is ``None``,
    /// otherwise they are mapped by ``func_false``.
    ///
    /// In this capacity it can be used as a precondition.
    pub fn from_poly(
        poly: Polytope,
        func_true: AffFunc,
        func_false: Option<&AffFunc>,
    ) -> AffTree<K> {
        assert_eq!(poly.indim(), func_true.indim());
        assert!(poly.n_constraints() > 0);
        if let Some(aff_false) = func_false {
            assert_eq!(poly.indim(), aff_false.indim());
        }

        let mut tree = Self::with_capacity(func_true.indim(), poly.n_constraints() + 1);
        let mut iter = poly.row_iter();
        let mut parent = tree.tree.get_root_idx();
        let mut aff = iter.next().unwrap().as_function().to_owned();
        aff.mat = -aff.mat;
        tree.tree.node_value_mut(parent).unwrap().aff = aff;

        for decision in iter {
            if let Some(aff_false) = func_false {
                tree.add_child_node(parent, 0, aff_false.clone());
            }
            let mut aff = decision.as_function().to_owned();
            aff.mat = -aff.mat;
            parent = tree.add_child_node(parent, 1, aff);
        }
        if let Some(aff_false) = func_false {
            tree.add_child_node(parent, 0, aff_false.clone());
        }
        tree.add_child_node(parent, 1, func_true);

        tree
    }

    /// Creates a new AffTree instance which corresponds to the affine [`AffFunc::slice`] function.
    pub fn from_slice(reference_point: &Array1<f64>) -> AffTree<K> {
        AffTree::from_aff(AffFunc::slice(reference_point))
    }

    /// Creates a new AffTree instance from a raw tree whose decisions are affine predicates and terminals are affine functions.
    #[inline]
    pub fn from_tree(tree: Tree<AffContent, K>, dim: usize) -> AffTree<K> {
        AffTree {
            in_dim: dim,
            tree,
            polytope_cache: RefCell::new(Vec::new()),
        }
    }

    /// Returns the input dimension of this tree.
    #[inline]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    // Returns the number of nodes in this tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.tree.len()
    }

    /// Returns true if there are no values in this tree.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// Returns the number of terminals in this tree.
    #[inline]
    pub fn num_terminals(&self) -> usize {
        self.tree.num_terminals()
    }

    /// Returns the depth of this tree, that is, the length of its longest path.
    ///
    /// Correspondingly, an empty tree has depth 0, and a tree with only a root node has depth 1.
    #[inline(always)]
    pub fn depth(&self) -> usize {
        self.tree.depth()
    }

    /// Reserve capacity for at least ``additional`` more nodes to be stored.
    ///
    /// See also [`Tree::reserve`].
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.tree.reserve(additional);
    }

    /// Returns an iterator over the nodes of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    #[rustfmt::skip]
    pub fn nodes(&self) -> impl Iterator<Item = &AffContent> {
        self.tree.nodes()
                 .map(|nd| nd.value)
    }

    /// Returns an iterator over the terminal nodes of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    #[rustfmt::skip]
    pub fn terminals(&self) -> impl Iterator<Item = &AffContent> {
        self.tree.terminals()
                 .map(|nd| nd.value)
    }

    /// Returns an iterator over the decision nodes of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    #[rustfmt::skip]
    pub fn decisions(&self) -> impl Iterator<Item = &AffContent> {
        self.tree.decisions()
                 .map(|nd| nd.value)
    }

    /// Returns an iterator like object that performs a depth-first search over this tree.
    /// For each node in the tree four entries are returned:
    /// 1. the current depth,
    /// 2. the index of the current node,
    /// 3. the number of siblings of the current node that have not been visited yet,
    /// 4. the halfspaces of the path leading to the current node
    ///
    /// In contrast to `polyhedra_iter` this function allows mutual access to the tree during iteration.
    ///
    /// # Examples
    ///
    /// ```
    /// use affinitree::pwl::afftree::AffTree;
    /// use affinitree::linalg::affine::{AffFunc, Polytope};
    ///
    /// let aff_tree = AffTree::<2>::from_aff(AffFunc::identity(2));
    /// let mut poly_iter = aff_tree.polyhedra();
    ///
    /// while let Some((dfs_data, poly)) = poly_iter.next(&aff_tree.tree) {
    ///     println!("{}", Polytope::intersection_n(2, poly));
    /// }
    /// ```
    #[inline]
    pub fn polyhedra(&self) -> PolyhedraGen {
        PolyhedraGen::new(&self.tree)
    }

    /// Returns an iterator that performs a depth-first search over this tree.
    /// For each node in the tree four entries are returned:
    /// 1. the current depth,
    /// 2. the index of the current node,
    /// 3. the number of siblings of the current node that have not been visited yet,
    /// 4. the halfspaces of the path leading to the current node
    ///
    /// For mutual access during iteration see `polyhedra_iter`.
    #[inline]
    pub fn polyhedra_iter(&self) -> PolyhedraIter<'_, K> {
        PolyhedraIter::new(&self.tree)
    }

    /* Helper methods */

    /// Adds a single node to the graph directly.
    ///
    /// **Use with caution:** Caller is responsible to uphold invariants.
    #[inline]
    pub fn add_child_node(&mut self, node: TreeIndex, label: Label, aff: AffFunc) -> TreeIndex {
        self.tree.add_child_node(node, label, AffContent::new(aff))
    }

    /// Adds a single terminal to the graph directly.
    ///
    /// **Use with caution:** Caller is responsible to uphold invariants.
    #[inline]
    pub fn add_terminal(&mut self, node: TreeIndex, label: Label, aff: AffFunc) -> TreeIndex {
        self.tree.add_child_node(node, label, AffContent::new(aff))
    }

    /// Adds a single decision to the graph directly.
    ///
    /// **Use with caution:** Caller is responsible to uphold invariants.
    #[inline]
    pub fn add_decision(&mut self, node: TreeIndex, label: Label, aff: AffFunc) -> TreeIndex {
        assert!(
            aff.outdim() <= K,
            "Decision has more branches than allowed in this tree: K={} but num decisions={}",
            K,
            aff.outdim()
        );
        self.tree.add_child_node(node, label, AffContent::new(aff))
    }

    /// Applies given function to a single node of the graph.
    /// Updates the solution cache.
    ///
    /// **Use with caution:** Caller is responsible to uphold invariants.
    pub fn apply_func_at_node(&mut self, node: TreeIndex, aff: &AffFunc) {
        let aff_node = self.tree.tree_node_mut(node).unwrap();

        aff_node.value.aff = aff.compose(&aff_node.value.aff);
    }

    /// Overwrite the currently stored function at ``node`` to ``aff`` and return the
    /// previously stored function.
    /// Does not reset the solution cache.
    ///
    /// **Use with caution:** Caller is responsible to uphold invariants.
    pub fn update_node(&mut self, node: TreeIndex, aff: AffFunc) -> Option<AffFunc> {
        let val = self.tree.node_value_mut(node)?;
        Some(mem::replace(&mut val.aff, aff))
    }

    /// Sets given function of a single node of the graph and removes all descendants of original node.
    ///
    /// **Use with caution:** Caller is responsible to uphold invariants.
    pub fn replace_node(&mut self, node_idx: TreeIndex, aff: AffFunc) -> TreeIndex {
        // assert!(old_node.parent.is_some(), "Cannot replace default root!");
        if node_idx == self.tree.get_root_idx() {
            self.update_node(0, aff);

            0
        } else {
            let edg = self.tree.parent(node_idx).unwrap();
            let label = edg.label;
            let parent_idx = edg.source_idx;
            self.tree.remove_child(parent_idx, label);

            self.add_child_node(parent_idx, label, aff)
        }
    }

    /* Graph manipulation */

    /// Combines this tree with the specified ``aff_func`` by composing ``aff_func`` to the left of all terminal nodes of this tree.
    /// This is semantically equivalent to first evaluating this tree and then applying ``aff_func`` to the output.
    pub fn apply_func(&mut self, aff_func: &AffFunc) {
        for leaf_idx in self.tree.terminal_indices().collect_vec() {
            self.apply_func_at_node(leaf_idx, aff_func);
        }
    }

    pub fn apply_partial_relu_at_node(&mut self, node: TreeIndex, relu_dim: usize) {
        let aff_node = self.tree.tree_node(node).unwrap();
        let mut aff_func_false = aff_node.value.aff.clone();
        let aff_func_true = aff_node.value.aff.clone();
        let afffunc = aff_node.value.aff.clone();

        aff_func_false.mat.row_mut(relu_dim).fill(0 as f64);
        aff_func_false.bias[relu_dim] = 0 as f64;
        self.add_child_node(node, 1, aff_func_true);
        self.add_child_node(node, 0, aff_func_false);

        // update old node to contain predicate
        let mut mat = Array2::zeros((0, afffunc.indim()));
        mat.push_row(afffunc.mat.row(relu_dim))
            .expect("Could not push row to matrix (critical error)");
        let mut bias = Array1::zeros(1);
        bias[0] = afffunc.bias[relu_dim];
        self.update_node(node, AffFunc::from_mats(mat, bias));
        self.tree.tree_node_mut(node).unwrap().isleaf = false;
    }

    /// Evaluates this AffTree instance as a piece-wise linear function under
    /// the given `input`.
    ///
    /// # Panics
    ///
    /// When a node id is not found in the tree or when usize::MAX edges were followed.
    #[inline]
    pub fn evaluate(&self, input: &Array1<f64>) -> Option<Array1<f64>> {
        self.find_terminal(self.tree.get_root(), input)
            .map(|(func, _)| func.value.aff.apply(input))
    }

    /// Evaluates this AffTree instance under the given `input` starting at `root`.
    /// Each encountered decision predicate is evaluated under the given input and the
    /// corresponding edge is followed until a terminal nodes is reached, which is
    /// then returned together with the label sequence of the path.
    ///
    /// Returns None when some taken edge has no target node.
    ///
    /// # Panics
    ///
    /// When a node id is not found in the tree or when usize::MAX edges were followed.
    pub fn find_terminal<'a>(
        &'a self,
        root: &'a AffNode<K>,
        input: &Array1<f64>,
    ) -> Option<(&TreeNode<AffContent, K>, Vec<Label>)> {
        let mut current_node = root;
        let mut label_seq = Vec::new();
        let mut iter = 0;

        while iter < usize::MAX {
            iter += 1;
            if current_node.isleaf {
                debug_assert!(
                    current_node.children.iter().all(Option::is_none),
                    "nodes that are marked as leafs should not have any children"
                );
                return Some((&current_node, label_seq));
            }

            let label = self.evaluate_decision(current_node, input);
            label_seq.push(label);
            let successor_idx = current_node.children[label]?;
            current_node = self
                .tree
                .tree_node(successor_idx)
                .expect("tree should contain the children of another node");
        }

        panic!("tree should by acyclic and have a maximial depth of usize::MAX-1");
    }

    /// Evaluates the given `node` as a linear decision under the given `input`.
    pub fn evaluate_decision(&self, node: &AffNode<K>, input: &Array1<f64>) -> Label {
        let node_eval = node.value.aff.apply(input).map(|x| *x >= 0.);
        Self::index_from_label(node_eval)
    }

    #[inline]
    fn index_from_label(result: Array1<bool>) -> Label {
        let mut idx = 0;
        for i in 0..result.len_of(Axis(0)) {
            match result[i] {
                false => (),
                true => idx += 1 << i,
            }
        }
        assert!(idx < K);
        idx
    }

    /// Removes the specified axes from this afftree.
    ///
    /// Keeps the input axes where ``mask`` is True and removes all others in the whole tree.
    ///
    /// **Note:** Marked axes are silently dropped, which alters the represented piece-wise linear function. Best used in combination with slicing.
    pub fn remove_axes(&mut self, mask: &Array1<bool>) {
        assert_eq!(self.in_dim(), mask.shape()[0]);

        let keep_idx = mask
            .iter()
            .enumerate()
            .filter(|(_, &x)| x)
            .map(|(i, _)| i)
            .collect_vec();

        self.in_dim = keep_idx.len();

        let mut iter = self.polyhedra();
        while let Some((data, _)) = iter.next(&self.tree) {
            let node = self.tree.node_value_mut(data.index).unwrap();

            let restricted_mat = keep_idx
                .iter()
                .map(|i| node.aff.mat.index_axis(Axis(1), *i).insert_axis(Axis(1)))
                .collect_vec();

            node.aff.mat = concatenate(Axis(1), restricted_mat.as_slice()).unwrap();
            node.state = NodeState::Indeterminate;
        }
    }
}

impl<const K: usize> Display for AffTree<K> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Decision Tree with {} nodes", &self.tree.len())?;

        for (node_idx, node) in self.tree.node_iter() {
            writeln!(
                f,
                "[{:>3}|{}] {}",
                node_idx,
                if node.isleaf { "T" } else { "D" },
                &node
            )?;

            if self.tree.num_children(node_idx) > 0 {
                write!(f, "children: ")?;
                write_children(f, node)?;
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

impl<const K: usize> fmt::Debug for AffTree<K> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Decision Tree with {} nodes", &self.tree.len())?;

        for (node_idx, node) in self.tree.node_iter() {
            writeln!(
                f,
                "[{:>3}|{}] {:?}",
                node_idx,
                if node.isleaf { "Ter" } else { "Dec" },
                &node
            )?;

            if self.tree.num_children(node_idx) > 0 {
                if node.isleaf {
                    write!(f, "!!!")?;
                }
                write!(f, "children: ")?;
                write_children(f, node)?;
                writeln!(f)?;
            }

            if node.parent.is_some() {
                writeln!(f, "parent: {:>3}", node.parent.unwrap())?;
            } else {
                writeln!(f, "parent:  - ")?;
            }

            let state_str = match &node.value.state {
                NodeState::Infeasible => "infeasible".to_string(),
                NodeState::Indeterminate => "indeterminate".to_string(),
                NodeState::Feasible => "feasible".to_string(),
                NodeState::FeasibleWitness(witnesses) => {
                    format!("feasible with {} witnesses", witnesses.len())
                }
            };
            writeln!(f, "state: {}", state_str)?;
        }
        Ok(())
    }
}

#[allow(unused_variables)]
#[cfg(test)]
mod tests {

    use ndarray::{arr1, arr2, array};

    use super::*;
    use crate::distill::schema;
    use crate::linalg::affine::AffFunc;
    use crate::pwl::iter::PolyhedraIter;
    use crate::pwl::node::NodeState;
    use crate::{aff, path, poly};

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

    #[test]
    fn test_from_poly() {
        let poly = poly!([[1, 0, -1], [0, 1, 0]] < [-2, 2]);
        let dd = AffTree::<2>::from_poly(poly, aff!([0, 0, 0] + 1), Some(&aff!([0, 0, 0] + 2)));

        assert_eq!(dd.evaluate(&arr1(&[1., 1., 4.])).unwrap(), arr1(&[1.]));
        assert_eq!(dd.evaluate(&arr1(&[2., 1., 1.])).unwrap(), arr1(&[2.]));
        assert_eq!(dd.evaluate(&arr1(&[-1., 4., 5.])).unwrap(), arr1(&[2.]));
    }

    #[test]
    fn test_apply_func_at_node_keeps_cache() {
        let mut dd = AffTree::from_aff(aff!(
            [[1., 2., 3., 4.], [4., 2., 0., -2.], [1., 3., -5., -7.]] + [11., 13., 17.]
        ));
        dd.compose::<false>(&schema::ReLU(3));
        dd.infeasible_elimination();

        // check test assumption
        let wit = match &dd.tree.node_value(path!(dd.tree, 0, 1, 1)).unwrap().state {
            NodeState::FeasibleWitness(val) => val.clone(),
            _ => panic!("Invalid state of test case"),
        };

        dd.apply_func_at_node(path!(dd.tree, 0, 1, 1), &aff!([1., -1., 1.] + -2.));

        let wit2 = match &dd.tree.node_value(path!(dd.tree, 0, 1, 1)).unwrap().state {
            NodeState::FeasibleWitness(val) => val,
            _ => panic!("Apply_function_at modified the cache"),
        };

        assert_eq!(&wit, wit2);
    }

    #[test]
    fn test_evaluate_0() {
        init_logger();

        let aff = AffFunc::from_mats(arr2(&[[-1., 0.], [0., 1.]]), arr1(&[0., 0.]));

        let mut dd = AffTree::<2>::from_aff(aff);
        let relu = schema::ReLU(2);
        dd.compose::<true>(&relu);

        assert_eq!(dd.evaluate(&arr1(&[1., 1.])).unwrap(), arr1(&[0., 1.]));
        assert_eq!(dd.evaluate(&arr1(&[-1., 1.])).unwrap(), arr1(&[1., 1.]));
        assert_eq!(dd.evaluate(&arr1(&[-1., -1.])).unwrap(), arr1(&[1., 0.]));
        assert_eq!(dd.evaluate(&arr1(&[1., -1.])).unwrap(), arr1(&[0., 0.]));
    }

    #[test]
    fn test_evaluate_1() {
        init_logger();

        let aff = AffFunc::from_mats(arr2(&[[-2., 2.], [0., 1.]]), arr1(&[1., -1.]));

        let mut dd = AffTree::<2>::from_aff(aff);
        let relu = schema::ReLU(2);
        dd.compose::<true>(&relu);

        assert_eq!(dd.evaluate(&arr1(&[2., 1.])).unwrap(), arr1(&[0., 0.]));
        assert_eq!(dd.evaluate(&arr1(&[-1., 2.])).unwrap(), arr1(&[7., 1.]));
        assert_eq!(dd.evaluate(&arr1(&[-2., -1.])).unwrap(), arr1(&[3., 0.]));
        assert_eq!(dd.evaluate(&arr1(&[1., -2.])).unwrap(), arr1(&[0., 0.]));
    }

    #[test]
    fn test_remove_axes() {
        init_logger();

        let mut dd = AffTree::<2>::from_aff(AffFunc::from_mats(arr2(&[[2., 1.]]), arr1(&[-1.])));

        dd.add_child_node(0, 1, aff!([[1., 2.]] + [-0.5]));
        dd.add_child_node(1, 1, aff!([[0.5, 5.]] + [0.]));
        dd.add_child_node(2, 1, aff!([[3., -1.]] + [0.]));
        dd.add_child_node(3, 1, aff!([[1., 1.]] + [-6.]));
        dd.add_child_node(4, 0, aff!([[-1., 7.]] + [4.]));
        dd.add_child_node(5, 1, aff!([[2., 0.2]] + [-3.]));
        dd.add_child_node(6, 0, aff!([[0., 0.]] + [1.]));
        dd.add_child_node(6, 1, aff!([[0., 0.]] + [0.]));

        assert_eq!(dd.evaluate(&arr1(&[1., 0.])).unwrap(), arr1(&[1.]));
        assert_eq!(dd.evaluate(&arr1(&[1.5, 0.])).unwrap(), arr1(&[0.]));
        assert_eq!(dd.evaluate(&arr1(&[4., 0.])).unwrap(), arr1(&[0.]));

        let mut dd1 = dd.clone();
        dd1.remove_axes(&arr1(&[true, false]));

        assert_eq!(dd1.in_dim(), 1);
        assert_eq!(dd1.evaluate(&arr1(&[1.])).unwrap(), arr1(&[1.]));
        assert_eq!(dd1.evaluate(&arr1(&[1.5])).unwrap(), arr1(&[0.]));
        assert_eq!(dd1.evaluate(&arr1(&[4.])).unwrap(), arr1(&[0.]));
    }

    #[test]
    fn bool_to_vec() {
        init_logger();

        assert_eq!(
            AffTree::<8>::index_from_label(arr1(&[true, false, true])),
            5
        );
        assert_eq!(
            AffTree::<8>::index_from_label(arr1(&[false, false, true])),
            4
        );
    }

    #[test]
    fn apply_function() {
        let dd = AffTree::<2>::from_aff(aff!([[2., -3.], [-7.5, 9.3]] + [-2., 4.]));

        assert_eq!(dd.tree.len(), 1);
        assert_eq!(dd.tree.num_children(0), 0);
        assert_eq!(
            dd.tree.node_value(0).unwrap().aff.mat,
            arr2(&[[2., -3.], [-7.5, 9.3]])
        );
        assert_eq!(dd.tree.node_value(0).unwrap().aff.bias, arr1(&[-2., 4.]));
    }

    fn add_leaf(tree: &mut AffTree<2>, node_idx: usize, label: usize, bias_val: f64) {
        let weights = arr2(&[[1., 1., 1.]]);
        let bias = arr1(&[bias_val]);
        let aff_func = AffFunc::from_mats(weights, bias);
        tree.add_child_node(node_idx, label, aff_func);
    }

    fn generic_tree(infeasible: bool) -> AffTree<2> {
        let mut tree = AffTree::<2>::from_aff(aff!([[0., 1., 0.]] + [2.]));
        let weights = match infeasible {
            true => arr2(&[[0., 1., 0.]]),
            false => arr2(&[[5., 2., 8.]]),
        };
        let bias = match infeasible {
            true => arr1(&[3.]),
            false => arr1(&[20.]),
        };
        let aff = AffFunc::from_mats(weights, bias);

        add_leaf(&mut tree, 0, 0, 1.);
        tree.add_child_node(0, 1, aff);
        add_leaf(&mut tree, 2, 0, 1.);
        add_leaf(&mut tree, 2, 1, 1.);
        tree
    }

    #[test]
    pub fn test_get_label() {
        init_logger();

        let tree = generic_tree(true);

        assert_eq!(tree.tree.get_label(0, 2).unwrap(), 1);
        assert_eq!(tree.tree.get_label(2, 3).unwrap(), 0);
        assert_eq!(tree.tree.get_label(2, 4).unwrap(), 1);
    }

    #[test]
    pub fn test_path_to_node() {
        init_logger();

        let tree = generic_tree(true);
        let path = tree.tree.path_to_node(3);
        assert_eq!(path, vec![(0, 1), (2, 0)]);
    }

    #[test]
    pub fn test_polyhedra_iter_root() {
        init_logger();

        let aff_tree = AffTree::<2>::from_aff(AffFunc::identity(2));
        let mut poly_iter = aff_tree.polyhedra();

        let (dfs_data, poly) = poly_iter.next(&aff_tree.tree).unwrap();
        assert_eq!(dfs_data.index, 0);
        assert_eq!(dfs_data.depth, 0);
        assert_eq!(dfs_data.n_remaining, 0);
        assert_eq!(poly.len(), 0);
    }

    #[test]
    pub fn test_polyhedra_iter() {
        init_logger();

        let val0 = AffFunc::from_mats(array![[3., 3.]], array![1.]);
        let val1 = AffFunc::from_mats(array![[1., 0.]], array![2.]);
        let val2 = AffFunc::from_mats(array![[0., 1.]], array![3.]);

        let mut tree = AffTree::<2>::from_aff(val0.clone());

        let c0 = tree.add_child_node(0, 0, val1.clone()); // 1
        let c1 = tree.add_child_node(0, 1, val1.clone()); // 2
        let l0 = tree.add_child_node(c0, 0, val2.clone()); // 3
        let l1 = tree.add_child_node(c0, 1, val2.clone()); // 4
        let r0 = tree.add_child_node(c1, 0, val2.clone()); // 5
        let r1 = tree.add_child_node(c1, 1, val2.clone()); // 6
        let rr1 = tree.add_child_node(r1, 1, val2.clone()); // 7

        let iter = PolyhedraIter::new(&tree.tree);
        let nodes = Vec::from_iter(iter.map(|(_, idx, _, _)| idx));

        assert_eq!(nodes, vec![0, c0, l0, l1, c1, r0, r1, rr1]);

        let iter = PolyhedraIter::new(&tree.tree);
        let remaining = Vec::from_iter(iter.map(|(_, _, remaining, _)| remaining));

        assert_eq!(remaining, vec![0, 1, 1, 0, 0, 1, 0, 0]);
    }

    #[test]
    pub fn test_relu() {
        init_logger();

        let mut a = AffTree::<2>::new(4);
        a.apply_partial_relu_at_node(0, 2);
        a.apply_partial_relu_at_node(a.tree.get_root().children[0].unwrap(), 0);

        assert_eq!(
            a.evaluate(&arr1(&[1., 1., 1., 1.])).unwrap(),
            arr1(&[1., 1., 1., 1.])
        );
        assert_eq!(
            a.evaluate(&arr1(&[-1., -1., -1., -1.])).unwrap(),
            arr1(&[0., -1., 0., -1.])
        );
        assert_eq!(
            a.evaluate(&arr1(&[-1., -1., 1., -1.])).unwrap(),
            arr1(&[-1., -1., 1., -1.])
        );
    }
}
