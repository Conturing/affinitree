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

//! Central data structure to store piece-wise linear functions

use core::fmt;
use std::cell::RefCell;
use std::fmt::Display;
use std::iter::IntoIterator;

use itertools::{enumerate, Itertools};
use log::{debug, warn};
use ndarray::{Array1, Array2, Axis};

use super::{
    iter::PolyhedraIter,
    node::{write_children, AffContent},
};
use crate::distill::iter::PolyhedraGen;
use crate::distill::node::AffNode;
use crate::linalg::affine::{AffFunc, Polytope};
use crate::linalg::polyhedron::PolytopeStatus;
use crate::tree::graph::{Label, Tree, TreeIndex};

/// A corner stone of `affinitree` that can represent any (continuous or non-continuous) [piece-wise linear function](https://en.wikipedia.org/wiki/Piecewise_linear_function).
///
/// This structure is based on an oblique decision tree:
/// * Its inner nodes form a decision structure that partitions the input space similar to a [BSP tree](https://en.wikipedia.org/wiki/Binary_space_partitioning) into convex regions.
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
/// use affinitree::{aff, poly, distill::afftree::AffTree, distill::dot::dot_str, linalg::affine::PolyRepr};
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
/// let mut str = String::new();
/// dot_str(&mut str, &dd).unwrap();
/// println!("{}", str);
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
//// / n2 -> n6 [label=1, style=solid];
/// // }
/// ```
///
/// # Technical Overview
/// `AffTree`s are implemented over an arena provided by the `slab` crate.
/// They have a compile time branching factor `K` (in most cases a binary tree is sufficient, i.e., K=2).
/// Elements of the tree have a unique index during their lifetime.
///
/// # Composition
/// `AffTree`s allow a modular construction by composition. The semantics of composition
/// follow directly from the mathematical definition of [function composition](https://en.wikipedia.org/wiki/Function_composition).
/// Roughly speaking, the composition of two AffTree instances corresponds to an sequential evaluation of the two trees, as demonstrated by the following example.
///
/// ```rust
/// use affinitree::{aff, distill::afftree::AffTree};
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
#[derive(Clone)]
pub struct AffTree<const K: usize> {
    pub tree: Tree<AffContent, K>,
    pub in_dim: usize,
    pub(super) polytope_cache: RefCell<Vec<Polytope>>,
}

impl<const K: usize> AffTree<K> {
    #[inline]
    pub fn new(dim: usize) -> AffTree<K> {
        Self::with_capacity(dim, 0)
    }

    #[inline]
    pub fn with_capacity(dim: usize, capacity: usize) -> AffTree<K> {
        AffTree {
            tree: Tree::with_root(AffContent::new(AffFunc::identity(dim)), capacity),
            in_dim: dim,
            polytope_cache: RefCell::new(Vec::new()),
        }
    }

    #[inline]
    pub fn from_aff(func: AffFunc) -> AffTree<K> {
        AffTree {
            in_dim: func.indim(),
            tree: Tree::with_root(AffContent::new(func), 1),
            polytope_cache: RefCell::new(Vec::new()),
        }
    }

    /// Crates an AffTree instance from the given ``precondition``.
    /// The resulting AffTree is a partial function that only accepts inputs inside
    /// the given ``precondition``, for which it returns the input as is.
    pub fn from_precondition(precondition: Polytope, func: AffFunc) -> AffTree<K> {
        assert!(precondition.indim() == func.indim());
        assert!(precondition.n_constraints() > 0);

        let mut tree = Self::with_capacity(func.indim(), precondition.n_constraints() + 1);
        let mut iter = precondition.row_iter();
        let mut parent = tree.tree.get_root_idx();
        let mut aff = iter.next().unwrap().as_function();
        aff.mat = -aff.mat;
        tree.tree.node_value_mut(parent).unwrap().aff = aff;

        for decision in iter {
            let mut aff = decision.as_function();
            aff.mat = -aff.mat;
            parent = tree.add_child_node(parent, 1, aff);
        }
        tree.add_child_node(parent, 1, func);

        tree
    }

    pub fn from_slice(reference_point: &Array1<f64>) -> AffTree<K> {
        AffTree::from_aff(AffFunc::slice(reference_point))
    }

    #[inline]
    pub fn from_tree(tree: Tree<AffContent, K>, dim: usize) -> AffTree<K> {
        AffTree {
            in_dim: dim,
            tree: tree,
            polytope_cache: RefCell::new(Vec::new()),
        }
    }

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

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.tree.reserve(additional);
    }

    #[inline]
    pub fn nodes(&self) -> impl Iterator<Item = &AffContent> {
        self.tree.nodes().map(|nd| nd.value)
    }

    #[inline]
    pub fn terminals(&self) -> impl Iterator<Item = &AffContent> {
        self.tree.terminals().map(|nd| nd.value)
    }

    #[inline]
    pub fn decisions(&self) -> impl Iterator<Item = &AffContent> {
        self.tree.decisions().map(|nd| nd.value)
    }

    /// Returns an iterator like object that performs a depth-first search over this tree.
    /// For each node in the tree four entries are returned:
    /// 1. the current depth,
    /// 2. the index of the current node,
    /// 3. the number of siblings of the current node that have not been vistied yet,
    /// 4. the halfspaces of the path leading to the current node
    ///
    /// In contrast to `polyhedra_iter` this function allows mutual access to the tree during iteration.
    ///
    /// # Examples
    ///
    /// ```
    /// use affinitree::distill::afftree::AffTree;
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
    /// 3. the number of siblings of the current node that have not been vistied yet,
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
        aff_node.value.solutions = RefCell::new(Vec::new());
    }

    /// Sets given function of a single node of the graph.
    /// Resets the solution cache.
    ///
    /// **Use with caution:** Caller is responsible to uphold invariants.
    pub fn update_node(&mut self, node: TreeIndex, aff: AffFunc) -> Option<AffFunc> {
        self.tree
            .update_node(node, AffContent::new(aff))
            .map(|nd| nd.aff)
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

    #[inline]
    pub fn evaluate(&self, input: &Array1<f64>) -> Option<Array1<f64>> {
        self.evaluate_to_terminal(self.tree.get_root(), input, 512)
            .map(|(func, _)| func.apply(input))
    }

    pub fn evaluate_to_terminal<'a>(
        &'a self,
        node: &'a AffNode<K>,
        input: &Array1<f64>,
        max_iter: i32,
    ) -> Option<(&AffFunc, Vec<Label>)> {
        let mut current_node = node;
        let mut label_seq = Vec::new();

        for _ in 0..max_iter {
            if current_node.isleaf {
                debug_assert!(
                    current_node.children.iter().all(Option::is_none),
                    "Tree invariant violated: leaf node has children."
                );
                return Some((&current_node.value.aff, label_seq));
            }

            let label = self.evaluate_node(current_node, input);
            label_seq.push(label);
            let successor_idx = current_node.children[label]
                .expect(&format!("Node has no successor for label {}", &label));
            current_node = self.tree.tree_node(successor_idx)?;
        }

        None
    }

    pub fn evaluate_node(&self, node: &AffNode<K>, input: &Array1<f64>) -> Label {
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

    /* Composition */

    /// Extends self such that it represents the result of mathematical function composition
    /// of this tree and other.
    ///
    /// # Example
    ///
    /// ```rust
    /// use affinitree::{aff, distill::afftree::AffTree};
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
    /// assert_eq!(
    ///     tree1.evaluate(&tree0.evaluate(&arr1(&[2., -7.])).unwrap()).unwrap(),
    ///     comp.evaluate(&arr1(&[2., -7.])).unwrap()
    /// );
    /// ```
    #[inline]
    pub fn compose<const prune: bool>(&mut self, other: &AffTree<K>) {
        if prune {
            AffTree::lift_func(
                other,
                self,
                self.tree.terminal_indices().collect_vec(),
                |a, b| a.compose(b),
                |a, b| a.compose(b),
                |tree, src_idx, dest_idx| tree.is_edge_feasible(src_idx, dest_idx, Option::None),
            )
        } else {
            AffTree::lift_func(
                other,
                self,
                self.tree.terminal_indices().collect_vec(),
                |a, b| a.compose(b),
                |a, b| a.compose(b),
                |_, _, _| true,
            )
        }
    }

    #[inline]
    pub fn compose_trees(lhs: &AffTree<K>, rhs: &mut AffTree<K>) {
        AffTree::lift_func(
            lhs,
            rhs,
            rhs.tree.terminal_indices().collect_vec(),
            |a, b| a.compose(b),
            |a, b| a.compose(b),
            |tree, src_idx, dest_idx| !tree.is_edge_feasible(src_idx, dest_idx, Option::None),
        )
    }

    /// Apply the function f encoded by lhs to rhs.
    /// Technically, this is implemented by appending a copy of lhs to every terminal in rhs.
    /// Two callbacks are provided that can update nodes depending on the operation performed, thus
    /// allowing to encode a variety of functions.
    ///
    /// Each inner node in lhs is updated with the callback nonterminal_func which receives the current inner node of lhs and the terminal node of rhs where this copy is appended.
    /// Each terminal node in lhs is updated with the callback terminal_func which receives the current terminal node of lhs and the terminal node of rhs where this copy is appended.
    pub fn lift_func<I, NFn, TFn, IFn>(
        lhs: &AffTree<K>,
        rhs: &mut AffTree<K>,
        terminals: I,
        nonterminal_func: NFn,
        terminal_func: TFn,
        is_feasible: IFn,
    ) where
        I: IntoIterator<Item = TreeIndex>,
        NFn: Fn(&AffFunc, &AffFunc) -> AffFunc,
        TFn: Fn(&AffFunc, &AffFunc) -> AffFunc,
        IFn: Fn(&AffTree<K>, TreeIndex, TreeIndex) -> bool,
    {
        for terminal_idx in terminals.into_iter() {
            let terminal = rhs.tree.tree_node(terminal_idx).unwrap();

            let terminal_val = terminal.value.aff.clone();
            let new_root_aff = match lhs.tree.get_root().isleaf {
                true => terminal_func(&lhs.tree.get_root().value.aff, &terminal_val),
                false => nonterminal_func(&lhs.tree.get_root().value.aff, &terminal_val),
            };

            let new_root = rhs.replace_node(terminal_idx, new_root_aff);

            let mut stack: Vec<(TreeIndex, TreeIndex)> =
                Vec::with_capacity(lhs.tree.num_terminals());
            stack.push((lhs.tree.get_root_idx(), new_root));

            while let Some((parent0_idx, parent1_idx)) = stack.pop() {
                for edg in lhs.tree.children(parent0_idx) {
                    let child0_idx = edg.target_idx;
                    let child0 = edg.target_value;
                    let label = edg.label;

                    debug!("Processing left tree child with id: {:?}", child0_idx);
                    let new_value = match lhs.tree.is_leaf(child0_idx).unwrap() {
                        true => terminal_func(&child0.aff, &terminal_val),
                        false => nonterminal_func(&child0.aff, &terminal_val),
                    };

                    let child1_idx =
                        rhs.tree
                            .add_child_node(parent1_idx, label, AffContent::new(new_value));

                    // Test feasibility of newly created edge, remove if infeasible
                    if is_feasible(rhs, parent1_idx, child1_idx) {
                        stack.push((child0_idx, child1_idx));
                    } else {
                        rhs.tree.remove_child(parent1_idx, label);
                    }
                }
            }
        }
        // return (rhs, new_nodes);
    }

    /* Feasibility functions */

    /// Iterate the tree using a breath-first search to find and remove infeasible edges.
    pub fn infeasible_elimination(&mut self, precondition: Option<Polytope>) {
        let mut stack: Vec<usize> = Vec::with_capacity(self.tree.num_terminals());
        stack.push(0);

        while let Some(node_idx) = stack.pop() {
            let node = self.tree.tree_node(node_idx).unwrap();
            if node.isleaf {
                continue;
            }

            for (label, child_idx) in enumerate(node.children) {
                if let Some(child_idx) = child_idx {
                    let feasible = self.is_edge_feasible(node_idx, child_idx, precondition.clone());

                    // If edge is not feasible, remove the child, automatically frees the entire subtree
                    if !feasible {
                        self.tree.remove_child(node_idx, label);
                    } else {
                        stack.push(child_idx);
                    }
                }
            }
        }
    }

    pub fn is_edge_feasible(
        &self,
        parent_idx: usize,
        node_idx: usize,
        precondition: Option<Polytope>,
    ) -> bool {
        // Edges from the root node are always feasible
        if parent_idx == 0 {
            return true;
        }

        let mut path = self.tree.path_to_node(parent_idx);

        let label = self
            .tree
            .get_label(parent_idx, node_idx)
            .expect("edge not contained in graph");
        path.push((parent_idx, label));

        // Calculate polytope representing the path
        let mut poly = self.polyhedral_path_characterization(&path);

        if let Some(ref precondition_poly) = precondition {
            poly = poly.intersection(precondition_poly);
        }

        // Check source cache for solution, delete all invalid ones
        let mut node_solutions = self
            .tree
            .node_value(node_idx)
            .unwrap()
            .solutions
            .borrow_mut();
        if node_solutions.iter().any(|x| poly.contains(x)) {
            return true;
        }

        // No cached solution found
        // Check if solution from parent node solves linear program -> superset
        let parent_node = self.tree.tree_node(parent_idx).unwrap();
        let inherited_solutions = Self::possible_solutions(parent_node, &poly);

        // Solution is valid, copy to child
        if !inherited_solutions.is_empty() {
            for sol in inherited_solutions {
                node_solutions.push(sol.to_owned());
            }
            return true;
        }

        // No existing solutions found, calculate a new solution based on polytope
        debug!("Passing LP to solver: {poly:?}");
        let status = poly.status();
        debug!("Solver status {status:?}");

        match status {
            PolytopeStatus::Infeasible => false,
            PolytopeStatus::Unbounded => {
                warn!("Target function of LP is reported as unbounded, but it is constant.\nPolytope: {:?}", &poly);
                true
            }
            PolytopeStatus::Optimal(solution) => {
                node_solutions.push(solution);
                true
            }
            PolytopeStatus::Error(err) => {
                // Definitely not good when the LP solver fails, but it is recoverable.
                // Therefore it would be wrong to panic.
                warn!("Error occurred while solving the linear program for an infeasible path!\nSolver status: {:?}\nPolytope: {:?}", err, &poly);
                true
            }
        }
    }

    #[inline]
    fn possible_solutions(node: &AffNode<K>, poly: &Polytope) -> Vec<Array1<f64>> {
        node.value
            .solutions
            .borrow()
            .iter()
            .filter(|&solution| poly.contains(solution))
            .map(|x| x.to_owned())
            .collect()
    }

    fn polyhedral_path_characterization(&self, path: &Vec<(TreeIndex, Label)>) -> Polytope {
        // Most expensive function outside of LP solver
        let mut cache = self.polytope_cache.borrow_mut();
        cache.clear();
        cache.reserve(path.len());

        for (idx, label) in path {
            let aff_func_node = &self.tree.node_value(*idx).unwrap().aff;
            // Flip for label 1 because >= becomes <=
            let factor = match label {
                1 => -1.0,
                0 => 1.0,
                _ => 0.0,
            };

            let poly_node =
                Polytope::from_mats(&aff_func_node.mat * factor, &aff_func_node.bias * -factor);
            cache.push(poly_node);
        }
        let in_dim = self.in_dim();

        let poly = Polytope::intersection_n(in_dim, cache.as_slice());
        cache.clear();
        poly
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

            writeln!(f, "cache: {:>3} items", node.value.solutions.borrow().len())?;
        }
        Ok(())
    }
}

#[allow(unused_variables)]
#[cfg(test)]
mod tests {

    use super::AffTree;
    use crate::{
        aff,
        distill::{iter::PolyhedraIter, schema},
        linalg::affine::{AffFunc, PolyRepr, Polytope},
        path, poly,
    };

    use assertables::*;
    use itertools::Itertools;
    use ndarray::{arr1, arr2, array};

    fn init_logger() {
        // minilp has a bug if logging is enabled
        // match fast_log::init(Config::new().console().chan_len(Some(100000))) {
        //     Ok(_) => (),
        //     Err(err) => println!("Error occurred while configuring logger: {:?}", err),
        // }
    }

    #[test]
    fn evaluate_0() {
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
    fn evaluate_1() {
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
    fn test_compose() {
        init_logger();

        let mut tree0 = AffTree::<2>::from_aff(aff!([[1., 0.]] + [2.]));
        tree0.add_child_node(0, 0, aff!([[2, 0], [0, 2]] + [1, 0]));
        tree0.add_child_node(0, 1, aff!([[2, 0], [0, 2]] + [0, 1]));

        let mut tree1 = AffTree::<2>::from_aff(aff!([[-0.5, 0.]] + [-1.]));
        tree1.add_child_node(0, 0, aff!([[3, 0], [0, 3]] + [5, 0]));
        tree1.add_child_node(0, 1, aff!([[-3, 0], [0, -3]] + [0, 5]));

        let mut comp = tree0.clone();
        comp.compose::<false>(&tree1);

        assert_eq!(
            comp.tree.node_value(path!(comp.tree, 0)).unwrap().aff,
            aff!([-1, 0] + -1.5)
        );

        assert_eq!(
            comp.tree.node_value(path!(comp.tree, 1, 1)).unwrap().aff,
            aff!([[-6, 0], [0, -6]] + [0, 2])
        );

        assert_eq!(
            tree1
                .evaluate(&tree0.evaluate(&arr1(&[2., -7.])).unwrap())
                .unwrap(),
            comp.evaluate(&arr1(&[2., -7.])).unwrap()
        );

        assert_eq!(
            tree1
                .evaluate(&tree0.evaluate(&arr1(&[-1., -0.3])).unwrap())
                .unwrap(),
            comp.evaluate(&arr1(&[-1., -0.3])).unwrap()
        );

        assert_eq!(
            tree1
                .evaluate(&tree0.evaluate(&arr1(&[12., 3.])).unwrap())
                .unwrap(),
            comp.evaluate(&arr1(&[12., 3.])).unwrap()
        );

        // full binary tree 1 + 2 + 4
        assert_eq!(comp.tree.len(), 7);
    }

    #[test]
    fn test_compose_infeasible() {
        init_logger();

        let mut dd0 = AffTree::<2>::from_aff(aff!([[1., 0.]] + [2.]));
        dd0.add_child_node(0, 0, aff!([[2, 0], [0, 2]] + [1, 0]));
        dd0.add_child_node(0, 1, aff!([[2, 0], [0, 2]] + [0, 1]));

        let mut dd1 = AffTree::<2>::from_aff(aff!([[-0.5, 0.]] + [-1.]));
        dd1.add_child_node(0, 0, aff!([[3, 0], [0, 3]] + [5, 0]));
        dd1.add_child_node(0, 1, aff!([[-3, 0], [0, -3]] + [0, 5]));

        dd0.compose::<true>(&dd1);

        // one node is infeasible and should be eliminated
        assert_eq!(dd0.tree.len(), 6);
        // dd1 should remain unchanged
        assert_eq!(dd1.tree.len(), 3);
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
    pub fn test_polyhedral_path_characterization() {
        init_logger();

        let tree = generic_tree(true);
        let path = tree.tree.path_to_node(3);
        let poly = tree.polyhedral_path_characterization(&path);

        assert_eq!(
            poly,
            Polytope::new(AffFunc::from_mats(
                arr2(&[[-0., -1., -0.], [0., 1., 0.]]),
                arr1(&[2., -3.])
            ))
        );
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
    pub fn test_is_edge_feasible() {
        init_logger();

        let tree = generic_tree(true);

        assert!(tree.is_edge_feasible(0, 1, Option::None));
        assert!(tree.is_edge_feasible(0, 2, Option::None));
        assert!(!tree.is_edge_feasible(2, 3, Option::None));
        assert!(tree.is_edge_feasible(2, 4, Option::None));
    }

    #[test]
    pub fn test_infeasible_elimination() {
        init_logger();

        let mut tree = generic_tree(true);

        tree.infeasible_elimination(None);

        let nodes: Vec<usize> = tree.tree.node_iter().map(|(idx, _)| idx).collect();

        assert_not_contains!(nodes, &3);
        assert!(tree.tree.tree_node(4).unwrap().isleaf);
    }

    #[test]
    pub fn test_lift_affine_func() {
        init_logger();

        let tree1 = generic_tree(false);
        let mut tree2 = AffTree::<2>::from_aff(aff!([[5., 1., 6.]] + [6.]));
        add_leaf(&mut tree2, 0, 0, 7.);
        add_leaf(&mut tree2, 0, 1, 2.);

        let terminals = tree2.tree.terminal_indices().collect_vec();
        AffTree::lift_func(
            &tree1,
            &mut tree2,
            terminals,
            |x, _| x.clone(),
            |x, y| x.clone().add(y),
            |tree, src, dest| tree.is_edge_feasible(src, dest, Option::None),
        );

        assert_eq!(tree2.tree.len(), 11);
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

    #[test]
    pub fn dd_is_feasible() {
        // Construct one path / polytope which has solutions in the upper quadrant

        let mut dd = AffTree::<2>::from_aff(AffFunc::from_mats(arr2(&[[2., 1.]]), arr1(&[-1.])));

        dd.add_child_node(0, 1, aff!([[1., 2.]] + [-1.5]));
        dd.add_child_node(1, 1, aff!([[0.5, 5.]] + [1.0]));
        dd.add_child_node(2, 1, aff!([[3., -1.]] + [0.]));
        dd.add_child_node(3, 1, aff!([[-1., -1.]] + [6.]));
        dd.add_child_node(4, 0, aff!([[-1., 7.]] + [4.]));
        dd.add_child_node(5, 1, aff!([[-2., -0.2]] + [3.]));
        // feasible
        dd.add_child_node(6, 0, aff!([[0., 0.]] + [1.]));
        // infeasible
        dd.add_child_node(6, 1, aff!([[0., 0.]] + [0.]));

        assert!(dd.is_edge_feasible(6, 7, None));
        assert!(!dd.is_edge_feasible(6, 8, None));

        assert_eq!(dd.evaluate(&arr1(&[6., 6.])).unwrap(), arr1(&[1.]));
        assert_eq!(dd.evaluate(&arr1(&[4.5, 2.5])).unwrap(), arr1(&[1.]));

        assert_eq!(
            dd.tree.path_to_node(7),
            &[(0, 1), (1, 1), (2, 1), (3, 1), (4, 0), (5, 1), (6, 0)]
        );
        assert_eq!(
            dd.tree.path_to_node(8),
            &[(0, 1), (1, 1), (2, 1), (3, 1), (4, 0), (5, 1), (6, 1)]
        );

        // Polytope uses Ax <= b while afftree uses Ax + b >= 0
        let poly = Polytope::from_mats(
            arr2(&[
                [-2., -1.],
                [-1., -2.],
                [-0.5, -5.],
                [-3., 1.],
                [-1., -1.],
                [1., -7.],
                [-2., -0.2],
            ]),
            arr1(&[-1., -1.5, 1.0, 0., -6., 4., -3.]),
        );

        assert_eq!(
            dd.polyhedral_path_characterization(&dd.tree.path_to_node(7)),
            poly
        );
    }

    #[test]
    pub fn dd_is_feasible2() {
        init_logger();
        // Construct one path / polytope which has solutions in the upper quadrant

        let mut dd = AffTree::<2>::from_aff(AffFunc::from_mats(arr2(&[[2., 1.]]), arr1(&[-1.])));

        dd.add_child_node(0, 1, aff!([[1., 2.]] + [-1.5]));
        dd.add_child_node(1, 1, aff!([[0.5, 5.]] + [1.0]));
        dd.add_child_node(2, 1, aff!([[3., -1.]] + [0.]));
        dd.add_child_node(3, 1, aff!([[-1., -1.]] + [6.]));
        dd.add_child_node(4, 0, aff!([[-1., 7.]] + [4.]));
        dd.add_child_node(5, 1, aff!([[-2., -0.2]] + [3.]));
        // feasible
        dd.add_child_node(6, 0, aff!([[0., 0.]] + [1.]));
        // infeasible
        dd.add_child_node(6, 1, aff!([[0., 0.]] + [0.]));

        dd.infeasible_elimination(None);

        // infeasible node removed
        let indices = dd.tree.node_indices().collect_vec();
        assert_not_contains!(indices, &9);
        assert_eq!(dd.len(), 8);
    }

    #[test]
    pub fn dd_is_feasible2_2() {
        init_logger();
        // Construct a polytope where every hyperplane is supportive (i.e., no hyperplane is redundant)
        // Solutions are roughly contained inside [-4, 4] x [-1, 7]

        let poly = poly!(
            [
                [1, 1],
                [-1, -2],
                [-2, -1],
                [1, 5],
                [7, 1],
                [-2, -3],
                [1, -3]
            ] < [5, -2, 0.5, 30, 28, -4, 5]
        );

        let mut dd = AffTree::<2>::from_precondition(poly, AffFunc::identity(2));

        dd.infeasible_elimination(None);

        // Each of the seven hyperplanes is required + one terminal
        assert_eq!(dd.len(), 8);
    }

    #[test]
    pub fn dd_is_feasible2_3() {
        init_logger();
        // Construct one path / polytope which has solutions in the upper quadrant

        let poly = poly!(
            [
                [-2, -1, 1, 1, -1, 0],
                [-1, -2, -1, -2, 0, -1],
                [-0.5, -5, -2, -1, 1, 0],
                [-3, 1, 1, 5, 0, 1],
                [1, 1, 7, 1, -1, -1],
                [1, 7, -2, -3, 1, 1],
                [2, 0.2, 1, -3, -1, 1]
            ] < [3, -4.5, 3.5, 34, 31, 6, 6]
        );

        let mut dd = AffTree::<2>::from_precondition(poly, AffFunc::identity(6));

        let idx = dd.add_child_node(
            path!(dd.tree, 1, 1, 1, 1),
            0,
            poly!([[1, 1, 1, 1, 1, 1]] < [-20]).convert_to(PolyRepr::MatrixBiasGeqZero),
        );
        dd.add_child_node(idx, 0, AffFunc::constant(6, -1.0));
        dd.add_child_node(idx, 1, AffFunc::constant(6, 1.0));

        dd.infeasible_elimination(None);

        assert_eq!(dd.len(), 11);
    }
}
