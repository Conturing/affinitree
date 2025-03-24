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

use std::time::{Duration, Instant};

use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::debug;

use super::afftree::*;
use crate::linalg::affine::*;
use crate::pwl::node::AffContent;
use crate::tree::graph::*;

/// A vistor for the composition of two [``AffTree``]s.
///
/// Use cases include logging or to display progress.
pub trait CompositionVisitor {
    fn start_composition(&mut self, expected_iterations: usize);
    fn start_subtree(&mut self, node: TreeIndex);
    fn finish_subtree(&mut self, n_nodes: usize);
    fn finish_composition(&mut self);
}

/// A set of rules that define an algebraic operation over two [``AffTree``] instances.
pub trait CompositionSchema {
    /// Creates a new decision node based on the two operands ``original`` and ``context``.
    /// Here, ``original`` is the the value of the lhs tree and ``context`` of the rhs tree.
    fn update_decision(original: &AffFunc, context: &AffFunc) -> AffFunc;

    /// Creates a new terminal node based on the two operands ``original`` and ``context``.
    /// Here, ``original`` is the the value of the lhs tree and ``context`` of the rhs tree.
    fn update_terminal(original: &AffFunc, context: &AffFunc) -> AffFunc;

    /// A filter to stop descending an edge in lhs.
    fn explore<const K: usize>(context: &AffTree<K>, parent: TreeIndex, child: TreeIndex) -> bool;
}

/// A [``CompositionVisitor``] with no operations.
#[derive(Clone, Debug)]
pub struct NoOpVis {}

impl CompositionVisitor for NoOpVis {
    fn start_composition(&mut self, _: usize) {}

    fn start_subtree(&mut self, _: TreeIndex) {}

    fn finish_subtree(&mut self, _: usize) {}

    fn finish_composition(&mut self) {}
}

/// A [``CompositionVisitor``] which displays a progress bar of the current state
/// of the composition at the console.
#[derive(Clone, Debug)]
pub struct CompositionConsole {
    pb: ProgressBar,
    timer: Instant,
    len: usize,
}

impl Default for CompositionConsole {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositionConsole {
    pub fn new() -> CompositionConsole {
        CompositionConsole {
            pb: ProgressBar::hidden(),
            timer: Instant::now(),
            len: 0,
        }
    }
}

impl CompositionVisitor for CompositionConsole {
    fn start_composition(&mut self, expected_iterations: usize) {
        self.pb = ProgressBar::new(expected_iterations as u64);
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

        self.timer = Instant::now();
        self.len = expected_iterations;
    }

    fn start_subtree(&mut self, _node: TreeIndex) {}

    fn finish_subtree(&mut self, _n_nodes: usize) {
        self.pb.inc(1);
    }

    fn finish_composition(&mut self) {
        self.pb.finish_and_clear();
    }
}

/// A [``CompositionSchema``] for mathematical function composition.
#[derive(Clone, Debug)]
pub struct FunctionComposition {}

impl CompositionSchema for FunctionComposition {
    fn update_decision(original: &AffFunc, context: &AffFunc) -> AffFunc {
        AffFunc::from_mats(
            original.mat.dot(&context.mat),
            -original.mat.dot(&context.bias) + &original.bias,
        )
    }

    fn update_terminal(original: &AffFunc, context: &AffFunc) -> AffFunc {
        original.compose(context)
    }

    fn explore<const K: usize>(
        _context: &AffTree<K>,
        _parent: TreeIndex,
        _child: TreeIndex,
    ) -> bool {
        true
    }
}

/// A [``CompositionSchema``] for mathematical function composition which performs on-the-fly infeasible elimination.
#[derive(Clone, Debug)]
pub struct FunctionCompositionInfeasible {}

impl CompositionSchema for FunctionCompositionInfeasible {
    fn update_decision(original: &AffFunc, context: &AffFunc) -> AffFunc {
        AffFunc::from_mats(
            original.mat.dot(&context.mat),
            -original.mat.dot(&context.bias) + &original.bias,
        )
    }

    fn update_terminal(original: &AffFunc, context: &AffFunc) -> AffFunc {
        original.compose(context)
    }

    fn explore<const K: usize>(context: &AffTree<K>, parent: TreeIndex, child: TreeIndex) -> bool {
        context.is_edge_feasible(parent, child)
    }
}

/// # Composition
impl<const K: usize> AffTree<K> {
    /// Performs mathematical function composition of ``self`` and ``other``.
    ///
    /// # Example
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
    /// comp.compose::<false, false>(&tree1);
    ///
    /// assert_eq!(
    ///     tree1.evaluate(&tree0.evaluate(&arr1(&[2., -7.])).unwrap()).unwrap(),
    ///     comp.evaluate(&arr1(&[2., -7.])).unwrap()
    /// );
    /// ```
    #[inline]
    pub fn compose<const PRUNE: bool, const VERBOSE: bool>(&mut self, other: &AffTree<K>) {
        if PRUNE && VERBOSE {
            AffTree::<K>::generic_composition_inplace(
                other,
                self,
                self.tree.terminal_indices().collect_vec(),
                FunctionCompositionInfeasible {},
                CompositionConsole::new(),
            );
        } else if PRUNE && !VERBOSE {
            AffTree::<K>::generic_composition_inplace(
                other,
                self,
                self.tree.terminal_indices().collect_vec(),
                FunctionCompositionInfeasible {},
                NoOpVis {},
            );
        } else if !PRUNE && VERBOSE {
            AffTree::<K>::generic_composition_inplace(
                other,
                self,
                self.tree.terminal_indices().collect_vec(),
                FunctionComposition {},
                CompositionConsole::new(),
            );
        } else {
            AffTree::<K>::generic_composition_inplace(
                other,
                self,
                self.tree.terminal_indices().collect_vec(),
                FunctionComposition {},
                NoOpVis {},
            );
        }
    }

    /// Applies the algebraic operation defined by ``schema`` to ``rhs`` and ``lhs``.
    /// This operation is applied inplace modifying ``rhs``.
    /// It is implemented as one tree traversal over ``lhs`` for each terminal in ``rhs``.
    pub fn generic_composition_inplace<I, C, V>(
        lhs: &AffTree<K>,
        rhs: &mut AffTree<K>,
        terminals: I,
        _schema: C,
        mut visitor: V,
    ) where
        I: IntoIterator<Item = TreeIndex>,
        C: CompositionSchema,
        V: CompositionVisitor,
    {
        let iter = terminals.into_iter();

        visitor.start_composition(iter.size_hint().0);

        for terminal_idx in iter {
            debug!("Processing terminal (rhs): id={:?}", terminal_idx);
            let terminal = rhs
                .tree
                .tree_node(terminal_idx)
                .expect("All nodes of the iterator should be terminals in the rhs tree");
            assert!(
                terminal.isleaf,
                "Terminal node of given iterator should be a leaf"
            );

            let terminal_aff: AffFuncBase<FunctionT, ndarray::OwnedRepr<f64>> =
                terminal.value.aff.clone();
            let new_root_aff = match lhs.tree.get_root().isleaf {
                true => C::update_terminal(&lhs.tree.get_root().value.aff, &terminal_aff),
                false => C::update_decision(&lhs.tree.get_root().value.aff, &terminal_aff),
            };
            debug!("New terminal value: {}", new_root_aff);

            visitor.start_subtree(terminal_idx);
            let mut n_nodes = 0;

            // Update the stored function to the new predicate while keeping the cache intact
            rhs.update_node(terminal_idx, new_root_aff).unwrap();

            let mut stack: Vec<(TreeIndex, TreeIndex)> =
                Vec::with_capacity(lhs.tree.num_terminals());
            stack.push((lhs.tree.get_root_idx(), terminal_idx));

            while let Some((parent0_idx, parent1_idx)) = stack.pop() {
                let mut created_children = 0;
                let mut skipped_children = 0;
                let mut label_created = None;

                for edg in lhs.tree.children(parent0_idx) {
                    let child0_idx = edg.target_idx;
                    let child0 = edg.target_value;
                    let label = edg.label;

                    let is_leaf = lhs.tree.is_leaf(child0_idx).unwrap();
                    debug!(
                        "Processing node from left tree with id={:?} ({})",
                        child0_idx,
                        if is_leaf { "T" } else { "D" }
                    );
                    let child1_aff = match is_leaf {
                        true => C::update_terminal(&child0.aff, &terminal_aff),
                        false => C::update_decision(&child0.aff, &terminal_aff),
                    };
                    debug!("New node value: {}", child1_aff);

                    let child1_idx = rhs
                        .tree
                        .add_child_node(parent1_idx, label, AffContent::new(child1_aff))
                        .unwrap();

                    // Test feasibility of newly created edge, remove if infeasible
                    if C::explore(rhs, parent1_idx, child1_idx) {
                        stack.push((child0_idx, child1_idx));
                        created_children += 1;
                        n_nodes += 1;
                        label_created = Some(label);
                    } else {
                        skipped_children += 1;
                        rhs.tree.remove_child(parent1_idx, label);
                    }
                }

                // In the case of no children remove_child already cleans up the tree
                if created_children == 1 && created_children + skipped_children == K {
                    debug!("Forwarding node");
                    // Move affine function to parent node and clean up tree
                    rhs.tree
                        .merge_child_with_parent(parent1_idx, label_created.unwrap())
                        .unwrap();
                }
            }

            visitor.finish_subtree(n_nodes);
        }
        visitor.finish_composition();
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;
    use crate::{aff, path};

    fn init_logger() {
        use env_logger::Target;
        use log::LevelFilter;

        let _ = env_logger::builder()
            .is_test(true)
            .filter_module("minilp", LevelFilter::Error)
            .target(Target::Stdout)
            .filter_level(LevelFilter::Debug)
            .try_init();
    }

    #[test]
    fn test_compose() {
        init_logger();

        let mut tree0 = AffTree::<2>::from_aff(aff!([[1., 0.]] + [2.]));
        tree0
            .add_child_node(0, 0, aff!([[2, 0], [0, 2]] + [1, 0]))
            .unwrap();
        tree0
            .add_child_node(0, 1, aff!([[2, 0], [0, 2]] + [0, 1]))
            .unwrap();

        let mut tree1 = AffTree::<2>::from_aff(aff!([[-0.5, 0.]] + [-1.]));
        tree1
            .add_child_node(0, 0, aff!([[3, 0], [0, 3]] + [5, 0]))
            .unwrap();
        tree1
            .add_child_node(0, 1, aff!([[-3, 0], [0, -3]] + [0, 5]))
            .unwrap();

        let mut comp = tree0.clone();

        let terminals = comp.tree.terminal_indices().collect_vec();
        AffTree::generic_composition_inplace(
            &tree1,
            &mut comp,
            terminals,
            FunctionComposition {},
            CompositionConsole::new(),
        );

        assert_eq!(
            comp.tree.node_value(path!(comp.tree, 0)).unwrap().aff,
            aff!([-1, 0] + -0.5)
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
        dd0.add_child_node(0, 0, aff!([[2, 0], [0, 2]] + [1, 0]))
            .unwrap();
        dd0.add_child_node(0, 1, aff!([[2, 0], [0, 2]] + [0, 1]))
            .unwrap();

        let mut dd1 = AffTree::<2>::from_aff(aff!([[-0.5, 0.]] + [-1.]));
        dd1.add_child_node(0, 0, aff!([[3, 0], [0, 3]] + [5, 0]))
            .unwrap();
        dd1.add_child_node(0, 1, aff!([[-3, 0], [0, -3]] + [0, 5]))
            .unwrap();

        dd0.compose::<true, false>(&dd1);

        // one node is infeasible and should be eliminated
        // and one forwarded
        assert_eq!(dd0.tree.len(), 5);
        // dd1 should remain unchanged
        assert_eq!(dd1.tree.len(), 3);
    }

    macro_rules! value_at {
        ($tree:expr , $( $label:literal ),* ) => {
            $tree.tree.node_value(path!($tree.tree, $( $label ),* )).unwrap().aff
        }
    }

    #[test]
    #[rustfmt::skip]
    fn test_compose_exact() {
        init_logger();

        let mut tree0 = AffTree::<2>::from_aff(aff!([1, 0, 0] + 2));
        tree0.add_child_node(0, 0, aff!([[2, 0, 0], [0, 2, 0]] + [-4, 0])).unwrap();
        tree0.add_child_node(0, 1, aff!([[2, 0, 0], [0, 0, 2]] + [-8, 1])).unwrap();

        let mut tree1 = AffTree::<2>::from_aff(aff!([[0.5, 0.]] + [-1.]));
        tree1.add_child_node(0, 0, aff!([[3, 0], [0, 3]] + [5, 0])).unwrap();
        tree1.add_child_node(0, 1, aff!([[-3, 0], [0, -3]] + [0, 5])).unwrap();

        tree0.compose::<false, false>(&tree1);

        eprintln!("{}", &tree0);

        assert_eq!(
            tree0.tree.node_value(0).unwrap().aff,
            aff!([1, 0, 0] + 2)
        );

        assert_eq!(
            value_at!(tree0, 0),
            aff!([1, 0, 0] + 1)
        );

        assert_eq!(
            value_at!(tree0, 1),
            aff!([1, 0, 0] + 3)
        );

        assert_eq!(
            value_at!(tree0, 0, 0),
            aff!([[6, 0, 0], [0, 6, 0]] + [-12 + 5, 0])
        );

        assert_eq!(
            value_at!(tree0, 0, 1),
            aff!([[-6, 0, 0], [0, -6, 0]] + [12, 5])
        );

        assert_eq!(
            value_at!(tree0, 1, 0),
            aff!([[6, 0, 0], [0, 0, 6]] + [-24 + 5, 3])
        );

        assert_eq!(
            value_at!(tree0, 1, 1),
            aff!([[-6, 0, 0], [0, 0, -6]] + [24, -3 + 5])
        );
    }

    #[test]
    #[rustfmt::skip]
    fn test_compose_exact_infeasible() {
        init_logger();

        let mut tree0 = AffTree::<2>::from_aff(aff!([1, 0, 0] + 2));
        tree0.add_child_node(0, 0, aff!([[2, 0, 0], [0, 2, 0]] + [-4, 0])).unwrap();
        tree0.add_child_node(0, 1, aff!([[2, 0, 0], [0, 0, 2]] + [-8, 1])).unwrap();

        let mut tree1 = AffTree::<2>::from_aff(aff!([[0.5, 0.]] + [-1.]));
        tree1.add_child_node(0, 0, aff!([[3, 0], [0, 3]] + [5, 0])).unwrap();
        tree1.add_child_node(0, 1, aff!([[-3, 0], [0, -3]] + [0, 5])).unwrap();

        tree0.compose::<true, false>(&tree1);

        eprintln!("{}", &tree0);

        assert_eq!(
            tree0.tree.node_value(0).unwrap().aff,
            aff!([1, 0, 0] + 2)
        );

        assert_eq!(
            value_at!(tree0, 0),
            aff!([[6, 0, 0], [0, 6, 0]] + [-12 + 5, 0])
        );

        assert_eq!(
            value_at!(tree0, 1),
            aff!([[-6, 0, 0], [0, 0, -6]] + [24, -3 + 5])
        );
    }

    #[test]
    fn test_terminal_tree() {
        init_logger();

        let mut dd0 = AffTree::<2>::from_slice(&arr1(&[f64::NAN, 0.5]));

        let dd1 = AffTree::<2>::from_poly(Polytope::hypercube(2, 1.0), AffFunc::identity(2), None)
            .unwrap();

        dd0.compose::<false, false>(&dd1);

        assert_eq!(dd0.len(), 5);
    }
}
