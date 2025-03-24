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

//! A collection of AffTree iterators

use log::trace;

use super::node::AffContent;
use crate::linalg::affine::Polytope;
use crate::tree::graph::{Tree, TreeIndex};
use crate::tree::iter::{DfsNodeData, DfsPre, TraversalMut};

/// A depth-first iterator over an [``crate::pwl::afftree::AffTree``] instance that also provides the path condition
/// in form of a [``Polytope``].
#[derive(Clone, Debug)]
pub struct PolyhedraGen {
    pub(super) predicates: Vec<Polytope>,
    pub(super) iter: DfsPre,
    pub(super) last_depth: usize,
}

impl PolyhedraGen {
    /// Creates a new iterator over the given ``tree`` starting at its root node.
    #[inline]
    pub fn new<const K: usize>(tree: &Tree<AffContent, K>) -> PolyhedraGen {
        Self::with_root(tree, tree.get_root_idx())
    }

    /// Creates a new iterator over the given ``tree`` starting at the given ``root``.
    #[inline]
    pub fn with_root<const K: usize>(tree: &Tree<AffContent, K>, root: TreeIndex) -> PolyhedraGen {
        PolyhedraGen {
            predicates: Vec::with_capacity((tree.len() as f64).log2().ceil() as usize),
            iter: DfsPre::new(tree, root),
            last_depth: 0,
        }
    }

    /// Alters the iteration such that the subtree of the current node is skipped.
    #[inline]
    pub fn skip_subtree(&mut self) {
        self.iter.skip_subtree()
    }

    /// Returns the next node in the iteration, together with the current depth and the path condition.
    pub fn next<const K: usize>(
        &mut self,
        tree: &Tree<AffContent, K>,
    ) -> Option<(DfsNodeData, &Vec<Polytope>)> {
        let data = self.iter.next(tree)?;
        let (depth, node_idx, _) = data.extract();

        if depth <= self.last_depth {
            let diff = 1 + self.last_depth - depth;
            trace!("Removing {} from stack", diff);
            for _ in 0..diff {
                self.predicates.pop();
            }
        }
        self.last_depth = depth;

        if let Ok(edg) = tree.parent(node_idx) {
            trace!(
                "Edge {} -{}-> {}",
                edg.source_idx, edg.label, edg.target_idx
            );
            let factor = match edg.label {
                1 => 1.0,
                0 => -1.0,
                _ => panic!("label should be 0 or 1, but got {}", &edg.label),
            };
            let aff = &tree.node_value(edg.source_idx).ok()?.aff;
            let poly = Polytope::from_mats(&aff.mat * factor, &aff.bias * factor);
            self.predicates.push(poly);
        }

        Some((data, &self.predicates))
    }

    /// Returns the current path conditions in the order of the path.
    pub fn current_polytope(&self) -> &Vec<Polytope> {
        &self.predicates
    }
}

/// Wrapper of [``PolyhedraGen``] that implements the [``Iterator``] trait.
///
/// While iterators are more convenient to use, they require certain borrowing patterns
/// making it impossible to modify the tree during iteration.
#[derive(Debug)]
pub struct PolyhedraIter<'a, const K: usize> {
    pub iter: PolyhedraGen,
    pub tree: &'a Tree<AffContent, K>,
}

impl<const K: usize> PolyhedraIter<'_, K> {
    /// Creates a new iterator over the given ``tree`` starting at its root node.
    pub fn new(tree: &Tree<AffContent, K>) -> PolyhedraIter<'_, K> {
        PolyhedraIter {
            iter: PolyhedraGen::new(tree),
            tree,
        }
    }

    /// Alters the iteration such that the subtree of the current node is skipped.
    pub fn skip_subtree(&mut self) {
        self.iter.skip_subtree()
    }
}

impl<const K: usize> Iterator for PolyhedraIter<'_, K> {
    type Item = (usize, TreeIndex, usize, Vec<Polytope>);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next(self.tree)
            .map(|(data, poly)| (data.depth, data.index, data.n_remaining, poly.clone()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.tree.len(), Some(self.tree.len()))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;
    use crate::pwl::afftree::AffTree;
    use crate::{aff, poly};

    fn construct_tree() -> AffTree<2> {
        let mut tree = AffTree::<2>::from_aff(aff!([1, 0, 0] + 2)); // idx 0
        tree.add_child_node(0, 0, aff!([1, 0, 0] + 1)).unwrap(); // idx 1
        tree.add_child_node(0, 1, aff!([1, 0, 0] + 3)).unwrap(); // idx 2
        tree.add_child_node(1, 0, aff!([[6, 0, 0], [0, 6, 0]] + [12 + 5, 0]))
            .unwrap(); // idx 3
        tree.add_child_node(1, 1, aff!([[-6, 0, 0], [0, -6, 0]] + [-12, 5]))
            .unwrap(); // idx 4
        tree.add_child_node(2, 0, aff!([[6, 0, 0], [0, 0, 6]] + [24 + 5, 3]))
            .unwrap(); // idx 5
        tree.add_child_node(2, 1, aff!([[-6, 0, 0], [0, 0, -6]] + [-24, -3 + 5]))
            .unwrap(); // idx 6

        tree
    }

    #[test]
    fn test_polyhedra() {
        let tree = construct_tree();
        let mut iter = PolyhedraGen::new(&tree.tree);

        let (dfs_data, polytope) = iter.next(&tree.tree).unwrap();
        assert_eq!(dfs_data.index, 0);
        assert_eq!(polytope, &vec![]);

        let (dfs_data, polytope) = iter.next(&tree.tree).unwrap();
        assert_eq!(dfs_data.index, 1);
        assert_eq!(polytope, &vec![poly!([[-1, 0, 0]] < [-2])]);

        let (dfs_data, polytope) = iter.next(&tree.tree).unwrap();
        assert_eq!(dfs_data.index, 3);
        assert_eq!(
            polytope,
            &vec![poly!([[-1, 0, 0]] < [-2]), poly!([[-1, 0, 0]] < [-1])]
        );

        // assert consistency with other tree methods
        assert_eq!(
            tree.find_terminal(tree.tree.get_root(), &arr1(&[3., 0., 0.]))
                .unwrap()
                .1,
            vec![0, 0]
        );
        assert!(Polytope::intersection_n(3, polytope).contains(&arr1(&[3., 0., 0.])));

        let (dfs_data, polytope) = iter.next(&tree.tree).unwrap();
        assert_eq!(dfs_data.index, 4);
        assert_eq!(
            polytope,
            &vec![poly!([[-1, 0, 0]] < [-2]), poly!([[1, 0, 0]] < [1])]
        );

        let (dfs_data, polytope) = iter.next(&tree.tree).unwrap();
        assert_eq!(dfs_data.index, 2);
        assert_eq!(polytope, &vec![poly!([[1, 0, 0]] < [2])]);

        let (dfs_data, polytope) = iter.next(&tree.tree).unwrap();
        assert_eq!(dfs_data.index, 5);
        assert_eq!(
            polytope,
            &vec![poly!([[1, 0, 0]] < [2]), poly!([[-1, 0, 0]] < [-3])]
        );

        let (dfs_data, polytope) = iter.next(&tree.tree).unwrap();
        assert_eq!(dfs_data.index, 6);
        assert_eq!(
            polytope,
            &vec![poly!([[1, 0, 0]] < [2]), poly!([[1, 0, 0]] < [3])]
        );
    }
}
