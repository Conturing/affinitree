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

use super::afftree::*;
use crate::tree::iter::{Bfs, TraversalMut};

/// # Redundancy Elimination
impl AffTree<2> {
    /// Performs simple redundancy elimination on this decision tree.
    ///
    /// A predicate (decision) is redundant if all of its children are identical terminals.
    pub fn reduce(&mut self) {
        let mut elements = Vec::from_iter(Bfs::iter(&self.tree, self.tree.get_root_idx()));
        elements.reverse();

        for value in elements.into_iter() {
            if let Some(node) = self.tree.tree_node(value.index) {
                // skip terminals
                if node.children_iter().count() == 0 {
                    continue;
                }
                // skip root node
                if value.index == self.tree.get_root_idx() {
                    continue;
                }

                if let (Some(left_idx), Some(right_idx)) = (node.children[0], node.children[1]) {
                    let left = self.tree.tree_node(left_idx).unwrap();
                    let right = self.tree.tree_node(right_idx).unwrap();

                    // can only merge if both children are terminals
                    if left.children_iter().count() != 0 || right.children_iter().count() != 0 {
                        continue;
                    }

                    if left.value.aff == right.value.aff {
                        self.tree.remove_child(value.index, 1);
                        self.tree.merge_child_with_parent(value.index, 0);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use assertables::*;
    use itertools::Itertools;
    use ndarray::{arr1, arr2};

    use super::*;
    use crate::aff;
    use crate::linalg::affine::AffFunc;

    #[test]
    fn test_reduce() {
        let mut dd = AffTree::<2>::from_aff(AffFunc::from_mats(arr2(&[[2., 1.]]), arr1(&[-1.])));

        dd.add_child_node(0, 1, aff!([[1., 2.]] + [-1.5])); // 1
        dd.add_child_node(1, 1, aff!([[0.5, 5.]] + [1.0])); // 2
        dd.add_child_node(2, 1, aff!([[3., -1.]] + [0.])); // 3
        dd.add_child_node(3, 1, aff!([[-1., -1.]] + [6.])); // 4
        dd.add_child_node(4, 0, aff!([[-1., 7.]] + [4.])); // 5
        dd.add_child_node(5, 1, aff!([[-2., -0.2]] + [3.])); // 6
        // feasible
        dd.add_child_node(6, 0, aff!([[0., 0.]] + [1.])); // 7
        // infeasible
        dd.add_child_node(6, 1, aff!([[0., 0.]] + [0.])); // 8

        dd.reduce();

        assert_eq!(dd.len(), 9);
    }

    #[test]
    fn test_reduce2() {
        let mut dd = AffTree::<2>::from_aff(AffFunc::from_mats(arr2(&[[2., 1.]]), arr1(&[-1.])));

        dd.add_child_node(0, 1, aff!([[1., 2.]] + [-1.5])); // 1
        dd.add_child_node(1, 1, aff!([[0.5, 5.]] + [1.0])); // 2
        dd.add_child_node(2, 1, aff!([[3., -1.]] + [0.])); // 3
        dd.add_child_node(3, 1, aff!([[-1., -1.]] + [6.])); // 4
        dd.add_child_node(4, 0, aff!([[-1., 7.]] + [4.])); // 5
        dd.add_child_node(5, 1, aff!([[-2., -0.2]] + [3.])); // 6
        // feasible
        dd.add_child_node(6, 0, aff!([[0., 0.]] + [0.])); // 7
        // infeasible
        dd.add_child_node(6, 1, aff!([[0., 0.]] + [0.])); // 8

        dd.add_child_node(5, 0, aff!([[0., 0.]] + [0.])); // 9
        dd.add_child_node(4, 1, aff!([[0., 0.]] + [0.]));

        dd.reduce();

        assert_eq!(dd.len(), 5);
        let nodes = dd.tree.node_indices().collect_vec();
        assert_contains!(nodes, &9);
        assert_not_contains!(nodes, &7);
        assert_not_contains!(nodes, &8);
        assert_not_contains!(nodes, &10);
    }

    #[test]
    fn test_reduce3() {
        let mut dd = AffTree::<2>::from_aff(AffFunc::from_mats(arr2(&[[2., 1.]]), arr1(&[-1.])));

        dd.add_child_node(0, 1, aff!([[1., 2.]] + [-1.5])); // 1
        dd.add_child_node(1, 1, aff!([[0.5, 5.]] + [1.0])); // 2
        dd.add_child_node(2, 1, aff!([[3., -1.]] + [0.])); // 3
        dd.add_child_node(3, 1, aff!([[-1., -1.]] + [6.])); // 4
        dd.add_child_node(4, 0, aff!([[-1., 7.]] + [4.])); // 5
        dd.add_child_node(5, 1, aff!([[-2., -0.2]] + [3.])); // 6
        // feasible
        dd.add_child_node(6, 0, aff!([[0., 0.]] + [0.])); // 7
        // infeasible
        dd.add_child_node(6, 1, aff!([[0., 0.]] + [0.])); // 8

        dd.add_child_node(5, 0, aff!([[0., 0.]] + [1.]));
        dd.add_child_node(4, 1, aff!([[0., 0.]] + [0.]));

        dd.reduce();

        assert_eq!(dd.len(), 9);
        let nodes = dd.tree.node_indices().collect_vec();
        assert_contains!(nodes, &7);
        assert_not_contains!(nodes, &8);
        assert_not_contains!(nodes, &6);
    }
}
