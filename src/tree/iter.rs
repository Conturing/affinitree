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

//! Graph iterators for trees

use std::collections::VecDeque;

use itertools::enumerate;

use super::graph::{Label, Tree, TreeIndex};

/// A collection of information provided by depth-first traversal of graphs.
#[derive(Clone, Copy, Debug)]
pub struct DfsNodeData {
    /// Current depth in the tree, which is the number of ancestors the current node has.
    pub depth: usize,
    /// Index of the current node for look-up.
    pub index: TreeIndex,
    /// Number of siblings remaining to be visited.
    /// Can be used to update information of the parent after all children have been visited.
    pub n_remaining: usize,
}

impl DfsNodeData {
    /// Unpack this struct into a tuple.
    ///
    /// Useful for chaining iterators.
    pub fn extract(self) -> (usize, TreeIndex, usize) {
        (self.depth, self.index, self.n_remaining)
    }
}

/// Traverses the specified tree.
///
/// For each visited node, an element of type [``DfsNodeData``] is returned listing
/// its index, depth, and the number of siblings that have to be visited (for clean up of parent).
/// This traversal is optimized for acyclic graphs and will not terminate if this assumption is
/// broken.
///
/// This traversal does not keep a reference to the graph, thereby allowing mutual access to the
/// graph during traversal.
pub trait TraversalMut {
    type Item;

    /// Creates a new traversal over the given ``tree`` starting at the given ``root``.
    fn new<N, const K: usize>(tree: &Tree<N, K>, root: TreeIndex) -> Self;

    /// Creates a new iterator using this traversal pattern.
    /// The iterator keeps a reference to the tree, inhibiting mutual access to it during the iteration.
    fn iter<N, const K: usize>(tree: &Tree<N, K>, root: TreeIndex) -> TraversalIter<'_, Self, N, K>
    where
        Self: Sized,
    {
        TraversalIter::from(Self::new(tree, root), tree)
    }

    /// Returns the next node in this traversal together with the nodes's depth in the tree, or None if the traversal is done.
    ///
    /// Note: The same graph must be provided in all calls.
    fn next<N, const K: usize>(&mut self, tree: &Tree<N, K>) -> Option<Self::Item>;

    /// Alters the iteration such that the subtree of the current node is skipped.
    fn skip_subtree(&mut self);

    /// Provides a hint for the number of remaining nodes, see [``Iterator::size_hint``]
    fn size_hint(&self) -> (usize, Option<usize>);
}

/// A wrapper of [``TraversalMut``] that implements the iterator interface by borrowing
/// the underlying tree. As such, the iterator keeps an active reference to the tree,
/// which prevents mutual access to the tree during iteration.
#[derive(Debug)]
pub struct TraversalIter<'a, T: TraversalMut, N, const K: usize> {
    traversal: T,
    tree: &'a Tree<N, K>,
}

impl<'a, T: TraversalMut, N, const K: usize> TraversalIter<'a, T, N, K> {
    /// Creates a new iterator over the given ``tree`` starting at the given ``root``.
    pub fn new(tree: &'a Tree<N, K>, root: TreeIndex) -> TraversalIter<'a, T, N, K> {
        TraversalIter {
            traversal: T::new(tree, root),
            tree,
        }
    }

    /// Creates a new iterator with traversal pattern given by ``traversal`` over the given ``tree``.
    pub fn from(traversal: T, tree: &'a Tree<N, K>) -> TraversalIter<'a, T, N, K> {
        TraversalIter { traversal, tree }
    }

    /// Alters the iteration such that the subtree of the current node is skipped.
    pub fn skip_subtree(&mut self) {
        self.traversal.skip_subtree()
    }
}

impl<T: TraversalMut, N, const K: usize> Iterator for TraversalIter<'_, T, N, K> {
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.traversal.next(self.tree)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.traversal.size_hint()
    }
}

/// A depth-first traversal for [``Tree``].
#[derive(Clone, Debug)]
pub struct DfsPre {
    pub(super) stack: Vec<DfsNodeData>,
    pub(super) last_push: usize,
    pub size_lb: usize,
    pub size_ub: usize,
}

impl TraversalMut for DfsPre {
    type Item = DfsNodeData;

    #[inline]
    fn new<N, const K: usize>(tree: &Tree<N, K>, root: TreeIndex) -> DfsPre {
        DfsPre {
            stack: vec![DfsNodeData {
                depth: 0,
                index: root,
                n_remaining: 0,
            }],
            last_push: 0,
            size_lb: if root == tree.get_root_idx() {
                tree.len()
            } else {
                0
            },
            size_ub: tree.len(),
        }
    }

    fn skip_subtree(&mut self) {
        self.size_lb = self.stack.len();
        self.size_ub -= self.last_push;
        for _ in 0..self.last_push {
            self.stack.pop();
        }
    }

    fn next<N, const K: usize>(&mut self, tree: &Tree<N, K>) -> Option<DfsNodeData> {
        let data = self.stack.pop()?;
        let node = tree
            .tree_node(data.index)
            .expect("node indicies should stay valid while traversing the tree");

        self.last_push = 0;
        for (n_remaining, child) in node.children.iter().rev().flatten().enumerate() {
            self.stack.push(DfsNodeData {
                depth: data.depth + 1,
                index: *child,
                n_remaining,
            });
            self.last_push += 1;
        }

        self.size_lb = self.size_lb.saturating_sub(1);
        self.size_ub = self.size_ub.saturating_sub(1);
        Some(data)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size_lb, Some(self.size_ub))
    }
}

/// Representation of an edge in [``Tree``].
#[derive(Debug)]
pub struct EdgeData {
    pub src: TreeIndex,
    pub label: Label,
    pub dest: TreeIndex,
}

/// A depth-first traversal for [``Tree``] returning the current edge instead of the node.
#[derive(Clone, Debug)]
pub struct DfsEdge {
    pub(super) stack: Vec<(usize, TreeIndex, Label, TreeIndex)>,
    pub(super) last_push: usize,
    pub size_lb: usize,
    pub size_ub: usize,
}

impl TraversalMut for DfsEdge {
    type Item = EdgeData;

    #[inline]
    fn new<N, const K: usize>(tree: &Tree<N, K>, root: TreeIndex) -> DfsEdge {
        let mut stack = Vec::with_capacity(K);
        let mut last_push = 0;
        for ed in tree.children(tree.get_root_idx()).rev() {
            stack.push((1, root, ed.label, ed.target_idx));
            last_push += 1;
        }
        DfsEdge {
            stack,
            last_push,
            size_lb: if root == tree.get_root_idx() {
                tree.len()
            } else {
                0
            },
            size_ub: tree.len(),
        }
    }

    fn skip_subtree(&mut self) {
        self.size_lb = self.stack.len();
        self.size_ub -= self.last_push;
        for _ in 0..self.last_push {
            self.stack.pop();
        }
    }

    fn next<N, const K: usize>(&mut self, tree: &Tree<N, K>) -> Option<Self::Item> {
        let (depth, src_idx, label, dest_idx) = self.stack.pop()?;
        let node = tree.tree_node(dest_idx).unwrap();

        self.last_push = 0;
        for (label, child) in enumerate(node.children.iter()).rev() {
            if let Some(val) = child {
                self.stack.push((depth + 1, dest_idx, label, *val));
                self.last_push += 1;
            }
        }

        self.size_lb = self.size_lb.saturating_sub(1);
        self.size_ub = self.size_ub.saturating_sub(1);
        Some(EdgeData {
            src: src_idx,
            label,
            dest: dest_idx,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size_lb, Some(self.size_ub))
    }
}

/// A breath-first traversal for [``Tree``].
#[derive(Clone, Debug)]
pub struct Bfs {
    pub(super) queue: VecDeque<DfsNodeData>,
    pub(super) last_push: usize,
    pub size_lb: usize,
    pub size_ub: usize,
}

impl TraversalMut for Bfs {
    type Item = DfsNodeData;

    #[inline]
    fn new<N, const K: usize>(tree: &Tree<N, K>, root: TreeIndex) -> Bfs {
        Bfs {
            queue: VecDeque::from([DfsNodeData {
                depth: 0,
                index: root,
                n_remaining: 0,
            }]),
            last_push: 0,
            size_lb: if root == tree.get_root_idx() {
                tree.len()
            } else {
                0
            },
            size_ub: tree.len(),
        }
    }

    fn skip_subtree(&mut self) {
        self.size_lb = self.queue.len();
        self.size_ub -= self.last_push;
        for _ in 0..self.last_push {
            self.queue.pop_back();
        }
    }

    fn next<N, const K: usize>(&mut self, tree: &Tree<N, K>) -> Option<DfsNodeData> {
        let data = self.queue.pop_front()?;
        let node = tree.tree_node(data.index).unwrap();

        self.last_push = 0;
        for (n_remaining, child) in node.children.iter().flatten().enumerate() {
            self.queue.push_back(DfsNodeData {
                depth: data.depth + 1,
                index: *child,
                n_remaining,
            });
            self.last_push += 1;
        }

        self.size_lb = self.size_lb.saturating_sub(1);
        self.size_ub = self.size_ub.saturating_sub(1);
        Some(data)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size_lb, Some(self.size_ub))
    }
}

#[allow(unused_variables)]
#[cfg(test)]
mod test {

    use assertables::*;

    use super::*;
    use crate::tree::iter::{Bfs, DfsEdge, DfsPre, TraversalMut};

    #[test]
    pub fn test_dfs_node_order() {
        let mut tree = Tree::<usize, 2>::new();

        let z = tree.add_root(10); // 0
        let c0 = tree.add_child_node(z, 0, 11).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, 12).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, 13).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, 14).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, 15).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, 16).unwrap(); // 6
        let l2 = tree.add_child_node(l0, 1, 17).unwrap(); // 7
        let l2r = tree.add_child_node(l2, 0, 18).unwrap(); // 8
        let l2l = tree.add_child_node(l2, 1, 19).unwrap(); // 9

        let iter = DfsPre::iter(&tree, z);
        let nodes = Vec::from_iter(iter.map(|data| data.index));

        assert_eq!(nodes, vec![z, c0, l0, l2, l2r, l2l, l1, c1, r0, r1]);
    }

    #[test]
    pub fn test_dfs_skip_subtree() {
        let mut tree = Tree::<usize, 2>::new();

        let z = tree.add_root(10); // 0
        let c0 = tree.add_child_node(z, 0, 11).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, 12).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, 13).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, 14).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, 15).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, 16).unwrap(); // 6
        let l2 = tree.add_child_node(l0, 1, 17).unwrap(); // 7
        let l2r = tree.add_child_node(l2, 0, 18).unwrap(); // 8
        let l2l = tree.add_child_node(l2, 1, 19).unwrap(); // 9

        let mut iter = DfsPre::iter(&tree, z);
        iter.next(); // z
        iter.next(); // c0

        // skip descendants of c0, i.e., l0, l2, l2r, l2l, l1
        iter.skip_subtree();
        let nodes = Vec::from_iter(iter.map(|data| data.index));

        assert_eq!(nodes, vec![c1, r0, r1]);
    }

    #[test]
    pub fn test_dfs_remaining() {
        let mut tree = Tree::<usize, 2>::new();

        let z = tree.add_root(10); // 0
        let c0 = tree.add_child_node(z, 0, 11).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, 12).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, 13).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, 14).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, 15).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, 16).unwrap(); // 6
        let l2 = tree.add_child_node(l0, 1, 17).unwrap(); // 7
        let l2r = tree.add_child_node(l2, 0, 18).unwrap(); // 8
        let l2l = tree.add_child_node(l2, 1, 19).unwrap(); // 9

        let iter = DfsPre::iter(&tree, z);
        let remaining = Vec::from_iter(iter.map(|data| data.n_remaining));

        //                         z, c0, l0, l2, l2r, l2l, l1, c1, r0, r1
        assert_eq!(remaining, vec![0, 1, 1, 0, 1, 0, 0, 0, 1, 0]);
    }

    #[test]
    pub fn test_dfs_size_hint() {
        let mut tree = Tree::<usize, 2>::new();

        let z = tree.add_root(10); // 0
        let c0 = tree.add_child_node(z, 0, 11).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, 12).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, 13).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, 14).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, 15).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, 16).unwrap(); // 6
        let l2 = tree.add_child_node(l0, 1, 17).unwrap(); // 7
        let l2r = tree.add_child_node(l2, 0, 18).unwrap(); // 8
        let l2l = tree.add_child_node(l2, 1, 19).unwrap(); // 9

        let mut iter = DfsPre::iter(&tree, z);
        iter.next(); // z
        iter.next(); // c0

        assert_eq!(iter.size_hint(), (8, Some(8)));

        // skip descendants of c0, i.e., l0, l2, l2r, l2r, l1
        iter.skip_subtree();

        assert_le!(iter.size_hint().0, 3);
        assert_ge!(iter.size_hint().1.unwrap(), 3);
    }

    #[test]
    pub fn test_dfs_edge_iter() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, ()).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, ()).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, ()).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, ()).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, ()).unwrap(); // 6
        let rr1 = tree.add_child_node(r1, 1, ()).unwrap(); // 7

        let iter = DfsEdge::iter(&tree, z);
        let nodes = Vec::from_iter(iter.map(|edge| (edge.src, edge.label, edge.dest)));

        assert_eq!(
            nodes,
            vec![
                (z, 0, c0),
                (c0, 0, l0),
                (c0, 1, l1),
                (z, 1, c1),
                (c1, 0, r0),
                (c1, 1, r1),
                (r1, 1, rr1)
            ]
        );

        let mut iter = DfsEdge::iter(&tree, z);
        iter.next();
        // skip descendants of c0, i.e., l0 and l1
        iter.skip_subtree();

        assert_eq!(iter.next().unwrap().dest, c1);
    }

    #[test]
    fn test_bfs_node_order() {
        let mut tree = Tree::<usize, 2>::new();

        let z = tree.add_root(10); // 0
        let c0 = tree.add_child_node(z, 0, 11).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, 12).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, 13).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, 14).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, 15).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, 16).unwrap(); // 6
        let l2 = tree.add_child_node(l0, 1, 17).unwrap(); // 7
        let l2r = tree.add_child_node(l2, 0, 18).unwrap(); // 8
        let l2l = tree.add_child_node(l2, 1, 19).unwrap(); // 9

        let iter = Bfs::iter(&tree, z);
        let nodes = Vec::from_iter(iter.map(|data| data.index));

        assert_eq!(nodes, vec![z, c0, c1, l0, l1, r0, r1, l2, l2r, l2l]);
    }

    #[test]
    pub fn test_bfs_skip_subtree() {
        let mut tree = Tree::<usize, 2>::new();

        let z = tree.add_root(10); // 0
        let c0 = tree.add_child_node(z, 0, 11).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, 12).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, 13).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, 14).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, 15).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, 16).unwrap(); // 6
        let l2 = tree.add_child_node(l0, 1, 17).unwrap(); // 7
        let l2r = tree.add_child_node(l2, 0, 18).unwrap(); // 8
        let l2l = tree.add_child_node(l2, 1, 19).unwrap(); // 9

        let mut iter = Bfs::iter(&tree, z);
        iter.next(); // z
        iter.next(); // c0

        // skip descendants of c0, i.e., l0, l2, l2r, l2l, l1
        iter.skip_subtree();
        let nodes = Vec::from_iter(iter.map(|data| data.index));

        assert_eq!(nodes, vec![c1, r0, r1]);
    }
}
