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

//! Graph iterators for k-trees

use itertools::enumerate;

use super::graph::{EdgeReference, Label, Tree, TreeIndex};

#[derive(Clone, Copy, Debug)]
pub struct DfsNodeData {
    pub depth: usize,
    pub index: TreeIndex,
    pub n_remaining: usize,
}

impl DfsNodeData {
    pub fn extract(self) -> (usize, TreeIndex, usize) {
        (self.depth, self.index, self.n_remaining)
    }
}

/// DfsPre does not handle cycles. If any cycles are present, then DfsPre will never terminate.
#[derive(Clone, Debug)]
pub struct DfsPre {
    pub(super) stack: Vec<DfsNodeData>,
    pub(super) last_push: usize,
}

impl DfsPre {
    #[inline]
    pub fn new<N, const K: usize>(tree: &Tree<N, K>) -> DfsPre {
        Self::with_root(tree, tree.get_root_idx())
    }

    #[inline]
    pub fn with_root<N, const K: usize>(_tree: &Tree<N, K>, root: TreeIndex) -> DfsPre {
        DfsPre {
            stack: vec![DfsNodeData {
                depth: 0,
                index: root,
                n_remaining: 0,
            }],
            last_push: 0,
        }
    }

    pub fn skip_subtree(&mut self) {
        for _ in 0..self.last_push {
            self.stack.pop();
        }
    }

    /// Returns the next node of this DfsPre together with its depth in the tree, or None if the traversal is done.
    ///
    /// Note: The same graph must be provided in all calls.
    pub fn next<N, const K: usize>(&mut self, tree: &Tree<N, K>) -> Option<DfsNodeData> {
        let data = self.stack.pop()?;
        let node = tree.tree_node(data.index).unwrap();

        self.last_push = 0;
        for (n_remaining, child) in node.children.iter().rev().flatten().enumerate() {
            self.stack.push(DfsNodeData {
                depth: data.depth + 1,
                index: *child,
                n_remaining: n_remaining,
            });
            self.last_push += 1;
        }

        Some(data)
    }

    pub fn iter<N, const K: usize>(
        self,
        tree: &Tree<N, K>,
    ) -> impl Iterator<Item = (usize, TreeIndex, usize)> + '_ {
        DfsPreIter {
            iter: self,
            tree,
            size_lb: 0,
            size_ub: tree.len(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DfsPreIter<'a, N, const K: usize> {
    pub iter: DfsPre,
    pub tree: &'a Tree<N, K>,
    pub size_lb: usize,
    pub size_ub: usize,
}

impl<'a, N, const K: usize> DfsPreIter<'a, N, K> {
    pub fn new(tree: &Tree<N, K>) -> DfsPreIter<'_, N, K> {
        DfsPreIter {
            iter: DfsPre::new(tree),
            tree,
            size_lb: tree.len(),
            size_ub: tree.len(),
        }
    }

    pub fn skip_subtree(&mut self) {
        self.size_lb = self.iter.stack.len();
        self.size_ub -= self.iter.last_push;
        self.iter.skip_subtree()
    }
}

impl<'a, N, const K: usize> Iterator for DfsPreIter<'a, N, K> {
    type Item = (usize, TreeIndex, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.size_lb = self.size_lb.saturating_sub(1);
        self.size_ub = self.size_ub.saturating_sub(1);
        self.iter.next(self.tree).map(DfsNodeData::extract)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size_lb, Some(self.size_ub))
    }
}

#[derive(Clone, Debug)]
pub struct DFSEdgeIter<'a, N, const K: usize> {
    pub(super) tree: &'a Tree<N, K>,
    pub(super) stack: Vec<(usize, TreeIndex, Label, TreeIndex)>,
    pub(super) last_push: usize,
}

impl<'a, N, const K: usize> DFSEdgeIter<'a, N, K> {
    #[inline]
    pub fn new(tree: &'a Tree<N, K>) -> DFSEdgeIter<'a, N, K> {
        Self::with_root(tree, tree.get_root_idx())
    }

    #[inline]
    pub fn with_root(tree: &'a Tree<N, K>, root: TreeIndex) -> DFSEdgeIter<'a, N, K> {
        let mut stack = Vec::with_capacity(K);
        let mut last_push = 0;
        for ed in tree.children(tree.get_root_idx()).rev() {
            stack.push((1, root, ed.label, ed.target_idx));
            last_push += 1;
        }
        DFSEdgeIter {
            tree: tree,
            stack: stack,
            last_push: last_push,
        }
    }

    pub fn skip_subtree(&mut self) {
        for _ in 0..self.last_push {
            self.stack.pop();
        }
    }
}

impl<'a, N, const K: usize> Iterator for DFSEdgeIter<'a, N, K> {
    type Item = (usize, EdgeReference<'a, N>);

    fn next(&mut self) -> Option<Self::Item> {
        let (depth, src_idx, label, dest_idx) = self.stack.pop()?;
        let node = self.tree.tree_node(dest_idx).unwrap();

        self.last_push = 0;
        for (label, child) in enumerate(node.children.iter()).rev() {
            if let Some(val) = child {
                self.stack.push((depth + 1, dest_idx, label, *val));
                self.last_push += 1;
            }
        }

        Some((
            depth,
            EdgeReference {
                source_idx: src_idx,
                source_value: self.tree.node_value(src_idx).unwrap(),
                label: label,
                target_idx: dest_idx,
                target_value: self.tree.node_value(dest_idx).unwrap(),
            },
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.tree.len(), Some(self.tree.len()))
    }
}

#[allow(unused_variables)]
#[cfg(test)]
mod test {

    use assertables::*;

    use crate::tree::iter::DfsPreIter;

    use super::Tree;

    #[test]
    pub fn test_node_order() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()); // 1
        let c1 = tree.add_child_node(z, 1, ()); // 2
        let l0 = tree.add_child_node(c0, 0, ()); // 3
        let l1 = tree.add_child_node(c0, 1, ()); // 4
        let r0 = tree.add_child_node(c1, 0, ()); // 5
        let r1 = tree.add_child_node(c1, 1, ()); // 6
        let rr1 = tree.add_child_node(r1, 1, ()); // 7

        let iter = DfsPreIter::new(&tree);
        let nodes = Vec::from_iter(iter.map(|(_, id, _)| id));

        assert_eq!(nodes, vec![z, c0, l0, l1, c1, r0, r1, rr1]);
    }

    #[test]
    pub fn test_skip_subtree() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()); // 1
        let c1 = tree.add_child_node(z, 1, ()); // 2
        let l0 = tree.add_child_node(c0, 0, ()); // 3
        let l1 = tree.add_child_node(c0, 1, ()); // 4
        let r0 = tree.add_child_node(c1, 0, ()); // 5
        let r1 = tree.add_child_node(c1, 1, ()); // 6
        let rr1 = tree.add_child_node(r1, 1, ()); // 7

        let mut iter = DfsPreIter::new(&tree);
        iter.next(); // z
        iter.next(); // c0

        // skip descendants of c0, i.e., l0 and l1
        iter.skip_subtree();

        assert_eq!(iter.next().unwrap().1, c1);
    }

    #[test]
    pub fn test_remaining() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()); // 1
        let c1 = tree.add_child_node(z, 1, ()); // 2
        let l0 = tree.add_child_node(c0, 0, ()); // 3
        let l1 = tree.add_child_node(c0, 1, ()); // 4
        let r0 = tree.add_child_node(c1, 0, ()); // 5
        let r1 = tree.add_child_node(c1, 1, ()); // 6
        let rr1 = tree.add_child_node(r1, 1, ()); // 7

        let iter = DfsPreIter::new(&tree);
        let remaining = Vec::from_iter(iter.map(|(_, _, remaining)| remaining));

        //                         z, c0, l0, l1, c1, r0, r1, rr1
        assert_eq!(remaining, vec![0, 1, 1, 0, 0, 1, 0, 0]);
    }

    #[test]
    pub fn test_size_hint() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()); // 1
        let c1 = tree.add_child_node(z, 1, ()); // 2
        let l0 = tree.add_child_node(c0, 0, ()); // 3
        let l1 = tree.add_child_node(c0, 1, ()); // 4
        let r0 = tree.add_child_node(c1, 0, ()); // 5
        let r1 = tree.add_child_node(c1, 1, ()); // 6
        let rr1 = tree.add_child_node(r1, 1, ()); // 7

        let mut iter = DfsPreIter::new(&tree);
        iter.next(); // z
        iter.next(); // c0

        assert_eq!(iter.size_hint(), (6, Some(6)));

        // skip descendants of c0, i.e., l0 and l1
        iter.skip_subtree();

        assert_le!(iter.size_hint().0, 4);
        assert_ge!(iter.size_hint().1.unwrap(), 4);
    }
}
