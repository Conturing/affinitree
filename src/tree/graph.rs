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

//! Generic tree with constant branching factor

use core::fmt;
use std::{
    mem,
    ops::{Index, IndexMut},
};

use itertools::enumerate;
use slab::Slab;

use super::iter::{DFSEdgeIter, DfsPreIter};

pub type TreeIndex = usize;
pub type Label = usize;

#[derive(Clone, Debug, PartialEq)]
pub struct TreeNode<T, const K: usize> {
    pub value: T,
    pub parent: Option<TreeIndex>,
    pub children: [Option<TreeIndex>; K],
    pub isleaf: bool,
}

impl<T, const K: usize> TreeNode<T, K> {
    #[inline]
    pub fn new(value: T, parent: Option<TreeIndex>) -> TreeNode<T, K> {
        TreeNode {
            value: value,
            parent: parent,
            children: [None; K],
            isleaf: true,
        }
    }

    #[inline]
    pub fn children_iter(&self) -> impl DoubleEndedIterator<Item = (Label, TreeIndex)> + '_ {
        self.children
            .iter()
            .enumerate()
            .filter_map(|(label, child)| child.map(|idx| (label, idx)))
    }

    #[inline]
    pub fn retain_children<F>(&mut self, mut f: F)
    where
        F: FnMut(TreeIndex) -> bool,
    {
        for idx in 0..K {
            if let Some(child_idx) = self.children[idx] {
                if !f(child_idx) {
                    self.children[idx] = None;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeReference<'a, N> {
    pub value: &'a N,
    pub idx: TreeIndex,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeReference<'a, N> {
    pub source_value: &'a N,
    pub source_idx: TreeIndex,
    pub label: Label,
    pub target_value: &'a N,
    pub target_idx: TreeIndex,
}

impl<'a, N> EdgeReference<'a, N> {
    pub fn extract(&self) -> (TreeIndex, &'a N, Label, TreeIndex, &'a N) {
        (
            self.source_idx,
            self.source_value,
            self.label,
            self.target_idx,
            self.target_value,
        )
    }
}

#[derive(Debug, PartialEq)]
pub struct EdgeReferenceMut<'a, N> {
    pub source_value: &'a mut N,
    pub source_idx: TreeIndex,
    pub label: Label,
    pub target_value: &'a mut N,
    pub target_idx: TreeIndex,
}

impl<'a, N> EdgeReferenceMut<'a, N> {
    pub fn extract(&'a mut self) -> (TreeIndex, &'a mut N, Label, TreeIndex, &'a mut N) {
        (
            self.source_idx,
            self.source_value,
            self.label,
            self.target_idx,
            self.target_value,
        )
    }
}

#[derive(Clone)]
pub struct Tree<N, const K: usize> {
    pub(super) arena: Slab<TreeNode<N, K>>,
    pub(super) root: Option<TreeIndex>,
}

impl<N, const K: usize> Default for Tree<N, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, const K: usize> Tree<N, K> {
    /// Constructs a new, empty `Tree<N, K>`.
    ///
    /// The tree will not allocate until nodes are added to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use affinitree::tree::graph::Tree;
    ///
    /// let mut dd: Tree<(), 2> = Tree::new();
    /// ```
    #[inline(always)]
    pub fn new() -> Tree<N, K> {
        Self::with_capacity(0)
    }

    /// Constructs a new, empty `Tree<N, K>` with the specified capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// use affinitree::tree::graph::Tree;
    ///
    /// let mut dd: Tree<(), 2> = Tree::with_capacity(128);
    /// ```
    pub fn with_capacity(capacity: usize) -> Tree<N, K> {
        assert!(
            K >= 2,
            "The tree should at least contain two children! Got {} as argument K.",
            K
        );
        assert!(
            K < usize::MAX,
            "Tree does not support branches of size usize::MAX or larger, got: {}",
            K
        );

        Tree {
            arena: Slab::with_capacity(capacity),
            root: None,
        }
    }

    /// Constructs a new, single-node `Tree<N, K>` with the specified capacity and root node.
    #[inline(always)]
    pub fn with_root(root: N, capacity: usize) -> Tree<N, K> {
        let mut tree = Self::with_capacity(capacity.max(1));
        tree.add_root(root);
        tree
    }

    /// Returns the number of nodes in this tree.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.arena.len()
    }

    /// Returns true if there are no values in this tree.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    /// Returns the number of terminal nodes in this tree
    pub fn num_terminals(&self) -> usize {
        self.terminal_indices().count()
    }

    /// Returns the depth of this tree, that is, the length of its longest path.
    ///
    /// Correspondingly, an empty tree has depth 0, and a tree with only a root node has depth 1.
    pub fn depth(&self) -> usize {
        self.dfs_iter()
            .map(|(depth, _, _)| depth)
            .max()
            .unwrap_or(0)
    }

    #[inline(always)]
    pub fn num_children(&self, node: TreeIndex) -> usize {
        self.arena[node]
            .children
            .iter()
            .filter(|&&x| x.is_some())
            .count()
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.arena.reserve(additional);
    }

    /// Returns the value associated with the node at the given index.
    #[inline(always)]
    pub fn node_value(&self, idx: TreeIndex) -> Option<&N> {
        self.arena.get(idx).map(|nd| &nd.value)
    }

    /// Returns the value associated with the node at the given index.
    #[inline(always)]
    pub fn node_value_mut(&mut self, idx: TreeIndex) -> Option<&mut N> {
        self.arena.get_mut(idx).map(|nd| &mut nd.value)
    }

    /// Returns if the node at the given index has children or not.
    /// If the node does not exist, `None` is returned.
    pub fn is_leaf(&self, idx: TreeIndex) -> Option<bool> {
        self.tree_node(idx).map(|nd| nd.isleaf)
    }

    /// Returns the internal structure used to store the node at the given index.
    ///
    /// **Use with caution**: Provides access to internal representation.
    #[inline(always)]
    pub fn tree_node(&self, idx: TreeIndex) -> Option<&TreeNode<N, K>> {
        self.arena.get(idx)
    }

    /// Returns the internal structure used to store the node at the given index.
    ///
    /// **Use with caution**: Provides mutable access to internal representation.
    #[inline(always)]
    pub fn tree_node_mut(&mut self, idx: TreeIndex) -> Option<&mut TreeNode<N, K>> {
        self.arena.get_mut(idx)
    }

    /// Returns two mutable references to the values associated with the two
    /// nodes at the given indices.
    ///
    /// # Panics
    ///
    /// This function will panic if `idx1` and `idx2` are the same.
    #[inline(always)]
    pub fn tree_node2_mut(
        &mut self,
        idx1: TreeIndex,
        idx2: TreeIndex,
    ) -> Option<(&mut TreeNode<N, K>, &mut TreeNode<N, K>)> {
        self.arena.get2_mut(idx1, idx2)
    }

    /// Returns the edge from the given node to its parent, if it exists. Otherwise `None` is returned.
    ///
    /// To find the corresponding label a linear search is performed in O(K).
    #[inline(always)]
    pub fn parent(&self, node_idx: TreeIndex) -> Option<EdgeReference<'_, N>> {
        let node = self.tree_node(node_idx)?;
        let parent_idx = node.parent?;
        let parent = self.tree_node(parent_idx)?;

        for (label, child) in enumerate(&parent.children) {
            if let Some(child_idx) = child {
                if *child_idx == node_idx {
                    return Some(EdgeReference {
                        source_value: &parent.value,
                        source_idx: parent_idx,
                        label: label,
                        target_value: &node.value,
                        target_idx: node_idx,
                    });
                }
            }
        }

        None
    }

    /// Returns the edge from the given node to its parent, if it exists. Otherwise `None` is returned.
    #[inline(always)]
    pub fn parent_mut(&mut self, node_idx: TreeIndex) -> Option<EdgeReferenceMut<'_, N>> {
        let parent_idx = self.tree_node(node_idx)?.parent?;

        let (node, parent) = self.tree_node2_mut(node_idx, parent_idx)?;

        for (label, child) in enumerate(&parent.children) {
            if let Some(child_idx) = child {
                if child_idx == &node_idx {
                    return Some(EdgeReferenceMut {
                        source_value: &mut parent.value,
                        source_idx: parent_idx,
                        label: label,
                        target_value: &mut node.value,
                        target_idx: node_idx,
                    });
                }
            }
        }

        None
    }

    /// Returns the edge from the given node to the child reached with the given label, if it exists. Otherwise `None` is returned.
    #[inline(always)]
    pub fn child(&self, node_idx: TreeIndex, label: Label) -> Option<EdgeReference<'_, N>> {
        let node = self.tree_node(node_idx)?;
        let child_idx = node.children[label]?;
        let child = self.tree_node(child_idx)?;

        Some(EdgeReference {
            source_value: &node.value,
            source_idx: node_idx,
            label: label,
            target_value: &child.value,
            target_idx: child_idx,
        })
    }

    /// Returns the edge from the given node to the child reached with the given label, if it exists. Otherwise `None` is returned.
    #[inline(always)]
    pub fn child_mut(
        &mut self,
        node_idx: TreeIndex,
        label: Label,
    ) -> Option<EdgeReferenceMut<'_, N>> {
        let child_idx = self.tree_node(node_idx)?.children[label]?;

        let (node, child) = self.tree_node2_mut(node_idx, child_idx)?;

        Some(EdgeReferenceMut {
            source_value: &mut node.value,
            source_idx: node_idx,
            label: label,
            target_value: &mut child.value,
            target_idx: child_idx,
        })
    }

    /// Iterates over the existing children of the given node. For each child, the edge connecting it to its parent is returned.
    ///
    /// Panics if the given index is not associated with a node in this tree.
    #[inline(always)]
    pub fn children(
        &self,
        node_idx: TreeIndex,
    ) -> impl DoubleEndedIterator<Item = EdgeReference<'_, N>> {
        let node = self.tree_node(node_idx).unwrap();
        node.children_iter()
            .map(move |(label, child_idx)| EdgeReference {
                source_value: &node.value,
                source_idx: node_idx,
                label: label,
                target_value: self.node_value(child_idx).unwrap(),
                target_idx: child_idx,
            })
    }

    /// Returns an iterator over the nodes of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    ///
    /// **Use with Caution:** Provides access to internal structures of this tree
    #[inline]
    pub fn node_iter(&self) -> impl DoubleEndedIterator<Item = (TreeIndex, &TreeNode<N, K>)> {
        self.arena.iter()
    }

    /// Returns an iterator over the nodes of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    pub fn nodes(&self) -> impl DoubleEndedIterator<Item = NodeReference<'_, N>> {
        self.arena.iter().map(|(idx, nd)| NodeReference {
            idx: idx,
            value: &nd.value,
        })
    }

    /// Returns an iterator over the terminal nodes of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    pub fn terminals(&self) -> impl DoubleEndedIterator<Item = NodeReference<'_, N>> {
        self.arena
            .iter()
            .filter(|(_, nd)| nd.isleaf)
            .map(|(idx, nd)| NodeReference {
                idx: idx,
                value: &nd.value,
            })
    }

    /// Returns an iterator over the decision nodes of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    pub fn decisions(&self) -> impl DoubleEndedIterator<Item = NodeReference<'_, N>> {
        self.arena
            .iter()
            .filter(|(_, nd)| !nd.isleaf)
            .map(|(idx, nd)| NodeReference {
                idx: idx,
                value: &nd.value,
            })
    }

    /// Returns an iterator over the node indices of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    pub fn node_indices(&self) -> impl DoubleEndedIterator<Item = TreeIndex> + '_ {
        self.arena.iter().map(|(idx, _)| idx)
    }

    /// Returns an iterator over the indices of the terminals of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    pub fn terminal_indices(&self) -> impl DoubleEndedIterator<Item = TreeIndex> + '_ {
        self.arena
            .iter()
            .filter(|(_, nd)| nd.isleaf)
            .map(|(idx, _)| idx)
    }

    /// Returns an iterator over the indices of the terminals of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    pub fn decision_indices(&self) -> impl DoubleEndedIterator<Item = TreeIndex> + '_ {
        self.arena
            .iter()
            .filter(|(_, nd)| !nd.isleaf)
            .map(|(idx, _)| idx)
    }

    /// Returns an iterator over the nodes of this tree.
    /// Nodes are ordered using depth-first-search.
    #[inline]
    pub fn dfs_iter(&self) -> DfsPreIter<'_, N, K> {
        DfsPreIter::new(self)
    }

    /// Returns an iterator over the edges of this tree.
    /// Edges are ordered using depth-first-search.
    #[inline]
    pub fn dfs_edge_iter(&self) -> DFSEdgeIter<'_, N, K> {
        DFSEdgeIter::new(self)
    }

    /// Returns an iterator over the edges of this tree.
    /// No guarantees are made over the ordering of the edges.
    pub fn edge_iter(&self) -> impl DoubleEndedIterator<Item = EdgeReference<'_, N>> {
        self.node_iter().filter_map(|(idx, _)| self.parent(idx))
    }

    /* Old utility methods, shouldn't be used in new code */

    /// Return the label of the edge src_idx -> dest_idx if it exists.
    /// Otherwise, return None.
    pub fn get_label(&self, src_idx: TreeIndex, dest_idx: TreeIndex) -> Option<Label> {
        let src_node = self.tree_node(src_idx)?;
        let dest_node = self.tree_node(dest_idx)?;
        let parent_idx = dest_node.parent?;

        if parent_idx != src_idx {
            return None;
        }

        for (label, child) in enumerate(&src_node.children) {
            if let Some(child_idx) = child {
                if child_idx == &dest_idx {
                    return Some(label);
                }
            }
        }

        None
    }

    #[inline(always)]
    pub fn get_root_idx(&self) -> TreeIndex {
        self.root.unwrap()
    }

    pub fn get_root(&self) -> &TreeNode<N, K> {
        let root = &self.arena[self.get_root_idx()];
        debug_assert!(
            root.parent.is_none(),
            "Tree invariant violated: root node has a parent"
        );
        root
    }

    /// Sets the given node as root of this tree, overwriting existing values.
    ///
    /// If a root node already exists then the this operation disconnects the former tree, making it unreachable from the root.
    pub fn add_root(&mut self, value: N) -> TreeIndex {
        let idx = self.arena.insert(TreeNode::new(value, None));
        self.root = Some(idx);
        idx
    }

    pub fn add_child_node(&mut self, parent: TreeIndex, label: Label, value: N) -> TreeIndex {
        assert!(
            self.arena.contains(parent),
            "Cannot add child: parent node not contained in graph"
        );

        let childnode = TreeNode::new(value, Some(parent));
        let node_idx = self.arena.insert(childnode);

        let parent_node = self.tree_node_mut(parent).unwrap();
        parent_node.isleaf = false;
        assert!(
            parent_node.children[label].is_none(),
            "Overwriting existing node"
        );
        parent_node.children[label] = Some(node_idx);

        node_idx
    }

    /// Removes the child from parent that is reachable by label.
    /// Any descendant that is only reachable from the child is also removed.
    pub fn remove_child(&mut self, parent: TreeIndex, label: Label) {
        let child_idx = self.tree_node(parent).unwrap().children[label].unwrap();
        self.arena[parent].children[label] = None;

        if self.num_children(parent) == 0 {
            self.arena[parent].isleaf = true;
        }

        let mut stack = Vec::new();
        stack.push(child_idx);

        while let Some(node_idx) = stack.pop() {
            let current_node = self.arena.remove(node_idx);
            if !current_node.isleaf {
                for child_idx in current_node.children.into_iter().flatten() {
                    stack.push(child_idx);
                }
            }
        }
    }

    pub fn merge_child_with_parent(&mut self, node: TreeIndex, label: Label) {
        assert!(self.num_children(node) == 1);
        let child_idx = self.arena[node].children[label].unwrap();

        // Copy children to parent
        let parent_idx = self.arena[node].parent.unwrap();
        let label = self.get_label(parent_idx, node).unwrap();

        // Skip node
        self.arena[parent_idx].children[label] = Some(child_idx);
        self.arena[child_idx].parent = Some(parent_idx);

        self.arena.remove(node);
    }

    pub fn update_node(&mut self, idx: TreeIndex, value: N) -> Option<N> {
        let node = self.tree_node_mut(idx)?;
        Some(mem::replace(&mut node.value, value))
    }

    pub fn path_to_node(&self, node_idx: usize) -> Vec<(TreeIndex, Label)> {
        assert!(
            self.arena.contains(node_idx),
            "Given node is not part of the tree"
        );
        let mut current_node_idx = node_idx;

        let capacity = (self.len() as f64).log(K as f64).ceil() as usize;
        let mut path: Vec<(usize, usize)> = Vec::with_capacity(capacity);

        while let Some(parent_idx) = self.tree_node(current_node_idx).unwrap().parent {
            let edge_label = self.get_label(parent_idx, current_node_idx).unwrap();
            path.push((parent_idx, edge_label));
            current_node_idx = parent_idx;
        }

        path.reverse();
        path
    }
}

impl<N, const K: usize> Index<usize> for Tree<N, K> {
    type Output = N;

    fn index(&self, index: usize) -> &Self::Output {
        self.node_value(index)
            .expect(&format!("No node exists with index {}", index))
    }
}

impl<N, const K: usize> IndexMut<usize> for Tree<N, K> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.node_value_mut(index)
            .expect(&format!("No node exists with index {}", index))
    }
}

impl<const K: usize, N> fmt::Debug for Tree<N, K>
where
    N: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Decision Tree with {} nodes", &self.len())?;

        for (node_idx, node) in self.node_iter() {
            writeln!(
                f,
                "[{:>3}] {}",
                node_idx,
                if node.isleaf { "T" } else { "D" }
            )?;

            if self.num_children(node_idx) > 0 {
                if node.isleaf {
                    write!(f, "!!!")?;
                }
                write!(f, "children: ")?;
                for edg in self.children(node_idx) {
                    write!(f, "{}->{},", edg.label, edg.target_idx)?;
                }
                writeln!(f)?;
            }

            if node.parent.is_some() {
                writeln!(f, "parent: {:>3}", node.parent.unwrap())?;
            } else {
                writeln!(f, "parent:  - ")?;
            }

            writeln!(f, "{:?}", node.value)?;
        }
        Ok(())
    }
}

/// Follow the given label sequence beginning at the root of the specified tree.
/// The resulting index at the end of the label sequence is returned.
///
/// # Example
///
/// ```rust
/// use affinitree::{path, tree::graph::{Tree}};
/// let mut tree = Tree::<(), 2>::new();
/// let z = tree.add_root(()); // 0
/// let c0 = tree.add_child_node(z, 0, ()); // 1
/// let c1 = tree.add_child_node(z, 1, ()); // 2
/// let l0 = tree.add_child_node(c0, 0, ()); // 3
///
/// assert_eq!(path!(tree, 1), 2);
/// assert_eq!(path!(tree, 0, 0), 3);
/// ```
#[macro_export]
macro_rules! path {
    ($tree:expr , $( $label:literal ),* ) => {{
        let mut index = $tree.get_root_idx();

        $(
            index = $tree.child(index, $label).unwrap().target_idx;
        )*

        index
    }};
}

#[allow(unused_variables)]
#[cfg(test)]
mod tests {

    use assertables::*;

    use crate::tree::iter::DFSEdgeIter;

    use super::{Label, Tree, TreeIndex};

    #[test]
    pub fn test_depth() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()); // 1
        let c1 = tree.add_child_node(z, 1, ()); // 2
        let l0 = tree.add_child_node(c0, 0, ()); // 3
        let l1 = tree.add_child_node(c0, 1, ()); // 4
        let r0 = tree.add_child_node(c1, 0, ()); // 5
        let r1 = tree.add_child_node(c1, 1, ()); // 6
        let rr1 = tree.add_child_node(r1, 1, ()); // 7

        assert_eq!(tree.depth(), 3);
    }

    #[test]
    pub fn test_node_iter_skips_deleted_nodes() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()); // 1
        let c1 = tree.add_child_node(z, 1, ()); // 2
        let l0 = tree.add_child_node(c0, 0, ()); // 3
        let l1 = tree.add_child_node(c0, 1, ()); // 4
        let r0 = tree.add_child_node(c1, 0, ()); // 5
        let r1 = tree.add_child_node(c1, 1, ()); // 6
        let rr1 = tree.add_child_node(r1, 1, ()); // 7

        tree.remove_child(0, 0);

        let indices: Vec<usize> = tree.node_iter().map(|(idx, node)| idx).collect();
        assert_eq!(
            &indices,
            &vec![0, 2, 5, 6, 7],
            "Iterators should not yield deleted nodes"
        );
    }

    #[test]
    pub fn test_terminal_iter() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()); // 1
        let c1 = tree.add_child_node(z, 1, ()); // 2
        let l0 = tree.add_child_node(c0, 0, ()); // 3
        let l1 = tree.add_child_node(c0, 1, ()); // 4
        let r0 = tree.add_child_node(c1, 0, ()); // 5
        let r1 = tree.add_child_node(c1, 1, ()); // 6
        let rr1 = tree.add_child_node(r1, 1, ()); // 7

        let indices: Vec<usize> = tree.terminal_indices().collect();
        assert_eq!(&indices, &vec![3, 4, 5, 7]);
    }

    #[test]
    pub fn test_dfs_edge_iter() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()); // 1
        let c1 = tree.add_child_node(z, 1, ()); // 2
        let l0 = tree.add_child_node(c0, 0, ()); // 3
        let l1 = tree.add_child_node(c0, 1, ()); // 4
        let r0 = tree.add_child_node(c1, 0, ()); // 5
        let r1 = tree.add_child_node(c1, 1, ()); // 6
        let rr1 = tree.add_child_node(r1, 1, ()); // 7

        let iter = DFSEdgeIter::new(&tree);
        let nodes = Vec::from_iter(iter.map(|(_, ed)| (ed.source_idx, ed.label, ed.target_idx)));

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

        let mut iter = DFSEdgeIter::new(&tree);
        iter.next();
        // skip descendants of c0, i.e., l0 and l1
        iter.skip_subtree();

        assert_eq!(iter.next().unwrap().1.target_idx, c1);
    }

    #[test]
    pub fn test_edge_iter() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()); // 1
        let c1 = tree.add_child_node(z, 1, ()); // 2
        let l0 = tree.add_child_node(c0, 0, ()); // 3
        let l1 = tree.add_child_node(c0, 1, ()); // 4
        let r0 = tree.add_child_node(c1, 0, ()); // 5
        let r1 = tree.add_child_node(c1, 1, ()); // 6
        let rr1 = tree.add_child_node(r1, 1, ()); // 7

        let indices: Vec<(TreeIndex, Label, TreeIndex)> = tree
            .edge_iter()
            .map(|edg| (edg.source_idx, edg.label, edg.target_idx))
            .collect();
        assert_set_eq!(
            &indices,
            &vec![
                (0, 0, c0),
                (0, 1, c1),
                (c0, 0, l0),
                (c0, 1, l1),
                (c1, 0, r0),
                (c1, 1, r1),
                (r1, 1, rr1)
            ]
        );
    }

    #[test]
    pub fn test_parent() {
        let mut tree = Tree::<i32, 2>::new();

        let z = tree.add_root(10); // 0
        let c0 = tree.add_child_node(z, 0, 11); // 1
        let c1 = tree.add_child_node(z, 1, 12); // 2
        let l0 = tree.add_child_node(c0, 0, 13); // 3

        let edg = tree.parent_mut(3).unwrap();

        assert_eq!(edg.source_idx, 1);
        assert_eq!(edg.target_idx, 3);
        assert_eq!(edg.label, 0);

        *edg.source_value = 7;

        assert_eq!(tree.tree_node(1).unwrap().value, 7);
    }

    #[test]
    pub fn test_path() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()); // 1
        let c1 = tree.add_child_node(z, 1, ()); // 2
        let l0 = tree.add_child_node(c0, 0, ()); // 3
        let l1 = tree.add_child_node(c0, 1, ()); // 4
        let r0 = tree.add_child_node(c1, 0, ()); // 5
        let r1 = tree.add_child_node(c1, 1, ()); // 6
        let rr1 = tree.add_child_node(r1, 1, ()); // 7

        assert_eq!(path!(tree, 0, 1), 4);
        assert_eq!(path!(tree, 1, 1, 1), 7);
    }
}
