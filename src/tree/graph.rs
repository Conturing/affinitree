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

//! Generic tree with constant branching factor

use core::fmt;
use std::mem;
use std::ops::{Index, IndexMut};

use itertools::{Itertools, enumerate};
use slab::Slab;
use thiserror::Error;

use super::iter::{DfsEdge, DfsPre, TraversalIter, TraversalMut};

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
            value,
            parent,
            children: [None; K],
            isleaf: true,
        }
    }

    #[inline]
    #[rustfmt::skip]
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

impl<N> NodeReference<'_, N> {
    pub fn index(&self) -> TreeIndex {
        self.idx
    }
}

#[derive(Debug, PartialEq)]
pub struct NodeReferenceMut<'a, N> {
    pub value: &'a mut N,
    pub idx: TreeIndex,
}

impl<N> NodeReferenceMut<'_, N> {
    pub fn index(&self) -> TreeIndex {
        self.idx
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub source_idx: TreeIndex,
    pub label: Label,
    pub target_idx: TreeIndex,
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

    pub fn edge(&self) -> Edge {
        Edge {
            source_idx: self.source_idx,
            label: self.label,
            target_idx: self.target_idx,
        }
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

    pub fn edge(&self) -> Edge {
        Edge {
            source_idx: self.source_idx,
            label: self.label,
            target_idx: self.target_idx,
        }
    }
}

#[derive(Error, Debug)]
#[error("invalid index received: node {index:?} is not part of the graph")]
pub struct InvalidTreeIndexError {
    index: TreeIndex,
}

#[derive(Error, Debug)]
pub enum NodeError {
    #[error(transparent)]
    InvalidIndex(#[from] InvalidTreeIndexError),
    #[error("expected a child to exist for node {parent:?} with label {label:?}")]
    MissingChild { parent: TreeIndex, label: Label },
    #[error("expected node {index:?} to have a parent")]
    MissingParent { index: TreeIndex },
    #[error("cannot complete operation without overwriting existing node {index:?}")]
    NodeExists { index: TreeIndex },
    #[error(
        "cannot complete operation without overwriting existing child of {parent:?} with label {label:?}"
    )]
    ChildExists { parent: TreeIndex, label: Label },
    #[error("the requested operation is not permissable for the root node")]
    RootNode,
    #[error("no node found with the desired property")]
    NodeNotFound,
}

// pub type Result<T> = core::result::Result<T, NodeError>;

/// A tree implementation with constant branching factor ``K`` over an arena.
///
/// This tree implementation is designed to be memory efficient and cache friendly.
/// It is based on an arena provided by the [``slab``] crate.
/// In this tree, each node has a unique index during its lifetime inside the tree.
/// Based in these indices, each node stores its parent and its ``K`` children.
/// Nodes are stored in insertion-order, which can be optimized by the caller for cache friendly traversal patterns.
///
/// This tree supports standard tree operations like adding and removing nodes, and depth-first, breath-first, and in-memory traversal.
/// This implementation handles a single connected tree, if at any point nodes become unreachable,they are automatically removed.
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
    #[inline]
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

    /// Constructs a new, single-node `Tree<N, K>` with the specified ``capacity`` and ``root`` node.
    #[inline]
    pub fn with_root(root: N, capacity: usize) -> Tree<N, K> {
        let mut tree = Self::with_capacity(capacity.max(1));
        tree.add_root(root);
        tree
    }

    /// Returns the number of nodes in this tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.arena.len()
    }

    /// Returns true if there are no values in this tree.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    /// Returns the number of terminal nodes in this tree
    #[inline]
    pub fn num_terminals(&self) -> usize {
        self.terminal_indices().count()
    }

    /// Returns the number of descendants of ``node`` (including ``node`` itself)
    #[inline]
    pub fn num_nodes(&self, node: TreeIndex) -> usize {
        DfsPre::iter(self, node).count()
    }

    /// Returns the depth of this tree, that is, the length of its longest path.
    ///
    /// For example, an empty tree has depth 0, and a tree with only a root node has depth 1.
    #[rustfmt::skip]
    pub fn depth(&self) -> usize {
        self.dfs_iter()
            .map(|data| data.depth)
            .max()
            .unwrap_or(0)
    }

    /// Returns statistics over the depth of this tree.
    ///
    /// Computes over the set of all terminal nodes the minimum, mean,
    /// variance, and maximum of their depth.
    #[rustfmt::skip]
    pub fn depth_stats(&self) -> (f64, f64, f64, f64) {
        use average::{Min, Max, Estimate, Variance};

        let mut min = Min::new();
        let mut max = Max::new();
        let mut var = Variance::new();

        for data in self.dfs_iter() {
            if !self.is_leaf(data.index).unwrap_or(false) {
                continue;
            }
            min.add(data.depth as f64);
            max.add(data.depth as f64);
            var.add(data.depth as f64);
        }

        (min.min(), var.mean(), var.sample_variance(), max.max())
    }

    /// Returns the number of children the given ``node`` has.
    #[rustfmt::skip]
    #[inline]
    pub fn num_children(&self, node: TreeIndex) -> usize {
        self.arena[node]
            .children
            .iter()
            .filter(|&&x| x.is_some())
            .count()
    }

    /// Returns the number of nodes that can be added without reallocation.
    ///
    /// See also [`Slab::capacity`].
    #[inline]
    pub fn capacity(&self) -> usize {
        self.arena.capacity()
    }

    /// Reserves capacity for at least ``additional`` more nodes to be stored.
    ///
    /// See also [`Slab::reserve`] and [`Vec::reserve`].
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.arena.reserve(additional);
    }

    /// Returns the value associated with the node at the given index.
    #[inline]
    #[rustfmt::skip]
    pub fn node_value(&self, idx: TreeIndex) -> Result<&N, InvalidTreeIndexError> {
        self.arena
            .get(idx)
            .map(|nd| &nd.value)
            .ok_or(InvalidTreeIndexError { index: idx })
    }

    /// Returns the value associated with the node at the given index.
    #[inline]
    #[rustfmt::skip]
    pub fn node_value_mut(&mut self, idx: TreeIndex) -> Result<&mut N, InvalidTreeIndexError> {
        self.arena
            .get_mut(idx)
            .map(|nd| &mut nd.value)
            .ok_or(InvalidTreeIndexError { index: idx })
    }

    /// Tests whether the node at the given index has children or not.
    /// If the node does not exist, `Err` is returned.
    #[rustfmt::skip]
    pub fn is_leaf(&self, idx: TreeIndex) -> Result<bool, InvalidTreeIndexError> {
        self.tree_node(idx)
            .map(|nd| nd.isleaf)
    }

    /// Tests whether the node at the given index is the root of this tree.
    /// If the node does not exist, `None` is returned.
    #[rustfmt::skip]
    pub fn is_root(&self, idx: TreeIndex) -> bool {
        if let Some(root_idx) = self.root {
            root_idx == idx
        } else {
            false
        }
    }

    /// Returns the internal structure used to store the node at the given index.
    ///
    /// **Use with caution**: Provides access to internal representation.
    #[inline(always)]
    pub fn tree_node(&self, idx: TreeIndex) -> Result<&TreeNode<N, K>, InvalidTreeIndexError> {
        self.arena
            .get(idx)
            .ok_or(InvalidTreeIndexError { index: idx })
    }

    /// Returns the internal structure used to store the node at the given index.
    ///
    /// **Use with caution**: Provides mutable access to internal representation.
    #[inline(always)]
    pub fn tree_node_mut(
        &mut self,
        idx: TreeIndex,
    ) -> Result<&mut TreeNode<N, K>, InvalidTreeIndexError> {
        self.arena
            .get_mut(idx)
            .ok_or(InvalidTreeIndexError { index: idx })
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
    ) -> Result<(&mut TreeNode<N, K>, &mut TreeNode<N, K>), InvalidTreeIndexError> {
        if !self.arena.contains(idx1) {
            Err(InvalidTreeIndexError { index: idx1 })
        } else if !self.arena.contains(idx2) {
            Err(InvalidTreeIndexError { index: idx2 })
        } else {
            Ok(self.arena.get2_mut(idx1, idx2).unwrap())
        }
    }

    /// Returns the edge from the given node to its parent, if it exists. Otherwise `None` is returned.
    ///
    /// To find the corresponding label a linear search is performed in O(K).
    #[inline(always)]
    pub fn parent(&self, node_idx: TreeIndex) -> Result<EdgeReference<'_, N>, NodeError> {
        let node = self.tree_node(node_idx)?;
        let parent_idx = node
            .parent
            .ok_or(NodeError::MissingParent { index: node_idx })?;
        let parent = self.tree_node(parent_idx)?;

        for (label, child) in enumerate(&parent.children) {
            if let Some(child_idx) = child {
                if *child_idx == node_idx {
                    return Ok(EdgeReference {
                        source_value: &parent.value,
                        source_idx: parent_idx,
                        label,
                        target_value: &node.value,
                        target_idx: node_idx,
                    });
                }
            }
        }

        panic!("data structure corrupted: the parent of a node should contain this node as a child")
    }

    /// Returns the edge from the given node to its parent, if it exists. Otherwise `None` is returned.
    #[inline(always)]
    pub fn parent_mut(
        &mut self,
        node_idx: TreeIndex,
    ) -> Result<EdgeReferenceMut<'_, N>, NodeError> {
        let parent_idx = self
            .tree_node(node_idx)?
            .parent
            .ok_or(NodeError::MissingParent { index: node_idx })?;

        let (node, parent) = self.tree_node2_mut(node_idx, parent_idx)?;

        for (label, child) in enumerate(&parent.children) {
            if let Some(child_idx) = child {
                if child_idx == &node_idx {
                    return Ok(EdgeReferenceMut {
                        source_value: &mut parent.value,
                        source_idx: parent_idx,
                        label,
                        target_value: &mut node.value,
                        target_idx: node_idx,
                    });
                }
            }
        }

        panic!("data structure corrupted: the parent of a node should contain this node as a child")
    }

    /// Returns the edge from the given node to the child reached with the given label, if it exists. Otherwise `None` is returned.
    #[inline(always)]
    pub fn child(
        &self,
        node_idx: TreeIndex,
        label: Label,
    ) -> Result<EdgeReference<'_, N>, NodeError> {
        let node = self.tree_node(node_idx)?;
        let child_idx = node.children[label].ok_or(NodeError::MissingChild {
            parent: node_idx,
            label,
        })?;
        let child = self.tree_node(child_idx)?;

        Ok(EdgeReference {
            source_value: &node.value,
            source_idx: node_idx,
            label,
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
    ) -> Result<EdgeReferenceMut<'_, N>, NodeError> {
        let child_idx =
            self.tree_node(node_idx)?.children[label].ok_or(NodeError::MissingChild {
                parent: node_idx,
                label,
            })?;

        let (node, child) = self.tree_node2_mut(node_idx, child_idx)?;

        Ok(EdgeReferenceMut {
            source_value: &mut node.value,
            source_idx: node_idx,
            label,
            target_value: &mut child.value,
            target_idx: child_idx,
        })
    }

    // Iterators

    /// Iterates over the existing children of the given node. For each child, the edge connecting it to its parent is returned.
    ///
    /// # Panics
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
                label,
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
    #[rustfmt::skip]
    pub fn nodes(&self) -> impl DoubleEndedIterator<Item = NodeReference<'_, N>> {
        self.arena
            .iter()
            .map(|(idx, nd)| NodeReference {
                idx,
                value: &nd.value,
            })
    }

    /// Returns an iterator over the terminal nodes of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    #[rustfmt::skip]
    pub fn terminals(&self) -> impl DoubleEndedIterator<Item = NodeReference<'_, N>> {
        self.arena
            .iter()
            .filter(|(_, nd)| nd.isleaf)
            .map(|(idx, nd)| NodeReference {
                idx,
                value: &nd.value,
            })
    }

    /// Returns an iterator over the terminal nodes of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    #[rustfmt::skip]
    pub fn terminals_mut(&mut self) -> impl DoubleEndedIterator<Item = NodeReferenceMut<'_, N>> {
        self.arena
            .iter_mut()
            .filter(|(_, nd)| nd.isleaf)
            .map(|(idx, nd)| NodeReferenceMut {
                idx,
                value: &mut nd.value,
            })
    }

    /// Returns an iterator over the decision nodes of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    #[rustfmt::skip]
    pub fn decisions(&self) -> impl DoubleEndedIterator<Item = NodeReference<'_, N>> {
        self.arena
            .iter()
            .filter(|(_, nd)| !nd.isleaf)
            .map(|(idx, nd)| NodeReference {
                idx,
                value: &nd.value,
            })
    }

    /// Returns an iterator over the node indices of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    #[rustfmt::skip]
    pub fn node_indices(&self) -> impl DoubleEndedIterator<Item = TreeIndex> + '_ {
        self.arena
            .iter()
            .map(|(idx, _)| idx)
    }

    /// Returns an iterator over the indices of the terminals of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    #[rustfmt::skip]
    pub fn terminal_indices(&self) -> impl DoubleEndedIterator<Item = TreeIndex> + '_ {
        self.arena
            .iter()
            .filter(|(_, nd)| nd.isleaf)
            .map(|(idx, _)| idx)
    }

    /// Returns an iterator over the indices of the terminals of this tree.
    /// Nodes are read in index order (continuous in memory) for increased efficiency.
    #[inline]
    #[rustfmt::skip]
    pub fn decision_indices(&self) -> impl DoubleEndedIterator<Item = TreeIndex> + '_ {
        self.arena
            .iter()
            .filter(|(_, nd)| !nd.isleaf)
            .map(|(idx, _)| idx)
    }

    /// Returns an iterator over the nodes of this tree.
    /// Nodes are ordered using depth-first-search.
    #[inline]
    pub fn dfs_iter(&self) -> TraversalIter<'_, DfsPre, N, K> {
        DfsPre::iter(self, self.get_root_idx())
    }

    /// Returns an iterator over the edges of this tree.
    /// Edges are ordered using depth-first-search.
    #[inline]
    pub fn dfs_edge_iter(&self) -> TraversalIter<'_, DfsEdge, N, K> {
        DfsEdge::iter(self, self.get_root_idx())
    }

    /// Returns an iterator over the edges of this tree.
    /// No guarantees are made over the ordering of the edges.
    #[rustfmt::skip]
    pub fn edge_iter(&self) -> impl DoubleEndedIterator<Item = EdgeReference<'_, N>> {
        self.node_iter()
            .filter_map(|(idx, _)| self.parent(idx).ok())
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
    /// If a root node already exists, then this operation disconnects the former tree, making it unreachable from the root.
    pub fn add_root(&mut self, value: N) -> TreeIndex {
        let idx = self.arena.insert(TreeNode::new(value, None));
        self.root = Some(idx);
        idx
    }

    /// Adds a new node to the graph with given ``value``.
    /// The node is placed as a child of ``parent`` reachable by ``label``.
    ///
    /// # Panics
    ///
    /// Panics when ``parent`` already has a child with ``label``.
    pub fn add_child_node(
        &mut self,
        parent: TreeIndex,
        label: Label,
        value: N,
    ) -> Result<TreeIndex, NodeError> {
        if !self.arena.contains(parent) {
            return Err(NodeError::InvalidIndex(InvalidTreeIndexError {
                index: parent,
            }));
        }

        let childnode = TreeNode::new(value, Some(parent));
        let node_idx = self.arena.insert(childnode);

        let parent_node = self.tree_node_mut(parent).expect(
            "invalid state: tree node should be available when index is contained in arena",
        );
        parent_node.isleaf = false;

        if parent_node.children[label].is_some() {
            Err(NodeError::ChildExists { parent, label })
        } else {
            parent_node.children[label] = Some(node_idx);
            Ok(node_idx)
        }
    }

    /// Tries to remove the node uniquely specified as the child of ``parent``
    /// reachable by ``label``.
    /// When the node exists, it is removed and its value is returned.
    /// Otherwise, ``None`` is returned.
    ///
    /// Any descendants that are only reachable from the node are also removed.
    pub fn try_remove_child(&mut self, parent: TreeIndex, label: Label) -> Result<N, NodeError> {
        let child_idx = self.child(parent, label)?.target_idx;
        self.remove_all_descendants(child_idx)?;

        self.arena[parent].children[label] = None;

        if self.num_children(parent) == 0 {
            self.arena[parent].isleaf = true;
        }

        Ok(self
            .arena
            .try_remove(child_idx)
            .expect("data structure corrupted: the index of a child should be valid")
            .value)
    }

    /// Removes the child from ``parent`` that is reachable by ``label``.
    /// Any descendant that is only reachable from the child is also removed.
    ///
    /// # Panics
    ///
    /// Panics if ``parent`` has no child reachable by ``label``.
    pub fn remove_child(&mut self, parent: TreeIndex, label: Label) -> N {
        self.try_remove_child(parent, label).expect("invalid index")
    }

    /// Removes all descendants of ``subtree_root`` in this tree.
    /// Returns ``Ok`` on success with the number of deleted nodes or
    /// ``Err`` if ``subtree_root`` is not contained in this tree.
    pub fn remove_all_descendants(
        &mut self,
        subtree_root: TreeIndex,
    ) -> Result<i32, InvalidTreeIndexError> {
        if !self.arena.contains(subtree_root) {
            return Err(InvalidTreeIndexError {
                index: subtree_root,
            });
        }

        let mut stack = self
            .children(subtree_root)
            .map(|edg| edg.target_idx)
            .collect_vec();

        let mut num_deleted = 0;
        while let Some(node_idx) = stack.pop() {
            let current_node = self
                .tree_node(node_idx)
                .expect("data structure corrupted: index stored in the tree should be valid");
            for child_idx in current_node.children.into_iter().flatten() {
                stack.push(child_idx);
            }
            self.arena.remove(node_idx);
            num_deleted += 1;
        }

        let node = self
            .tree_node_mut(subtree_root)
            .expect("invalid state: subtree root should still be contained in tree");
        node.isleaf = true;
        for child in &mut node.children {
            *child = None;
        }

        Ok(num_deleted)
    }

    /// Skips ``parent_idx`` by creating a direct edge from the grandparent of ``parent_idx`` to its child reachable by ``label``.
    /// Afterwards ``parent_idx`` is removed from the graph.
    ///
    /// This is a useful optimization when ``parent_idx`` is redundant.
    ///
    /// Returns ``Err`` when ``parent_idx`` is not part of this tree or when it has no child reachable by ``label``.
    pub fn merge_child_with_parent(
        &mut self,
        parent_idx: TreeIndex,
        label: Label,
    ) -> Result<TreeNode<N, K>, NodeError> {
        assert!(self.num_children(parent_idx) == 1);
        if let Some(root_idx) = self.root {
            if root_idx == parent_idx {
                return Err(NodeError::RootNode);
            }
        }

        let child_idx = self.child(parent_idx, label)?.target_idx;
        let edge = self.parent(parent_idx)?.edge();
        let grandparent_idx = edge.source_idx;
        let grandparent_label = edge.label;

        // Skip node
        self.arena[grandparent_idx].children[grandparent_label] = Some(child_idx);
        self.arena[child_idx].parent = Some(grandparent_idx);

        Ok(self.arena.remove(parent_idx))
    }

    /// Moves the given ``value`` into the node at ``idx``, returning its previous value.
    pub fn update_node(&mut self, idx: TreeIndex, value: N) -> Result<N, NodeError> {
        let node = self.tree_node_mut(idx)?;
        Ok(mem::replace(&mut node.value, value))
    }

    /// Returns true if ``node_idx`` is a valid index for this tree.
    ///
    /// Note: An index may be reused after its node was deleted from the graph, thus ``node_idx`` might refer to a different node.
    pub fn contains(&self, node_idx: TreeIndex) -> bool {
        self.arena.contains(node_idx)
    }

    /// Returns the unique sequence of nodes and labels that are encountered along the
    /// path from the root to ``node_idx``.
    pub fn path_to_node(&self, node_idx: usize) -> Result<Vec<(TreeIndex, Label)>, NodeError> {
        if !self.arena.contains(node_idx) {
            return Err(NodeError::InvalidIndex(InvalidTreeIndexError {
                index: node_idx,
            }));
        }
        let mut current_node_idx = node_idx;

        let capacity = (self.len() as f64).log(K as f64).ceil() as usize;
        let mut path: Vec<(usize, usize)> = Vec::with_capacity(capacity);

        loop {
            let edge = match self.parent(current_node_idx) {
                Ok(edge) => edge,
                Err(NodeError::MissingParent { .. }) => break,
                Err(other) => return Err(other),
            };

            path.push((edge.source_idx, edge.label));
            current_node_idx = edge.source_idx;
        }

        path.reverse();
        Ok(path)
    }

    /// Creates a short summary of the tree in form of a string.
    pub fn describe(&self) -> String {
        let (min, mean, var, max) = self.depth_stats();
        format!(
            "Tree {{ nodes={}, terminals={}, depth=[min={}, mean={:.2}, var={:.2}, max={}] }}",
            self.len(),
            self.num_terminals(),
            min,
            mean,
            var,
            max
        )
    }
}

impl<N, const K: usize> Index<usize> for Tree<N, K> {
    type Output = N;

    fn index(&self, index: usize) -> &Self::Output {
        self.node_value(index)
            .unwrap_or_else(|_| panic!("No node exists with index {}", index))
    }
}

impl<N, const K: usize> IndexMut<usize> for Tree<N, K> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.node_value_mut(index)
            .unwrap_or_else(|_| panic!("No node exists with index {}", index))
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
/// let c0 = tree.add_child_node(z, 0, ()).unwrap(); // 1
/// let c1 = tree.add_child_node(z, 1, ()).unwrap(); // 2
/// let l0 = tree.add_child_node(c0, 0, ()).unwrap(); // 3
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

    use super::*;

    #[test]
    pub fn test_depth() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, ()).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, ()).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, ()).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, ()).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, ()).unwrap(); // 6
        let rr1 = tree.add_child_node(r1, 1, ()).unwrap(); // 7

        assert_eq!(tree.depth(), 3);
    }

    #[test]
    pub fn test_node_iter_skips_deleted_nodes() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, ()).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, ()).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, ()).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, ()).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, ()).unwrap(); // 6
        let rr1 = tree.add_child_node(r1, 1, ()).unwrap(); // 7

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
        let c0 = tree.add_child_node(z, 0, ()).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, ()).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, ()).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, ()).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, ()).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, ()).unwrap(); // 6
        let rr1 = tree.add_child_node(r1, 1, ()).unwrap(); // 7

        let indices: Vec<usize> = tree.terminal_indices().collect();
        assert_eq!(&indices, &vec![3, 4, 5, 7]);
    }

    #[test]
    pub fn test_edge_iter() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, ()).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, ()).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, ()).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, ()).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, ()).unwrap(); // 6
        let rr1 = tree.add_child_node(r1, 1, ()).unwrap(); // 7

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
        let c0 = tree.add_child_node(z, 0, 11).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, 12).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, 13).unwrap(); // 3

        let edg = tree.parent_mut(3).unwrap();

        assert_eq!(edg.source_idx, 1);
        assert_eq!(edg.target_idx, 3);
        assert_eq!(edg.label, 0);

        *edg.source_value = 7;

        assert_eq!(tree.tree_node(1).unwrap().value, 7);
    }

    #[test]
    fn test_remove_node() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, ()).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, ()).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, ()).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, ()).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, ()).unwrap(); // 6
        let rr1 = tree.add_child_node(r1, 1, ()).unwrap(); // 7

        tree.remove_child(z, 1);
        assert!(!tree.contains(c1));
        assert!(!tree.contains(r0));
        assert!(!tree.contains(r1));
        assert!(!tree.contains(rr1));
        assert_eq!(tree.len(), 4);
    }

    #[test]
    fn test_merge() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, ()).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, ()).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, ()).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, ()).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, ()).unwrap(); // 6
        let rr1 = tree.add_child_node(r1, 0, ()).unwrap(); // 7

        tree.merge_child_with_parent(r1, 0).unwrap();

        assert!(!tree.contains(r1));
        assert_eq!(tree.parent(rr1).unwrap().source_idx, c1);
        assert_eq!(tree.parent(rr1).unwrap().label, 1);
        assert_eq!(tree.child(c1, 1).unwrap().target_idx, rr1);
        assert_eq!(tree.child(c1, 1).unwrap().label, 1);
        assert_eq!(tree.len(), 7);
    }

    #[test]
    pub fn test_path() {
        let mut tree = Tree::<(), 2>::new();

        let z = tree.add_root(()); // 0
        let c0 = tree.add_child_node(z, 0, ()).unwrap(); // 1
        let c1 = tree.add_child_node(z, 1, ()).unwrap(); // 2
        let l0 = tree.add_child_node(c0, 0, ()).unwrap(); // 3
        let l1 = tree.add_child_node(c0, 1, ()).unwrap(); // 4
        let r0 = tree.add_child_node(c1, 0, ()).unwrap(); // 5
        let r1 = tree.add_child_node(c1, 1, ()).unwrap(); // 6
        let rr1 = tree.add_child_node(r1, 1, ()).unwrap(); // 7

        assert_eq!(path!(tree, 0, 1), 4);
        assert_eq!(path!(tree, 1, 1, 1), 7);
    }
}
