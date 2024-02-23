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

//! A collection of AffTree iterators

use log::trace;

use crate::linalg::affine::Polytope;

use super::node::AffContent;
use crate::tree::graph::{Tree, TreeIndex};
use crate::tree::iter::{DfsNodeData, DfsPre};

#[derive(Clone, Debug)]
pub struct PolyhedraGen {
    pub(super) predicates: Vec<Polytope>,
    pub(super) iter: DfsPre,
    pub(super) last_depth: usize,
}

impl PolyhedraGen {
    #[inline]
    pub fn new<const K: usize>(tree: &Tree<AffContent, K>) -> PolyhedraGen {
        Self::with_root(tree, tree.get_root_idx())
    }

    #[inline]
    pub fn with_root<const K: usize>(tree: &Tree<AffContent, K>, root: TreeIndex) -> PolyhedraGen {
        PolyhedraGen {
            predicates: Vec::with_capacity((tree.len() as f64).log2().ceil() as usize),
            iter: DfsPre::with_root(tree, root),
            last_depth: 0,
        }
    }

    #[inline]
    pub fn skip_subtree(&mut self) {
        self.iter.skip_subtree()
    }

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

        if let Some(edg) = tree.parent(node_idx) {
            trace!(
                "Edge {} -{}-> {}",
                edg.source_idx,
                edg.label,
                edg.target_idx
            );
            let factor = match edg.label {
                1 => -1.0,
                0 => 1.0,
                _ => 0.0,
            };
            let aff = &tree.node_value(edg.source_idx)?.aff;
            let poly = Polytope::from_mats(&aff.mat * factor, &aff.bias * -factor);
            self.predicates.push(poly);
        }

        Some((data, &self.predicates))
    }

    pub fn current_polytope(&self) -> &Vec<Polytope> {
        &self.predicates
    }
}

#[derive(Debug)]
pub struct PolyhedraIter<'a, const K: usize> {
    pub iter: PolyhedraGen,
    pub tree: &'a Tree<AffContent, K>,
}

impl<'a, const K: usize> PolyhedraIter<'a, K> {
    pub fn new(tree: &Tree<AffContent, K>) -> PolyhedraIter<'_, K> {
        PolyhedraIter {
            iter: PolyhedraGen::new(tree),
            tree,
        }
    }

    pub fn skip_subtree(&mut self) {
        self.iter.skip_subtree()
    }
}

impl<'a, const K: usize> Iterator for PolyhedraIter<'a, K> {
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
