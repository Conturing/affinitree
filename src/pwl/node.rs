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

use core::fmt;
use std::fmt::{Display, Write};

use itertools::Itertools;
use itertools::Position::{First, Last, Middle, Only};
use ndarray::Array1;

use crate::linalg::affine::{AffFunc, Polytope};
use crate::linalg::impl_affineformat::{FormatOptions, write_func, write_poly};
use crate::tree::graph::TreeNode;

/// An enum of possible feasibility states a node can attain.
///
/// The feasibility state of a node depends exclusively on the path that leads to it,
/// and can therefore only be seen with respect to the [``crate::pwl::afftree::AffTree``]
/// that contains the node.
#[derive(Clone, Debug)]
pub enum NodeState {
    /// Feasibility is unknown
    Indeterminate,
    /// Node is feasible
    Infeasible,
    /// Node is feasible and a witness is no longer required or not known
    Feasible,
    /// Node is feasible and at least one witness is known
    FeasibleWitness(Vec<Array1<f64>>),
}

impl NodeState {
    /// Returns true iff this node is feasible
    #[inline(always)]
    pub fn is_feasible(&self) -> bool {
        matches!(self, NodeState::Feasible | NodeState::FeasibleWitness(_))
    }

    /// Returns true iff this node is infeasible
    #[inline(always)]
    pub fn is_infeasible(&self) -> bool {
        matches!(self, NodeState::Infeasible)
    }

    /// Returns true iff the feasibility state of this node is unknown
    #[inline(always)]
    pub fn is_indetermined(&self) -> bool {
        matches!(self, NodeState::Indeterminate)
    }
}

/// A node type to store either a linear decision or linear function for an [``crate::pwl::afftree::AffTree``].
///
/// This type also tracks the feasibility of this node in form of a [``NodeState``].
#[derive(Clone, Debug)]
pub struct AffContent {
    /// An affine function representing either the decision predicate A @ x <= b,
    /// or the terminal function A @ x + b.
    pub aff: AffFunc,
    /// Feasibility state of this node based on the unique path from the root to this node
    pub state: NodeState,
}

impl AffContent {
    pub fn new(aff: AffFunc) -> AffContent {
        AffContent {
            aff,
            state: NodeState::Indeterminate,
        }
    }

    pub fn to_poly(&self) -> Polytope {
        Polytope::from_mats(self.aff.mat.clone(), self.aff.bias.clone())
    }

    pub fn feasible_witnesses(&self) -> Vec<Array1<f64>> {
        match &self.state {
            NodeState::FeasibleWitness(witnesses) => witnesses.clone(),
            _ => Vec::new(),
        }
    }
}

pub type AffNode<const K: usize> = TreeNode<AffContent, K>;

impl<const K: usize> Display for AffNode<K> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.isleaf {
            write_terminal(f, &self.value.aff, true)
        } else {
            write_predicate(f, &self.value.aff, true)
        }
    }
}

/// Write a representation of ``pred`` interpreted as a decision node into ``f``.
///
/// Here ``pred`` is interpreted as A @ x <= b
pub fn write_predicate(
    f: &mut fmt::Formatter,
    pred: &AffFunc,
    _pretty_print: bool,
) -> std::fmt::Result {
    let opt = FormatOptions::default_poly();
    write_poly(
        f,
        Polytope::from_mats(pred.mat.clone(), pred.bias.clone()).view(),
        &opt,
    )
}

/// Write a representation of ``pred`` interpreted as a terminal node into ``f``.
pub fn write_terminal(
    f: &mut fmt::Formatter,
    pred: &AffFunc,
    _pretty_print: bool,
) -> std::fmt::Result {
    let opt = FormatOptions::default_func();
    write_func(f, pred.view(), &opt)
}

/// Write a representation of the children array of ``node`` into ``f``.
pub fn write_children<T, F: Write, const K: usize>(
    f: &mut F,
    node: &TreeNode<T, K>,
) -> std::fmt::Result {
    for (pos, (label, idx)) in node.children_iter().with_position() {
        match pos {
            First | Middle => write!(f, "{}->{}, ", label, idx)?,
            Only | Last => write!(f, "{}->{}", label, idx)?,
        }
    }
    Ok(())
}
