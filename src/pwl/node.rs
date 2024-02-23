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

use crate::linalg::{
    affine::{AffFunc, Polytope},
    display::{write_aff, write_polytope},
};

use crate::tree::graph::TreeNode;

use core::fmt;

use std::fmt::Display;

use ndarray::Array1;

use itertools::{
    Itertools,
    Position::{First, Last, Middle, Only},
};
use std::fmt::Write;

#[derive(Clone, Debug)]
pub enum NodeState {
    Indeterminate,
    Infeasible,
    Feasible,
    FeasibleWitness(Vec<Array1<f64>>),
}

impl NodeState {
    #[inline(always)]
    pub fn is_feasible(&self) -> bool {
        matches!(self, NodeState::Feasible | NodeState::FeasibleWitness(_))
    }

    #[inline(always)]
    pub fn is_infeasible(&self) -> bool {
        matches!(self, NodeState::Infeasible)
    }

    #[inline(always)]
    pub fn is_indetermined(&self) -> bool {
        matches!(self, NodeState::Indeterminate)
    }
}

#[derive(Clone, Debug)]
pub struct AffContent {
    // When interpreted as decision A @ x + b >= 0
    // Note: inconsistency with polytope class (which uses A @ x <= b)
    pub aff: AffFunc,
    // State of *this* node based on the path from the root to this node
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
        Polytope::from_mats(-self.aff.mat.clone(), self.aff.bias.clone())
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

/// Write a string representation of pred into f.
/// Here pred is interpreted as A @ x + b >= 0
pub fn write_predicate<T: Write>(
    f: &mut T,
    pred: &AffFunc,
    pretty_print: bool,
) -> std::fmt::Result {
    write_polytope(
        f,
        &Polytope::from_mats(-pred.mat.clone(), pred.bias.clone()),
        pretty_print,
    )
}

pub fn write_terminal<T: Write>(f: &mut T, pred: &AffFunc, pretty_print: bool) -> std::fmt::Result {
    write_aff(f, pred, pretty_print)
}

pub fn write_children<T, F: Write, const K: usize>(
    f: &mut F,
    node: &TreeNode<T, K>,
) -> std::fmt::Result {
    for child in node.children_iter().with_position() {
        match child {
            First((label, idx)) | Middle((label, idx)) => write!(f, "{}->{}, ", label, idx)?,
            Only((label, idx)) | Last((label, idx)) => write!(f, "{}->{}", label, idx)?,
        }
    }
    Ok(())
}
