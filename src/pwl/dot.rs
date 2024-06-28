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

//! An interface to graphviz's DOT language to export and visualize AffTrees

use std::fmt;

use crate::linalg::affine::Polytope;
use crate::linalg::impl_affineformat::{write_func, write_poly, FormatOptions};
use crate::pwl::afftree::AffTree;

/// A formatter to convert [``AffTree``] instances to graphviz's DOT format.
///
/// The DOT string can be accessed via the standard methods of the [``Display``] trait
/// which are implemented for this class.
#[derive(Debug, Clone)]
pub struct Dot<'a> {
    pub tree: &'a AffTree<2>,
    pub graph_name: String,
    pub terminal_attr: String,
    pub decision_attr: String,
    pub true_edge_attr: String,
    pub false_edge_attr: String,
    pub terminal_opt: FormatOptions,
    pub decision_opt: FormatOptions,
}

impl<'a> Dot<'a> {
    /// Creates a new [``Dot``] instance with default formatting arguments for the given [``AffTree``].
    pub fn from(tree: &'a AffTree<2>) -> Dot<'a> {
        Dot {
            tree,
            graph_name: "afftree".to_owned(),
            terminal_attr: "shape=ellipse".to_owned(),
            decision_attr: "shape=box".to_owned(),
            true_edge_attr: "style=solid".to_owned(),
            false_edge_attr: "style=dashed".to_owned(),
            terminal_opt: FormatOptions::default_func(),
            decision_opt: FormatOptions::default_poly(),
        }
    }
}

impl<'a> fmt::Display for Dot<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "digraph {} {{\nbgcolor=transparent;\nconcentrate=true;\nmargin=0;\n",
            self.graph_name
        )?;

        for (idx, node) in self.tree.tree.node_iter() {
            write!(f, "n{} [label=\"", idx)?;
            if node.isleaf {
                write_func(f, node.value.aff.view(), &self.terminal_opt)?;
                writeln!(f, "\", {}];", self.decision_attr)?;
            } else {
                let poly = Polytope::from_mats(
                    -node.value.aff.mat.to_owned(),
                    node.value.aff.bias.to_owned(),
                );
                write_poly(f, poly.view(), &self.decision_opt)?;
                writeln!(f, "\", {}];", self.terminal_attr)?;
            }
        }

        for edg in self.tree.tree.edge_iter() {
            writeln!(
                f,
                "n{} -> n{} [label={}, {}];",
                edg.source_idx,
                edg.target_idx,
                edg.label,
                if edg.label == 0 {
                    &self.false_edge_attr
                } else {
                    &self.true_edge_attr
                }
            )?;
        }

        write!(f, "}}")
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::distill::schema;

    #[test]
    fn test_dot_str() {
        let dd = schema::ReLU(2);

        let exp = "digraph afftree {\n\
            bgcolor=transparent;\n\
            concentrate=true;\n\
            margin=0;\n\
            n0 [label=\"−1.00 $0 −0.00 $1 ≤ +0.00\", shape=ellipse];\n\
            n1 [label=\"−0.00 $0 −1.00 $1 ≤ +0.00\", shape=ellipse];\n\
            n2 [label=\"−0.00 $0 −1.00 $1 ≤ +0.00\", shape=ellipse];\n\
            n3 [label=\"+0.00 \n\
            +0.00 +0.00 $0 +1.00 $1\", shape=box];\n\
            n4 [label=\"+0.00 \n\
            +0.00 \", shape=box];\n\
            n5 [label=\"+0.00 +1.00 $0 +0.00 $1\n\
            +0.00 +0.00 $0 +1.00 $1\", shape=box];\n\
            n6 [label=\"+0.00 +1.00 $0 +0.00 $1\n\
            +0.00 \", shape=box];\n\
            n0 -> n1 [label=1, style=solid];\n\
            n0 -> n2 [label=0, style=dashed];\n\
            n2 -> n3 [label=1, style=solid];\n\
            n2 -> n4 [label=0, style=dashed];\n\
            n1 -> n5 [label=1, style=solid];\n\
            n1 -> n6 [label=0, style=dashed];\n}";

        let actual = format!("{}", Dot::from(&dd));
        assert_eq!(actual, exp);
    }
}
