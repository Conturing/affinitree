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

//! An interface to graphviz's DOT language to export and visualize AffTrees

use crate::distill::{
    afftree::AffTree,
    node::{write_predicate, write_terminal},
};

use std::fmt::Write;

pub fn dot_str<const K: usize, T: Write>(f: &mut T, tree: &AffTree<K>) -> std::fmt::Result {
    write!(
        f,
        "digraph dd {{\nbgcolor=transparent;\nconcentrate=true;\nmargin=0;\n"
    )?;

    for (idx, node) in tree.tree.node_iter() {
        write!(f, "n{} [label=\"", idx)?;
        if !node.isleaf {
            write_predicate(f, &node.value.aff, true)?;
            writeln!(f, "\", shape=box];")?;
        } else {
            write_terminal(f, &node.value.aff, true)?;
            writeln!(f, "\", shape=ellipse];")?;
        }
    }

    for edg in tree.tree.edge_iter() {
        writeln!(
            f,
            "n{} -> n{} [label={}, style={}];",
            edg.source_idx,
            edg.target_idx,
            edg.label,
            if edg.label == 0 { "dashed" } else { "solid" }
        )?;
    }

    write!(f, "}}")
}

#[cfg(test)]
pub mod test {
    use crate::{distill::dot::dot_str, distill::schema};

    #[test]
    fn test_dot_str() {
        let dd = schema::ReLU(2);

        let exp = "digraph dd {\n\
            bgcolor=transparent;\n\
            concentrate=true;\n\
            margin=0;\n\
            n0 [label=\"−$0 1.00 −$1 0.00 <= 0.00\", shape=box];\n\
            n1 [label=\"−$0 0.00 −$1 1.00 <= 0.00\", shape=box];\n\
            n2 [label=\"−$0 0.00 −$1 1.00 <= 0.00\", shape=box];\n\
            n3 [label=\"0.00\n $0 0.00 +$1 1.00 + 0.00\", shape=ellipse];\n\
            n4 [label=\"0.00\n0.00\", shape=ellipse];\n\
            n5 [label=\" $0 1.00 +$1 0.00 + 0.00\n $0 0.00 +$1 1.00 + 0.00\", shape=ellipse];\n\
            n6 [label=\" $0 1.00 +$1 0.00 + 0.00\n0.00\", shape=ellipse];\n\
            n0 -> n1 [label=1, style=solid];\n\
            n0 -> n2 [label=0, style=dashed];\n\
            n2 -> n3 [label=1, style=solid];\n\
            n2 -> n4 [label=0, style=dashed];\n\
            n1 -> n5 [label=1, style=solid];\n\
            n1 -> n6 [label=0, style=dashed];\n}";

        let mut str = String::new();
        dot_str(&mut str, &dd).unwrap();
        assert_eq!(&str, exp);
    }
}
