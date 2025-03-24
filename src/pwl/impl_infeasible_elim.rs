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

use std::iter::zip;

use itertools::Itertools;
use log::{debug, error, info, warn};
use ndarray::concatenate;
use ndarray::prelude::*;

use super::afftree::*;
use super::node::*;
use crate::linalg::affine::*;
use crate::linalg::polyhedron::PolytopeStatus;
use crate::tree::graph::*;

#[derive(Debug, Clone)]
/// A collection of performance metrics of infeasible path elimination.
pub struct PerformanceCounter {
    pub nodes_checked: usize,
    pub cached_state: usize,
    pub skipped_nodes: usize,
    pub parent_sol_inherited: usize,
    pub mirror_iter: Vec<usize>,
    pub lps_solved: usize,
    pub lps_feasible: usize,
    pub lps_infeasible: usize,
    pub lps_error: usize,
}

impl Default for PerformanceCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceCounter {
    pub fn new() -> PerformanceCounter {
        PerformanceCounter {
            nodes_checked: 0,
            cached_state: 0,
            skipped_nodes: 0,
            parent_sol_inherited: 0,
            mirror_iter: Vec::with_capacity(32),
            lps_solved: 0,
            lps_feasible: 0,
            lps_infeasible: 0,
            lps_error: 0,
        }
    }
}

impl std::fmt::Display for PerformanceCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Infeasible Performance\n======================\nTOT: {:>9}\nSKI: {:>9}\nCAH: {:>9}\nINH: {:>9}\nMIR: {:>9}  (avg {:.2})\nLPS: {:>9}\nLPF: {:>9}  ({}%)\nLPI: {:>9}\nLPE: {:>9}",
            self.nodes_checked,
            self.skipped_nodes,
            self.cached_state,
            self.parent_sol_inherited,
            self.mirror_iter.len(),
            self.mirror_iter.iter().sum::<usize>() as f32 / self.mirror_iter.len() as f32,
            self.lps_solved,
            self.lps_feasible,
            ((self.lps_feasible as f32 / self.lps_solved as f32) * 100.0) as usize,
            self.lps_infeasible,
            self.lps_error
        )
    }
}

/// # Feasibility functions
impl<const K: usize> AffTree<K> {
    /// Tries to find points inside ``poly`` based on the given ``points`` in its proximity.
    ///
    /// This function iteratively maximizes the signed distance of each point to each of the
    /// supporting hyperplanes of ``poly``. When all distances are positive, the point lies
    /// inside the polytope.
    /// This is a heuristic that ignores the interaction between the hyperplanes for improved speed
    /// at the cost of completeness.
    /// If after ``n_iterations`` no solution is found, ``None`` is returned.
    /// Otherwise, if at any point a valid solution is found, the function returns all valid solutions up to that point.
    ///
    /// The first iteration corresponds to a simple containment check.
    ///
    /// The columns of ``points`` and of the returned value represent the points.
    pub fn mirror_points(
        poly: &Polytope,
        points: &Array2<f64>,
        n_iterations: usize,
    ) -> Option<(Array2<f64>, usize)> {
        assert_eq!(poly.indim(), points.shape()[0]);

        let poly_norm = poly.clone().normalize();

        let n_points = points.shape()[1];
        let mut candidates = points.to_owned();

        for count in 0..n_iterations {
            let b_bias = poly_norm
                .bias
                .broadcast((n_points, poly_norm.bias.shape()[0]))
                .unwrap();
            let mut distances = &b_bias.t() - &poly_norm.mat.dot(&candidates);

            // move distances a little away from the polytope to increase numerical robustness
            distances.map_inplace(|val| {
                *val -= 1e-10;
            });

            // check if new points are solutions of poly
            let contained_points = zip(candidates.axis_iter(Axis(1)), distances.axis_iter(Axis(1)))
                .filter(|(_, dist)| dist.iter().all(|val| *val >= 0.))
                .map(|(point, _)| point.insert_axis(Axis(1)))
                .collect_vec();

            if !contained_points.is_empty() {
                return Some((
                    concatenate(Axis(1), contained_points.as_slice()).unwrap(),
                    count,
                ));
            }

            distances.map_inplace(|val| {
                if *val >= 0. {
                    *val = 0.;
                } else {
                    *val *= 1.1;
                    *val -= 1e-10;
                }
            });

            for (mut point, dist) in zip(
                candidates.axis_iter_mut(Axis(1)),
                distances.axis_iter_mut(Axis(1)),
            ) {
                let step_vec = &poly_norm.mat
                    * &dist
                        .broadcast((poly_norm.mat.shape()[1], dist.shape()[0]))
                        .unwrap()
                        .t();

                point += &step_vec
                    .outer_iter()
                    .fold(Array1::zeros(poly_norm.mat.shape()[1]), |acc, x| acc + x);
            }
        }

        None
    }

    /// Creates a direct edge from the grandparent to the child of ``parent_idx``, effectively
    /// skipping it. Afterwards, ``parent_idx`` is removed from the graph.
    ///
    /// This is a useful optimization when ``parent_idx`` is redundant.
    pub fn merge_child_with_parent(
        &mut self,
        parent_idx: usize,
        label: Label,
    ) -> Result<TreeNode<AffContent, K>, NodeError> {
        self.tree.merge_child_with_parent(parent_idx, label)
    }

    /// Skips over redundant predicates in the tree.
    ///
    /// # Warning
    ///
    /// This function can alter the semantics of tree by removing paths that would
    /// otherwise result in errors. It is up to the caller to ensure that this
    /// behavior is either impossible (e.g., when all such paths are infeasible) or
    /// that this is indeed the intended effect.
    pub fn forward_if_redundant(&mut self, parent_idx: usize) -> Option<TreeNode<AffContent, K>> {
        // check if parent_idx has exactly one child
        let mut feasible_children = self
            .tree
            .children(parent_idx)
            .filter(|child| child.target_value.state.is_feasible())
            .collect_vec();

        if feasible_children.len() != 1 {
            return None;
        }

        // check if all other children are indeed infeasible
        let infeasible_children = self
            .tree
            .children(parent_idx)
            .filter(|child| child.target_value.state.is_infeasible())
            .map(|x| x.edge())
            .collect_vec();

        debug_assert!(feasible_children.len() + infeasible_children.len() <= K);
        if infeasible_children.len() != K - 1 {
            return None;
        }

        // remove the infeasible children and forward the feasible
        let feasible_child = feasible_children.pop().unwrap().edge();

        for edg in &infeasible_children {
            self.tree.remove_child(parent_idx, edg.label);
        }

        self.tree
            .merge_child_with_parent(feasible_child.source_idx, feasible_child.label)
            .ok()
    }

    /// Removes nodes from this tree that lie on infeasible paths.
    ///
    /// In a decision structure, a path is called infeasible when the conjunction of
    /// all decisions on that path is not satisfiable. As no input could ever take
    /// such a path, these can be safely removed from the decision structure without
    /// altering its semantics (i.e., without changing the represented piece-wise
    /// linear function).
    pub fn infeasible_elimination(&mut self) -> PerformanceCounter {
        let mut counter = PerformanceCounter::new();

        let mut to_remove: Vec<(Label, TreeIndex)> = Vec::with_capacity(16);

        let mut iter = self.polyhedra();
        while let Some((data, polyhedra)) = iter.next(&self.tree) {
            let (depth, node_idx, n_remaining) = data.extract();
            debug!("Visiting node {node_idx} in depth {depth}");

            counter.nodes_checked += 1;

            if node_idx == self.tree.get_root_idx() {
                continue;
            }

            let node_value = self.tree.node_value(node_idx).unwrap();

            // check node for cached solutions from previous runs
            // continue only when the node hasn't been checked before (NodeState::Indeterminate)
            match &node_value.state {
                NodeState::Infeasible => {
                    iter.skip_subtree();
                    counter.cached_state += 1;
                    counter.skipped_nodes += self.tree.num_nodes(node_idx) - 1;
                    continue;
                }
                NodeState::Feasible | NodeState::FeasibleWitness(_) => {
                    counter.cached_state += 1;
                    continue;
                }
                NodeState::Indeterminate => {}
            }

            let (parent_idx, _parent_value, label) = {
                let edg = self.tree.parent(node_idx).unwrap();
                (edg.source_idx, edg.source_value, edg.label)
            };

            let hyperplane = polyhedra.last().unwrap();
            // the polytope implied by the path from root to current node
            let poly = Polytope::intersection_n(self.in_dim(), polyhedra.as_slice());

            let mut state = self.phase_inh(parent_idx, hyperplane, &mut counter);

            if matches!(state, NodeState::Indeterminate) {
                state = self.phase_one(parent_idx, &poly, &mut counter);
            }

            if matches!(state, NodeState::Indeterminate) {
                state = self.phase_two(node_idx, &poly, &mut counter);
            }

            if let NodeState::Infeasible = &state {
                to_remove.push((label, parent_idx));
                // self.tree.remove_child(parent_idx, label);
                iter.skip_subtree();
                counter.skipped_nodes += self.tree.num_nodes(node_idx) - 1;
            }

            let node_value = self.tree.node_value_mut(node_idx).unwrap();
            node_value.state = state;

            // after all siblings have been checked, clean up if parent is redundant
            if n_remaining == 0 {
                self.forward_if_redundant(parent_idx);
            }
        }

        for (label, node) in to_remove {
            let _ = self.tree.try_remove_child(node, label);
        }

        counter
    }

    /// Tests if a solution can be inherited from ``parent_idx``.
    ///
    /// That is, test if any cached solution of parent satisfies the additional constraint ``hyperplane``.
    fn phase_inh(
        &self,
        parent_idx: TreeIndex,
        hyperplane: &Polytope,
        counter: &mut PerformanceCounter,
    ) -> NodeState {
        let parent_value = self.tree.node_value(parent_idx).unwrap();
        if let NodeState::FeasibleWitness(solution) = &parent_value.state {
            assert!(
                !solution.is_empty(),
                "nodes with the state FeasibleWitness should conatain a non-empty vector"
            );

            let inherited_solutions = solution
                .iter()
                .filter(|point| hyperplane.contains(point))
                .map(|point| point.to_owned())
                .collect_vec();

            if !inherited_solutions.is_empty() {
                debug!("solution of parent (id={}) can be inherited", parent_idx);
                counter.parent_sol_inherited += 1;
                return NodeState::FeasibleWitness(inherited_solutions);
            }
        }
        NodeState::Indeterminate
    }

    /// Tries to derive new solutions based on the cached solutions of the parent node.
    /// The returned node state is with respect to the child that matches poly.
    fn phase_one(
        &self,
        parent_idx: TreeIndex,
        poly: &Polytope,
        counter: &mut PerformanceCounter,
    ) -> NodeState {
        let parent_value = self.tree.node_value(parent_idx).unwrap();
        match &parent_value.state {
            NodeState::FeasibleWitness(solution) => {
                assert!(
                    !solution.is_empty(),
                    "nodes with the state FeasibleWitness should conatain a non-empty vector"
                );

                if solution.len() > 64 {
                    warn!(
                        "number of considered solutions should be bounded for performance reasons"
                    );
                }

                // pack all solutions into a 2D array such that each column represents one solution
                let mut array = Array2::<f64>::zeros((solution[0].shape()[0], solution.len()));
                for (mut column, point) in zip(array.axis_iter_mut(Axis(1)), solution.iter()) {
                    column.assign(point);
                }

                let node_solutions = AffTree::<K>::mirror_points(poly, &array, 8);

                if let Some((node_solutions, iter)) = node_solutions {
                    let vec = node_solutions
                        .axis_iter(Axis(1))
                        .map(|x| x.to_owned())
                        .collect_vec();

                    debug!(
                        "mirror heuristic derived {} solution(s) in {} iteration(s)",
                        vec.len(),
                        iter
                    );

                    debug_assert!(
                        iter > 0,
                        "if solutions can be inherited, it should have already occurred in a previous step"
                    );
                    debug_assert!(vec.iter().all(|point| poly.contains(point)));
                    counter.mirror_iter.push(iter);

                    return NodeState::FeasibleWitness(vec);
                }
            }
            NodeState::Feasible => {
                warn!("parent cache should store witnesses until all children are processed")
            }
            NodeState::Infeasible => error!(
                "when a parent node is infeasible, non of its children should be processed further"
            ),
            NodeState::Indeterminate => {
                info!("Undesired state: Parent node of child has indetermined state")
            }
        }
        NodeState::Indeterminate
    }

    /// Determines if the polytope is feasible utilizing LP solvers.
    fn phase_two(
        &self,
        _node_idx: TreeIndex,
        poly: &Polytope,
        counter: &mut PerformanceCounter,
    ) -> NodeState {
        debug!("Phase II. Solving LP {:?}", &poly);
        counter.lps_solved += 1;

        // let (ineqs, costs) = poly.chebyshev_center();
        // ineqs.solve_linprog(costs, false);

        match poly.status() {
            PolytopeStatus::Optimal(solution) => {
                if !poly.contains(&solution) {
                    let new_solution =
                        Self::mirror_points(poly, &solution.clone().insert_axis(Axis(1)), 20);

                    if let Some((val, _)) = new_solution {
                        let new_solution = val.t().row(0).to_owned();

                        if !poly.contains(&new_solution) {
                            error!(
                                "LP solver returned an incorrect solution (distance={}) that could not be fixed",
                                poly.distance(&solution)
                            );
                            counter.lps_error += 1;
                            NodeState::Indeterminate
                        } else {
                            warn!("LP solver returned an incorrect solution that could be fixed");
                            counter.lps_feasible += 1;
                            NodeState::FeasibleWitness(vec![new_solution.to_owned()])
                        }
                    } else {
                        error!(
                            "LP solver returned an incorrect solution (distance={}) that could not be fixed",
                            poly.distance(&solution)
                        );
                        counter.lps_error += 1;
                        NodeState::Indeterminate
                    }
                } else {
                    counter.lps_feasible += 1;
                    NodeState::FeasibleWitness(vec![solution])
                }
            }
            PolytopeStatus::Infeasible => {
                counter.lps_infeasible += 1;
                NodeState::Infeasible
            }
            PolytopeStatus::Unbounded => NodeState::Feasible,
            PolytopeStatus::Error(err_msg) => {
                error!("LP solver terminated with error: {}", err_msg);
                counter.lps_error += 1;
                NodeState::Indeterminate
            }
        }
    }

    /// Tests if the edge from ``parent_idx`` to ``node_idx`` is feasible.
    ///
    /// ***Deprecated:*** use [`AffTree::infeasible_elimination`] to prune infeasible paths instead.
    #[deprecated(note = "use `infeasible_elimination` instead", since = "0.21.0")]
    pub fn is_edge_feasible(&self, parent_idx: usize, node_idx: usize) -> bool {
        // Edges from the root node are always feasible
        if parent_idx == 0 {
            return true;
        }

        // Check for cached solutions
        let node = self.tree.node_value(node_idx).unwrap();
        match &node.state {
            NodeState::Infeasible => return false,
            NodeState::Feasible | NodeState::FeasibleWitness(_) => return true,
            NodeState::Indeterminate => {}
        }

        let mut path = self.tree.path_to_node(parent_idx).unwrap();
        debug!("Checking path for feasibility: {:?}", &path);

        let label = self
            .tree
            .parent(node_idx)
            .expect("edge not contained in graph")
            .label;
        path.push((parent_idx, label));

        // Calculate polytope representing the path
        let poly = self.polyhedral_path_characterization(&path);
        debug!("Path condition: {}", &poly);

        // Check if solution from parent node solves linear program -> superset
        let parent_node = self.tree.tree_node(parent_idx).unwrap();
        match &parent_node.value.state {
            NodeState::Infeasible => return false,
            NodeState::FeasibleWitness(wit) => {
                if wit.iter().any(|point| poly.contains(point)) {
                    return true;
                }
            }
            NodeState::Feasible | NodeState::Indeterminate => {}
        }

        // No existing solutions found, calculate a new solution based on polytope
        debug!("Passing LP to solver: {poly:?}");
        let status = poly.status();
        debug!("Solver status {status:?}");

        match status {
            PolytopeStatus::Infeasible => false,
            PolytopeStatus::Unbounded => {
                warn!(
                    "Target function of LP is reported as unbounded, but it is constant.\nPolytope: {:?}",
                    &poly
                );
                true
            }
            PolytopeStatus::Optimal(_solution) => true,
            PolytopeStatus::Error(err) => {
                // Definitely not good when the LP solver fails, but it is recoverable.
                // Therefore it would be wrong to panic.
                warn!(
                    "Error occurred while solving the linear program for an infeasible path!\nSolver status: {:?}\nPolytope: {:?}",
                    err, &poly
                );
                true
            }
        }
    }

    /// Collects the path conditions of the given ``path`` as a [``Polytope``].
    fn polyhedral_path_characterization(&self, path: &Vec<(TreeIndex, Label)>) -> Polytope {
        let mut cache = self.polytope_cache.borrow_mut();
        cache.clear();
        cache.reserve(path.len());

        for (idx, label) in path {
            let current_node = self.tree.tree_node(*idx).unwrap();

            assert!(
                !current_node.isleaf,
                "path should only contain decision nodes"
            );

            let factor = match label {
                1 => 1.0,
                0 => -1.0,
                _ => panic!("label should be 0 or 1, but got {}", &label),
            };

            let aff = &current_node.value.aff;
            let poly_node = Polytope::from_mats(&aff.mat * factor, &aff.bias * factor);
            cache.push(poly_node);
        }
        let in_dim = self.in_dim();

        let poly = Polytope::intersection_n(in_dim, cache.as_slice());
        cache.clear();
        poly
    }
}

#[cfg(test)]
mod tests {
    use assertables::*;

    use super::*;
    use crate::linalg::affine::{AffFunc, PolyRepr, Polytope};
    use crate::{aff, path, poly};

    fn init_logger() {
        use env_logger::Target;
        use log::LevelFilter;

        let _ = env_logger::builder()
            .is_test(true)
            .filter_module("minilp", LevelFilter::Error)
            .target(Target::Stdout)
            .filter_level(LevelFilter::Debug)
            .try_init();
    }

    /// Constructs a simple balanced tree with depth 3.
    /// One path is infeasible.
    fn construct_simple_tree() -> AffTree<2> {
        let mut tree = AffTree::<2>::from_aff(aff!([[0, 1, 0]] + [2]));
        tree.add_child_node(0, 0, aff!([[1, 1, 1]] + [1])).unwrap();
        tree.add_child_node(0, 1, aff!([[0, 1, 0]] + [3])).unwrap();
        tree.add_child_node(1, 0, aff!([[0, 0, 0]] + [1])).unwrap();
        tree.add_child_node(1, 1, aff!([[0, 0, 0]] + [2])).unwrap();
        tree.add_child_node(2, 0, aff!([[0, 0, 0]] + [3])).unwrap();
        tree.add_child_node(2, 1, aff!([[0, 0, 0]] + [4])).unwrap();

        tree
    }

    #[test]
    pub fn test_polyhedral_path_characterization() {
        init_logger();

        let tree = construct_simple_tree();

        assert_eq!(
            tree.polyhedral_path_characterization(&vec![(0, 1), (2, 0)]),
            poly!([[0, 1, 0], [0, -1, 0]] < [2, -3])
        );

        assert_eq!(
            tree.polyhedral_path_characterization(&vec![(0, 0), (1, 1)]),
            poly!([[0, -1, 0], [1, 1, 1]] < [-2, 1])
        );
    }

    #[test]
    pub fn test_is_edge_feasible() {
        init_logger();

        let tree = construct_simple_tree();

        assert!(tree.is_edge_feasible(0, 1));
        assert!(tree.is_edge_feasible(0, 2));
        assert!(tree.is_edge_feasible(1, 3));
        assert!(tree.is_edge_feasible(1, 4));
        assert!(!tree.is_edge_feasible(2, 5));
        assert!(tree.is_edge_feasible(2, 6));
    }

    #[test]
    pub fn test_infeasible_elimination() {
        init_logger();

        let mut tree = construct_simple_tree();

        tree.infeasible_elimination();

        let nodes: Vec<usize> = tree.tree.node_iter().map(|(idx, _)| idx).collect();
        assert_not_contains!(nodes, &2); // forwarding
        assert_not_contains!(nodes, &5); // infeasible
        assert!(tree.tree.tree_node(6).unwrap().isleaf);
    }

    /// Constructs a tree consisting of a single path with two terminals at its end
    /// On terminal (index 7) is feasible with solutions in the upper quadrant of R^2.
    /// The other terminal (index 8) is infeasible.
    fn construct_list_like_tree() -> AffTree<2> {
        let mut tree = AffTree::<2>::from_aff(aff!([[-2, -1]] + [-1])); // 0

        tree.add_child_node(0, 1, aff!([[-1., -2.]] + [-1.5]))
            .unwrap(); // 1
        tree.add_child_node(1, 1, aff!([[-0.5, -5.]] + [1.0]))
            .unwrap(); // 2
        tree.add_child_node(2, 1, aff!([[-3., 1.]] + [0.])).unwrap(); // 3
        tree.add_child_node(3, 1, aff!([[1., 1.]] + [6.])).unwrap(); // 4
        tree.add_child_node(4, 0, aff!([[1., -7.]] + [4.])).unwrap(); // 5
        tree.add_child_node(5, 1, aff!([[2., 0.2]] + [3.])).unwrap(); // 6
        // feasible
        tree.add_child_node(6, 0, aff!([[0., 0.]] + [1.])).unwrap(); // 7
        // infeasible
        tree.add_child_node(6, 1, aff!([[0., 0.]] + [0.])).unwrap(); // 8

        tree
    }

    #[test]
    pub fn test_is_feasible() {
        init_logger();

        let dd = construct_list_like_tree();

        assert!(dd.is_edge_feasible(6, 7));
        assert!(!dd.is_edge_feasible(6, 8));

        assert_eq!(dd.evaluate(&arr1(&[6., 6.])).unwrap(), arr1(&[1.]));
        assert_eq!(dd.evaluate(&arr1(&[4.5, 2.5])).unwrap(), arr1(&[1.]));

        assert_eq!(
            dd.tree.path_to_node(7).unwrap(),
            &[(0, 1), (1, 1), (2, 1), (3, 1), (4, 0), (5, 1), (6, 0)]
        );
        assert_eq!(
            dd.tree.path_to_node(8).unwrap(),
            &[(0, 1), (1, 1), (2, 1), (3, 1), (4, 0), (5, 1), (6, 1)]
        );

        let poly = Polytope::from_mats(
            arr2(&[
                [-2., -1.],
                [-1., -2.],
                [-0.5, -5.],
                [-3., 1.],
                [-1., -1.],
                [1., -7.],
                [-2., -0.2],
            ]),
            arr1(&[-1., -1.5, 1.0, 0., -6., 4., -3.]),
        );

        assert_eq!(
            dd.polyhedral_path_characterization(&dd.tree.path_to_node(7).unwrap()),
            poly
        );
    }

    #[test]
    pub fn test_infeasible_elimination_feasible_path_2d() {
        init_logger();
        // Construct a polytope where every hyperplane is supportive (i.e., no hyperplane is redundant)
        // Solutions are roughly contained inside [-4, 4] x [-1, 7]

        let poly = poly!(
            [
                [1, 1],
                [-1, -2],
                [-2, -1],
                [1, 5],
                [7, 1],
                [-2, -3],
                [1, -3]
            ] < [5, -2, 0.5, 30, 28, -4, 5]
        );

        let mut dd = AffTree::<2>::from_poly(poly, AffFunc::identity(2), None).unwrap();

        dd.infeasible_elimination();

        // Each of the seven hyperplanes is required + one terminal
        assert_eq!(dd.len(), 8);
    }

    #[test]
    pub fn test_infeasible_elimination_feasible_path_6d() {
        init_logger();
        // Project the previous test case into higher dimensions

        let poly = poly!(
            [
                [-2, -1, 1, 1, -1, 0],
                [-1, -2, -1, -2, 0, -1],
                [-0.5, -5, -2, -1, 1, 0],
                [-3, 1, 1, 5, 0, 1],
                [1, 1, 7, 1, -1, -1],
                [1, 7, -2, -3, 1, 1],
                [2, 0.2, 1, -3, -1, 1]
            ] < [3, -4.5, 3.5, 34, 31, 6, 6]
        );

        let mut dd = AffTree::<2>::from_poly(poly, AffFunc::identity(6), None).unwrap();

        let idx = dd
            .add_child_node(
                path!(dd.tree, 1, 1, 1, 1),
                0,
                poly!([[1, 1, 1, 1, 1, 1]] < [-20]).convert_to(PolyRepr::MatrixBiasGeqZero),
            )
            .unwrap();
        dd.add_child_node(idx, 0, AffFunc::constant(6, -1.0))
            .unwrap();
        dd.add_child_node(idx, 1, AffFunc::constant(6, 1.0))
            .unwrap();

        assert_eq!(dd.len(), 11);
        assert_eq!(dd.num_terminals(), 3);

        dd.infeasible_elimination();

        // 7 conditions from poly + 1 terminal + 1 decision + 2 terminals
        assert_eq!(dd.len(), 11);
        assert_eq!(dd.num_terminals(), 3);
    }

    #[test]
    pub fn test_infeasible_elimination_removes_node() {
        init_logger();

        let mut dd = construct_list_like_tree();

        // verify that node 7 is indeed feasible
        assert_eq!(dd.evaluate(&arr1(&[3., 6.])).unwrap()[0], 1.);
        assert_eq!(dd.evaluate(&arr1(&[5., 4.])).unwrap()[0], 1.);
        assert_eq!(dd.evaluate(&arr1(&[7., 2.])).unwrap()[0], 1.);

        // test that not all inputs are contained
        assert!(dd.evaluate(&arr1(&[1., 8.])).is_none());
        assert!(dd.evaluate(&arr1(&[2., 1.])).is_none());
        assert!(dd.evaluate(&arr1(&[5., 0.])).is_none());
        assert!(dd.evaluate(&arr1(&[10., -1.])).is_none());

        dd.infeasible_elimination();

        let indices = dd.tree.node_indices().collect_vec();
        // infeasible node removed
        assert_not_contains!(indices, &8);
        // redundant parent forwarded
        assert_not_contains!(indices, &6);

        // feasible nodes kept
        assert_contains!(indices, &0);
        assert_contains!(indices, &1);
        assert_contains!(indices, &2);
        assert_contains!(indices, &3);
        assert_contains!(indices, &4);
        assert_contains!(indices, &5);
        assert_contains!(indices, &7);

        // verify evaluation still correct
        assert_eq!(dd.evaluate(&arr1(&[3., 6.])).unwrap()[0], 1.);
        assert_eq!(dd.evaluate(&arr1(&[5., 4.])).unwrap()[0], 1.);
        assert_eq!(dd.evaluate(&arr1(&[7., 2.])).unwrap()[0], 1.);
    }

    #[test]
    fn test_mirror() {
        init_logger();

        let poly = poly!([[1, 0], [0, 1]] > [2, 3]);
        let points = array![[-2.], [-7.]];

        let sol = AffTree::<2>::mirror_points(&poly, &points, 2);

        assert!(sol.is_some());
        for point in sol.unwrap().0.axis_iter(Axis(1)) {
            assert!(poly.contains(&point));
        }
    }

    #[test]
    fn test_mirror2() {
        init_logger();

        let poly = poly!([[1, 0], [0, 1], [1, 1], [2, 1], [-1, 1], [1, -1]] > [2, 3, 4, 2, -2, -4]);
        let points = array![[-8., -20.], [-12., -1.]];

        let sol = AffTree::<2>::mirror_points(&poly, &points, 5);

        assert!(sol.is_some());
        for point in sol.unwrap().0.axis_iter(Axis(1)) {
            assert!(poly.contains(&point));
        }
    }
}
