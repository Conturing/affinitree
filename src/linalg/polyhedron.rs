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

//! Feasibility tests for polytopes based on LP solving

use minilp::{Problem, Variable};

use ndarray::{self, Array1};
use std::iter::zip;

use super::affine::Polytope;

#[derive(Clone, Debug)]
pub struct LinearProgram {
    pub solver: Problem,
    pub vars: Vec<Variable>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PolytopeStatus {
    Infeasible,
    Unbounded,
    Optimal(Array1<f64>),
    Error(String),
}

/* LP solving */
impl Polytope {
    #[inline]
    pub fn is_feasible(&self) -> bool {
        match self.status() {
            PolytopeStatus::Infeasible => false,
            PolytopeStatus::Optimal(_) | PolytopeStatus::Unbounded => true,
            PolytopeStatus::Error(err) => panic!("Polytope is not well formed: {err:?}"),
        }
    }

    #[inline]
    pub fn status(&self) -> PolytopeStatus {
        self.status_minilp()
    }

    #[inline]
    pub fn status_minilp(&self) -> PolytopeStatus {
        match self.solve_linprog_minilp(Array1::zeros(self.mat.raw_dim()[1]), false) {
            Ok((_, sol)) => PolytopeStatus::Optimal(sol),
            Err(minilp::Error::Infeasible) => PolytopeStatus::Infeasible,
            Err(minilp::Error::Unbounded) => PolytopeStatus::Unbounded,
        }
    }

    pub fn solve_linprog_minilp(
        &self,
        coeffs: Array1<f64>,
        _verbose: bool,
    ) -> Result<(minilp::Solution, Array1<f64>), minilp::Error> {
        let problem = self.as_linprog(coeffs);
        let pb = problem.solver;
        let vars = problem.vars;

        match pb.solve() {
            Ok(sol) => {
                let wit = Array1::from_iter(vars.iter().map(|var| sol[*var]));
                Ok((sol, wit))
            }
            Err(x) => Err(x),
        }
    }

    /// Solves a linear program built form this polytope and coeffs as target func.
    /// Concretly, the resulting linear program is
    /// min coeffs.T x
    /// s.t. mat x <= bias
    pub fn as_linprog(&self, cost_function: Array1<f64>) -> LinearProgram {
        use minilp::{ComparisonOp, OptimizationDirection};

        let mut pb = Problem::new(OptimizationDirection::Minimize);

        // create the variables for the linear program (objective function + variable bounds)
        let vars: Vec<Variable> = cost_function
            .iter()
            .map(|x| pb.add_var(*x, (f64::NEG_INFINITY, f64::INFINITY)))
            .collect();

        // add linear constraints
        for (row, bias) in zip(self.mat.rows(), &self.bias) {
            let constraint: Vec<(Variable, f64)> =
                zip(&vars, row).map(|(var, coeff)| (*var, *coeff)).collect();

            // set bias as upper bound (inclusive) of the linear constraint
            pb.add_constraint(constraint.as_slice(), ComparisonOp::Le, *bias);
        }

        LinearProgram {
            solver: pb,
            vars: vars,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::polyhedron::Polytope;
    use crate::poly;
    use approx::assert_relative_eq;

    use ndarray::{arr1, arr2, array, s, Array1};

    fn init_logger() {
        // minilp has a bug if logging is enabled
        // match fast_log::init(Config::new().console().chan_len(Some(100000))) {
        //     Ok(_) => (),
        //     Err(err) => println!("Error occurred while configuring logger: {:?}", err),
        // }
    }

    #[test]
    pub fn test_lp_solve_feasible01() {
        init_logger();

        let weights = arr2(&[
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-2.0, -1.0],
        ]);
        let bias = arr1(&[-1.0, -1.0, 4.0, 4.0, -3.0, 6.0, -2.0, -8.0]);
        let poly = Polytope::from_mats(weights, bias);

        assert!(poly.is_feasible());
    }

    #[test]
    pub fn test_lp_solve_infeasible01() {
        init_logger();

        let weights = arr2(&[
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-2.0, -1.0],
        ]);
        let bias = arr1(&[-1.0, -1.0, 4.0, 4.0, -3.0, 6.0, -2.0, -11.0]);
        let poly = Polytope::from_mats(weights, bias);

        assert!(!poly.is_feasible());
    }

    #[test]
    pub fn test_lp_solve_infeasible02() {
        init_logger();

        let weights = arr2(&[
            [
                -0.3255753815174103,
                -0.2293112725019455,
                -0.6689934134483337,
                0.12802177667617798,
                0.3714849054813385,
                1.7622292041778564,
                0.3637314438819885,
            ],
            [
                1.090343952178955,
                0.659135103225708,
                -0.5783752799034119,
                0.3744732737541199,
                0.0975840762257576,
                0.29021647572517395,
                0.32744163274765015,
            ],
            [
                -0.43671178817749023,
                0.20265570282936096,
                -0.04638584330677986,
                -1.1266447305679321,
                -0.9601702690124512,
                0.8724989891052246,
                1.321545958518982,
            ],
            [
                1.017880916595459,
                -0.41071557998657227,
                -0.3797197639942169,
                0.8332198858261108,
                0.02103915438055992,
                0.10907302051782608,
                0.7036119699478149,
            ],
            [
                -0.9301143288612366,
                -1.036597490310669,
                -1.14026939868927,
                0.274497389793396,
                0.02076120860874653,
                -2.2522382736206055,
                0.01438702829182148,
            ],
            [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
        ]);
        let bias = arr1(&[
            -0.004004549700766802,
            -0.6895583271980286,
            -0.1956268846988678,
            0.3603391647338867,
            -0.8825269341468811,
            -0.12980055809020996,
        ]);
        let poly = Polytope::from_mats(weights, bias);

        assert!(!poly.is_feasible());
    }

    #[test]
    pub fn test_lp_solve_infeasible03() {
        init_logger();

        let weights = arr2(&[[0.0, 0.0, 0.0, 0.0]]);
        let bias = arr1(&[-0.12]);
        let poly = Polytope::from_mats(weights, bias);

        assert!(!poly.is_feasible());
    }

    #[test]
    pub fn test_lp_solve_infeasible04() {
        init_logger();

        let weights = arr2(&[
            [1.0, -2.0, -3.0],
            [2.0, -1.0, 3.0],
            [-3.0, -1.0, 2.0],
            [-10.0, -1.0, 9.0],
            [9.0, -4.0, -13.0],
            [14.0, -7.0, -21.0],
            [-55.0, -2.0, 53.0],
            [-160.0, 5.0, 165.0],
            [193.0, -78.0, -271.0],
            [999.0, -122.0, -1121.0],
            [311.0, -13.0, -324.0],
            [-81719.0, 60517.0, 142236.0],
        ]);
        let bias = arr1(&[
            -3.0, -1.0, -2.0, -6.0, -7.0, -21.0, -17.0, -56.0, -109.0, 61.0, -169.0, 86782.0,
        ]);

        let poly = Polytope::from_mats(
            weights.slice(s![..10, ..]).to_owned(),
            bias.slice(s![..10]).to_owned(),
        );
        assert!(poly.is_feasible());

        let poly = Polytope::from_mats(weights, bias);
        assert!(!poly.is_feasible());
    }

    #[test]
    pub fn test_lp_solve_feasible04() {
        init_logger();

        let weights = arr2(&[
            [1.0, -2.0, -3.0],
            [2.0, -1.0, 3.0],
            [-3.0, -1.0, 2.0],
            [-10.0, -1.0, 9.0],
            [9.0, -4.0, -13.0],
            [14.0, -7.0, -21.0],
            [-55.0, -2.0, 53.0],
            [-160.0, 5.0, 165.0],
            [193.0, -78.0, -271.0],
            [999.0, -122.0, -1121.0],
            [311.0, -13.0, -324.0],
            [-81719.0, 60517.0, 142236.0],
        ]);
        let bias = arr1(&[
            -3.0, -1.0, -2.0, -6.0, -7.0, -21.0, -17.0, -56.0, -109.0, 61.0, -169.0, 10000000.0,
        ]);

        let poly = Polytope::from_mats(weights, bias);
        assert!(poly.is_feasible());
    }

    #[test]
    pub fn test_lp_solve_infeasible05() {
        init_logger();

        let weights = arr2(&[
            [1.0, -2.0, -3.0],
            [2.0, -1.0, 3.0],
            [-3.0, -1.0, 2.0],
            [-10.0, -1.0, 9.0],
            [9.0, -4.0, -13.0],
            [14.0, -7.0, -21.0],
            [-55.0, -2.0, 53.0],
            [-160.0, 5.0, 165.0],
            [193.0, -78.0, -271.0],
            [999.0, -122.0, -1121.0],
            [311.0, -13.0, -324.0],
            [-81719.0, 60517.0, 142236.0],
        ]);
        let bias = arr1(&[
            -3.0, -1.0, -2.0, -6.0, -7.0, -21.0, -17.0, -56.0, -109.0, 61.0, -169.0, 0.0,
        ]);

        let poly = Polytope::from_mats(
            weights.slice(s![2.., ..]).to_owned(),
            bias.slice(s![2..]).to_owned(),
        );
        assert!(!poly.is_feasible());
    }

    #[test]
    pub fn test_lp_solve_infeasible06() {
        init_logger();

        let weights = arr2(&[
            [1.0, -2.0, -3.0],
            [-2.0, 1.0, -3.0],
            [-3.0, -1.0, 2.0],
            [-14.0, 1.0, 3.0],
            [-5.0, -8.0, 3.0],
            [7.0, -3.0, -16.0],
            [25.0, -26.0, -45.0],
            [-2.0, 1.0, -45.0],
            [-61.0, 1.0, 44.0],
            [-212.0, 31.0, 87.0],
            [-131.0, -50.0, 87.0],
            [195.0, -79.0, -268.0],
            [535.0, -368.0, -717.0],
            [292.0, -125.0, -717.0],
            [-1157.0, 201.0, 884.0],
            [3422.0, -721.0, -1935.0],
            [-2693.0, -8.0, 1935.0],
            [4211.0, -1631.0, -4636.0],
            [-7247.0, 4909.0, 10038.0],
            [15326.0, -4885.0, -15843.0],
            [-11115.0, 3254.0, 11207.0],
        ]);
        let bias = arr1(&[
            -3.0, 1.0, -2.0, -4.0, -12.0, -6.0, -41.0, -13.0, -14.0, -30.0, -110.0, -110.0, -523.0,
            -279.0, 18.0, -16.0, -712.0, -1886.0, 7085.0, -4945.0, 3058.0,
        ]);

        let poly = Polytope::from_mats(
            weights.slice(s![..10, ..]).to_owned(),
            bias.slice(s![..10]).to_owned(),
        );
        assert!(poly.is_feasible());

        let poly = Polytope::from_mats(weights, bias);
        assert!(!poly.is_feasible());
    }

    #[test]
    pub fn test_lp_solve_infeasible07() {
        init_logger();

        let weights = arr2(&[[-2.0, 2.0], [-0.0, -1.0]]);
        let bias = arr1(&[-1.0, -1.0]);

        let poly = Polytope::from_mats(weights, bias);
        assert!(poly.is_feasible());
    }

    #[test]
    pub fn test_solve_feasible() {
        let poly = poly!(
            [
                [-2, -1],
                [-1, -2],
                [-0.5, -5],
                [-3, 1],
                [1, 1],
                [1, -7],
                [2, 0.2]
            ] < [-1, -1.5, -1, 0, 6, 4, 3]
        );

        assert!(poly.solve_linprog_minilp(Array1::zeros(2), false).is_ok());
    }

    #[test]
    pub fn test_solve_feasible_2() {
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

        assert!(poly.solve_linprog_minilp(Array1::zeros(2), false).is_ok());
    }

    #[test]
    pub fn test_solve_feasible_3() {
        let poly = poly!(
            [
                [-1, 0],
                [0, -1],
                [1, 0],
                [0, 1],
                [-1, -1],
                [1, 1],
                [-1, 1],
                [-2, -1]
            ] < [-1, -1, 4, 4, -3, 6, -2, -8]
        );

        assert!(poly.solve_linprog_minilp(Array1::zeros(2), false).is_ok());
    }

    #[test]
    pub fn test_chebyshev_box() {
        let poly = Polytope::from_mats(
            array![[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
            array![1., 1., 1., 1.],
        );

        let (p, c) = poly.chebyshev_center();

        let (_, sol) = p.solve_linprog_minilp(c, false).unwrap();

        assert_relative_eq!(
            sol,
            array![0., 0., 1.],
            epsilon = 1e-08,
            max_relative = 1e-02
        );
    }

    #[test]
    pub fn test_chebyshev_box_2() {
        let poly = Polytope::from_mats(
            array![[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
            array![2., 1., 1., 1.],
        );

        let (p, c) = poly.chebyshev_center();

        let (_, sol) = p.solve_linprog_minilp(c, false).unwrap();

        assert!(0.0 <= sol[0]);
        assert!(sol[0] <= 1.0);
        assert_eq!(sol[1], 0.0);
        assert_eq!(sol[2], 1.0);
    }

    #[test]
    pub fn test_chebyshev_triangle() {
        let poly = Polytope::from_mats(array![[1., 1.], [-1., 1.], [0., -1.]], array![0., 0., 2.4]);

        let (p, c) = poly.chebyshev_center();

        let (_, sol) = p.solve_linprog_minilp(c, false).unwrap();

        assert_relative_eq!(
            sol,
            array![0., -1.414, 1.],
            epsilon = 1e-08,
            max_relative = 1e-02
        );
    }
}
