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

#[cfg(feature = "minilp")]
use minilp::{Problem, Variable};

#[cfg(feature = "highs")]
use highs::{self, Col, HighsModelStatus, RowProblem, Sense, SolvedModel};

use ndarray::{self, Array1};
use std::iter::zip;

use super::affine::Polytope;

#[cfg(feature = "minilp")]
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

/// # LP solving
impl Polytope {
    /// Tests if the polytope is feasible.
    #[inline]
    pub fn is_feasible(&self) -> bool {
        match self.status() {
            PolytopeStatus::Infeasible => false,
            PolytopeStatus::Optimal(_) | PolytopeStatus::Unbounded => true,
            PolytopeStatus::Error(err) => panic!("Polytope is not well formed: {err:?}"),
        }
    }

    /// Tests if the polytope is feasible
    #[inline]
    pub fn status(&self) -> PolytopeStatus {
        self.solve_linprog(Array1::zeros(self.mat.raw_dim()[1]), false)
    }

    /// Solves a linear program built form this polytope and coeffs as target func.
    /// Concretely, the resulting linear program is
    /// min coeffs.T x
    /// s.t. mat x <= bias
    #[cfg(feature = "minilp")]
    pub fn solve_linprog(&self, coeffs: Array1<f64>, _verbose: bool) -> PolytopeStatus {
        let problem = self.as_linprog(coeffs);
        let pb = problem.solver;
        let vars = problem.vars;

        match pb.solve() {
            Ok(sol) => {
                let wit = Array1::from_iter(vars.iter().map(|var| sol[*var]));
                PolytopeStatus::Optimal(wit)
            }
            Err(minilp::Error::Infeasible) => PolytopeStatus::Infeasible,
            Err(minilp::Error::Unbounded) => PolytopeStatus::Unbounded,
        }
    }

    /// Solves a linear program built form this polytope and coeffs as target func.
    /// Concretly, the resulting linear program is
    /// min coeffs.T x
    /// s.t. mat x <= bias
    #[cfg(feature = "minilp")]
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

        // print!("{:?}", solved.status());
        // print!("{}", solved.get_solution().columns()[0]);
        LinearProgram { solver: pb, vars }
    }

    /// Solves a linear program built form this polytope and coeffs as target func.
    /// Concretely, the resulting linear program is
    /// min coeffs.T x
    /// s.t. mat x <= bias
    #[cfg(feature = "highs")]
    pub fn solve_linprog(&self, coeffs: Array1<f64>, verbose: bool) -> PolytopeStatus {
        let mut pb = RowProblem::default();

        // create the variables for the linear program (objective function + variable bounds)
        let vars: Vec<Col> = coeffs
            .iter()
            .map(|x| pb.add_column::<f64, _>(*x, ..))
            .collect();

        // add linear constraints
        for (row, bias) in zip(self.mat.rows(), &self.bias) {
            let constraint: Vec<(Col, f64)> =
                zip(&vars, row).map(|(var, coeff)| (*var, *coeff)).collect();

            // set bias as upper bound (inclusive) of the linear constraint
            pb.add_row(..=*bias, constraint)
        }

        let mut model = pb.optimise(Sense::Minimise);
        // Performance improvement of around 10%
        model.set_option("threads", 1);

        // Presolver does not work in our case
        //TODO: test this option
        // presolve detects trivial infeasibilities and is atually needed in such cases
        // model.set_option("presolve", "on");
        model.set_option("presolve", "off");

        if verbose {
            model.set_option("output_flag", true);
            model.set_option("log_to_console", true);
            model.set_option("log_dev_level", 2);
        }

        // Possible options to configure highs
        // model.set_option("parallel", "on");
        // model.set_option("solver", "simplex");

        let solved = model.solve();
        let val = solved.status();
        match val {
            HighsModelStatus::NotSet
            | HighsModelStatus::LoadError
            | HighsModelStatus::ModelError
            | HighsModelStatus::PresolveError
            | HighsModelStatus::SolveError
            | HighsModelStatus::PostsolveError
            | HighsModelStatus::ReachedTimeLimit
            | HighsModelStatus::ReachedIterationLimit
            | HighsModelStatus::Unknown
            | HighsModelStatus::ModelEmpty
            | HighsModelStatus::ObjectiveBound
            | HighsModelStatus::ObjectiveTarget => {
                PolytopeStatus::Error(format!("Error {:?}", val))
            }
            HighsModelStatus::Infeasible => PolytopeStatus::Infeasible,
            HighsModelStatus::UnboundedOrInfeasible | HighsModelStatus::Unbounded => {
                PolytopeStatus::Unbounded
            }
            HighsModelStatus::Optimal => {
                //TODO optimize the conversion if necessary
                let solution_vals = solved.get_solution();
                let solution_f64 = Array1::from_iter(solution_vals.columns().iter().copied());
                PolytopeStatus::Optimal(solution_f64)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::polyhedron::Polytope;
    use crate::linalg::polyhedron::PolytopeStatus;
    use crate::poly;
    use approx::assert_relative_eq;

    #[cfg(feature = "highs")]
    use highs::{HighsModelStatus, RowProblem, Sense};

    use ndarray::{arr1, arr2, array, s, Array1};

    #[cfg(feature = "minilp")]
    fn init_logger() {
        // minilp has a bug if logging is enabled
    }

    #[cfg(feature = "highs")]
    fn init_logger() {
        match fast_log::init(fast_log::Config::new().console().chan_len(Some(100000))) {
            Ok(_) => (),
            Err(err) => println!("Error occurred while configuring logger: {:?}", err),
        }
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

    #[cfg(feature = "highs")]
    #[test]
    pub fn test_highs() {
        init_logger();

        let mut pb = RowProblem::default();
        let x = pb.add_column(0., f64::NEG_INFINITY..f64::INFINITY);
        let y = pb.add_column(0., f64::NEG_INFINITY..f64::INFINITY);
        let z = pb.add_column(0., f64::NEG_INFINITY..f64::INFINITY);

        pb.add_row(..=-6, &[(x, 5.), (y, 1.), (z, 6.)]);
        pb.add_row(..=2, &[(x, -0.), (y, -1.), (z, 0.)]);
        pb.add_row(..=-20, &[(x, -6.), (y, 2.), (z, 20.)]);

        let mut model = pb.optimise(Sense::Maximise);
        model.set_option("presolve", "off");
        let solved = model.solve();
        let status = solved.status();
        let solution = solved.get_solution();
        println!("{:?}, {:?}", status, solution.columns());
        assert!(status != HighsModelStatus::Unknown);
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

        assert!(matches!(
            poly.solve_linprog(Array1::zeros(2), false),
            PolytopeStatus::Optimal(_)
        ));
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

        assert!(matches!(
            poly.solve_linprog(Array1::zeros(2), false),
            PolytopeStatus::Optimal(_)
        ));
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

        assert!(matches!(
            poly.solve_linprog(Array1::zeros(2), false),
            PolytopeStatus::Optimal(_)
        ));
    }

    // The following two tests check whether the HIGHS configuration fails with simple linear programs

    #[cfg(feature = "highs")]
    #[test]
    pub fn test_row_problem() {
        init_logger();

        use highs::*;
        let mut pb = RowProblem::new();

        let x = pb.add_column(3., ..6);
        let y = pb.add_column(-2., 5..);
        pb.add_row(2.., &[(x, 3.), (y, 8.)]); // 2 <= x*3 + y*8
        pb.add_row(..3, &[(x, 0.), (y, -2.)]);
        pb.add_row(..2, &[(y, 0.), (x, 1.)]);
        pb.add_row(..6, &[(x, 0.), (y, -4.)]);

        print!("{pb:?}");

        let mut model = pb.optimise(Sense::Minimise);
        model.set_option("threads", 1);
        model.set_option("presolve", "off");

        let solved = model.solve();

        print!("{solved:?}");
    }

    #[test]
    pub fn test_polytope_status_0() {
        init_logger();

        let weights = arr2(&[
            [
                0.30361074209213257,
                -0.4362505376338959,
                0.47955194115638733,
                0.17859648168087006,
            ],
            [
                -0.609990656375885,
                -0.4114791750907898,
                0.7140181064605713,
                0.6034472584724426,
            ],
            [
                -0.6196367144584656,
                0.3565647304058075,
                -0.06185908988118172,
                -0.6381561160087585,
            ],
            [
                0.4521157741546631,
                -0.46737807989120483,
                0.1406061202287674,
                0.5742049813270569,
            ],
            [
                -0.26851293444633484,
                0.278455525636673,
                -0.6617708802223206,
                0.12146630883216858,
            ],
            [
                0.2860890030860901,
                -0.3795221745967865,
                0.2328789383172989,
                -0.4218177795410156,
            ],
            [
                0.15694883465766907,
                0.43815314769744873,
                0.19395361840724945,
                -1.0046908855438232,
            ],
        ]);
        let bias = arr1(&[
            0.009999999776482582,
            0.21777614951133728,
            -0.09026645123958588,
            0.009999999776482582,
            0.05117252469062805,
            0.009999999776482582,
            -0.1,
        ]);

        let poly = Polytope::from_mats(weights, bias);
        assert!(!matches!(poly.status(), PolytopeStatus::Infeasible));
    }

    #[test]
    #[ignore = "error in rust version of highs solver"]
    pub fn test_polytope_status_1() {
        init_logger();

        let weights = arr2(&[
            [
                0.30361074209213257,
                -0.4362505376338959,
                0.47955194115638733,
                0.17859648168087006,
            ],
            [
                -0.609990656375885,
                -0.4114791750907898,
                0.7140181064605713,
                0.6034472584724426,
            ],
            [
                -0.6196367144584656,
                0.3565647304058075,
                -0.06185908988118172,
                -0.6381561160087585,
            ],
            [
                0.4521157741546631,
                -0.46737807989120483,
                0.1406061202287674,
                0.5742049813270569,
            ],
            [
                -0.26851293444633484,
                0.278455525636673,
                -0.6617708802223206,
                0.12146630883216858,
            ],
            [
                0.2860890030860901,
                -0.3795221745967865,
                0.2328789383172989,
                -0.4218177795410156,
            ],
            [
                0.15694883465766907,
                0.43815314769744873,
                0.19395361840724945,
                -1.0046908855438232,
            ],
        ]);
        let bias = arr1(&[
            0.009999999776482582,
            0.21777614951133728,
            -0.09026645123958588,
            0.009999999776482582,
            0.05117252469062805,
            0.009999999776482582,
            -0.1943315714597702,
        ]);

        let poly = Polytope::from_mats(weights, bias);
        assert!(!matches!(poly.status(), PolytopeStatus::Infeasible));
    }

    #[test]
    pub fn test_polytope_status_2() {
        init_logger();

        let weights = arr2(&[
            [
                0.30361074209213257,
                -0.4362505376338959,
                0.47955194115638733,
                0.17859648168087006,
            ],
            [
                -0.609990656375885,
                -0.4114791750907898,
                0.7140181064605713,
                0.6034472584724426,
            ],
            [
                -0.6196367144584656,
                0.3565647304058075,
                -0.06185908988118172,
                -0.6381561160087585,
            ],
            [
                -0.4521157741546631,
                0.46737807989120483,
                -0.1406061202287674,
                -0.5742049813270569,
            ],
            [
                0.26851293444633484,
                -0.278455525636673,
                0.6617708802223206,
                -0.12146630883216858,
            ],
            [
                -0.2860890030860901,
                0.3795221745967865,
                -0.2328789383172989,
                0.4218177795410156,
            ],
            [
                0.15694883465766907,
                0.43815314769744873,
                0.19395361840724945,
                -1.0046908855438232,
            ],
            [
                -0.12349238991737366,
                0.0022206550929695368,
                0.5541185140609741,
                -0.5391226410865784,
            ],
            [
                -0.07354876399040222,
                0.6130779385566711,
                -0.5336642265319824,
                0.5246383547782898,
            ],
            [
                0.8576876672897056,
                -0.26357735880021815,
                -0.15968780062118348,
                0.5470196019665636,
            ],
        ]);
        let bias = arr1(&[
            0.009999999776482582,
            0.21777614951133728,
            -0.09026645123958588,
            -0.009999999776482582,
            -0.05117252469062805,
            -0.009999999776482582,
            -0.1943315714597702,
            -0.17030474543571472,
            -0.2569865584373474,
            -0.25349966740775787,
        ]);

        let poly = Polytope::from_mats(weights, bias);
        assert!(matches!(poly.status(), PolytopeStatus::Infeasible));
    }

    #[test]
    pub fn test_chebyshev_box() {
        let poly = Polytope::from_mats(
            array![[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
            array![1., 1., 1., 1.],
        );

        let (p, c) = poly.chebyshev_center();

        let sol = p.solve_linprog(c, false);

        let wit = match sol {
            PolytopeStatus::Optimal(wit) => wit,
            _ => panic!(),
        };

        assert_relative_eq!(
            wit,
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

        let sol = p.solve_linprog(c, false);

        let wit = match sol {
            PolytopeStatus::Optimal(wit) => wit,
            _ => panic!(),
        };

        assert!(0.0 <= wit[0]);
        assert!(wit[0] <= 1.0);
        assert_eq!(wit[1], 0.0);
        assert_eq!(wit[2], 1.0);
    }

    #[test]
    pub fn test_chebyshev_triangle() {
        let poly = Polytope::from_mats(array![[1., 1.], [-1., 1.], [0., -1.]], array![0., 0., 2.4]);

        let (p, c) = poly.chebyshev_center();

        let sol = p.solve_linprog(c, false);

        let wit = match sol {
            PolytopeStatus::Optimal(wit) => wit,
            _ => panic!(),
        };

        assert_relative_eq!(
            wit,
            array![0., -1.414, 1.],
            epsilon = 1e-08,
            max_relative = 1e-02
        );
    }
}
