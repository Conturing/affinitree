/// Equivalence test based on mnist.
///
/// In this test case a DNN trained on the mnist dataset is loaded
/// and distilled into an AffTree, which is then tested against a
/// sequential evaluation of the DNN layers. Also the frequency with
/// which the terminals are reached are logged.
#[cfg(test)]
mod tests {
    use affinitree::distill::builder::{afftree_from_layers_verbose, read_layers, Layer};
    use affinitree::linalg::affine::Polytope;
    use affinitree::pwl::node::NodeState;
    use approx::assert_relative_eq;

    use ndarray::{Array, Array1};
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use rand::SeedableRng;

    use std::path::Path;

    fn init_logger() {
        use env_logger::Target;
        use log::LevelFilter;

        let _ = env_logger::builder()
            .is_test(true)
            .filter_module("minilp", LevelFilter::Error)
            .target(Target::Stdout)
            .filter_level(LevelFilter::Warn)
            .try_init();
    }

    pub fn eval_layers(layers: &[Layer], input: &Array1<f64>) -> Array1<f64> {
        let mut val = input.to_owned();
        for layer in layers {
            match layer {
                Layer::ReLU(_) => val = val.map(|x| x.max(0.)),
                Layer::Linear(func) => val = func.apply(&val),
                _ => panic!("Unsupported operation"),
            }
        }
        val
    }

    #[test]
    pub fn test_equivalence_mnist_dnn() {
        init_logger();

        println!("Reading network ...");
        let layers = read_layers(&Path::new("tests/mnist_60-4x10.npz")).unwrap();
        println!("{:?}", &layers);

        println!("Building AffTree ...");
        let dd0 = afftree_from_layers_verbose(60, &layers[..11], None);

        println!("dd0 {}", dd0.tree.describe());

        let normal = Normal::new(0., 1.).unwrap();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        println!("Testing equivalence ...");
        for _idx in 0..100000 {
            let x = Array::random_using(60, normal, &mut rng);

            let gt = eval_layers(&layers[0..7], &x);
            let out = dd0
                .evaluate_to_terminal(dd0.tree.get_root(), &x, 512)
                .unwrap();
            let val = out.0.apply(&x);

            assert_relative_eq!(gt, val, epsilon = 1e-08, max_relative = 1e-05);
        }
    }

    #[test]
    pub fn test_node_cache() {
        init_logger();

        println!("Reading network ...");
        let layers = read_layers(&Path::new("tests/mnist_60-4x10.npz")).unwrap();

        println!("Building AffTree ...");
        let dd0 = afftree_from_layers_verbose(60, &layers[..11], None);

        println!("dd0 {}", dd0.tree.describe());

        let mut iter = dd0.polyhedra();

        while let Some((data, ineqs)) = iter.next(&dd0.tree) {
            let poly = Polytope::intersection_n(dd0.in_dim(), ineqs.as_slice());

            if data.index == 0 {
                continue;
            }

            let node = dd0.tree.tree_node(data.index).unwrap();

            match &node.value.state {
                NodeState::FeasibleWitness(wit) => {
                    for point in wit.iter() {
                        assert!(
                            poly.contains(point),
                            "Cached witness does not solve path poly: {:?}",
                            data
                        );
                    }
                }
                NodeState::Indeterminate => {}
                NodeState::Infeasible => panic!("Infeasible state found: {:?}", data),
                NodeState::Feasible => {}
            }
        }
    }

    #[test]
    pub fn test_inherited_solutions() {
        init_logger();

        println!("Reading network ...");
        let layers = read_layers(&Path::new("tests/mnist_60-4x10.npz")).unwrap();

        println!("Building AffTree ...");
        let dd0 = afftree_from_layers_verbose(60, &layers[..15], None);

        println!("dd0 {}", dd0.tree.describe());

        let mut iter = dd0.polyhedra();

        while let Some((data, ineqs)) = iter.next(&dd0.tree) {
            let poly = Polytope::intersection_n(dd0.in_dim(), ineqs.as_slice());

            if data.index == 0 {
                continue;
            }

            let node = dd0.tree.tree_node(data.index).unwrap();

            match &node.value.state {
                NodeState::FeasibleWitness(wit) => {
                    match &dd0
                        .tree
                        .node_value(dd0.tree.parent(data.index).unwrap().source_idx)
                        .unwrap()
                        .state
                    {
                        NodeState::FeasibleWitness(p_wit) => {
                            for point in p_wit {
                                if poly.contains(point) {
                                    if !wit.contains(point) {
                                        println!(
                                            "Parent solution not inherited: {:?} {}",
                                            data,
                                            poly.distance(point)
                                        );
                                    }
                                } else {
                                    if wit.contains(point) {
                                        println!(
                                            "Parent solution incorrectly inherited: {:?}",
                                            data
                                        );
                                    }
                                }
                            }
                        }
                        NodeState::Indeterminate => println!("Parent indeterminate"),
                        NodeState::Infeasible => panic!("Infeasible state found: {:?}", data),
                        NodeState::Feasible => println!("Parent only feasible"),
                    }
                }
                NodeState::Indeterminate => {}
                NodeState::Infeasible => panic!("Infeasible state found: {:?}", data),
                NodeState::Feasible => {}
            }
        }
    }
}
