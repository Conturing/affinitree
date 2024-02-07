/// Equivalence test based on mnist.
///
/// In this test case a DNN trained on the mnist dataset is loaded
/// and distilled into an AffTree, which is then tested against a
/// sequential evaluation of the DNN layers. Also the frequency with
/// which the terminals are reached are logged.
#[cfg(test)]
mod tests {
    use affinitree::core::builder::Layer;
    use affinitree::core::builder::{afftree_from_layers, read_layers};
    use approx::assert_relative_eq;
    use itertools::Itertools;
    use ndarray::{Array, Array1};
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;

    use std::collections::BTreeMap;
    use std::path::Path;

    pub fn eval_layers(layers: &[Layer], input: &Array1<f64>) -> Array1<f64> {
        let mut val = input.to_owned();
        for layer in layers {
            match layer {
                Layer::ReLU => val = val.map(|x| x.max(0.)),
                Layer::Linear(func) => val = func.apply(&val),
                _ => panic!("Unsupported operation"),
            }
        }
        val
    }

    #[test]
    pub fn test_read_npy() {
        println!("Reading network ...");
        let layers = read_layers(&Path::new("tests/mnist_10-4x10.npz")).unwrap();

        println!("Building AffTree ...");
        let dd = afftree_from_layers(10, &layers[0..3]);

        let normal = Normal::new(0., 1.).unwrap();
        let mut r = rand::rngs::OsRng;

        let mut map = BTreeMap::<String, i32>::new();

        println!("Testing equivalence ...");
        for _idx in 0..10000 {
            let x = Array::random_using(10, normal, &mut r);

            let gt = eval_layers(&layers[0..3], &x);
            let out = dd
                .evaluate_to_terminal(dd.tree.get_root(), &x, 512)
                .unwrap();

            let key: String = out.1.iter().map(|x| x.to_string()).join(",");
            if let Some(count) = map.get_mut(&key) {
                *count += 1;
            } else {
                map.insert(key, 1);
            }

            let val = out.0.apply(&x);

            assert_relative_eq!(gt, val, epsilon = 1e-08, max_relative = 1e-05);
        }
    }
}
