#[cfg(test)]
mod tests {
    use affinitree::distill::builder::Layer;
    use affinitree::distill::builder::{afftree_from_layers, read_layers};
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
        let layers = read_layers(&Path::new("tests/iris_44.npz")).unwrap();

        let dd = afftree_from_layers(4, &layers);

        let normal = Normal::new(0., 1.).unwrap();
        let mut r = rand::rngs::OsRng;

        let mut map = BTreeMap::<String, i32>::new();

        for _idx in 0..10000 {
            let x = Array::random_using(4, normal, &mut r);

            let gt = eval_layers(layers.as_slice(), &x);
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

        // visualize distribution
        for (key, value) in map.iter() {
            println!("{}: {}", key, value)
        }
    }
}
