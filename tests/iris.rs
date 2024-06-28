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

/// Equivalence test based on iris.
///
/// In this test case a DNN trained on the iris dataset is loaded
/// and distilled into an AffTree, which is then tested against a
/// sequential evaluation of the DNN layers. Also the frequency with
/// which the terminals are reached are logged.
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;

    use affinitree::distill::builder::{afftree_from_layers, read_layers, Layer};
    use approx::assert_relative_eq;
    use itertools::Itertools;
    use ndarray::{Array, Array1};
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;

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
    pub fn test_equivalence_iris_dnn() {
        let layers = read_layers(&Path::new("tests/iris_44.npz")).unwrap();

        let dd = afftree_from_layers(4, &layers, None);

        let normal = Normal::new(0., 1.).unwrap();
        let mut r = rand::rngs::OsRng;

        let mut map = BTreeMap::<String, i32>::new();

        for _idx in 0..10000 {
            let x = Array::random_using(4, normal, &mut r);

            let gt = eval_layers(layers.as_slice(), &x);
            let out = dd.find_terminal(dd.tree.get_root(), &x).unwrap();

            let key: String = out.1.iter().map(|x| x.to_string()).join(",");
            if let Some(count) = map.get_mut(&key) {
                *count += 1;
            } else {
                map.insert(key, 1);
            }

            let val = out.0.value.aff.apply(&x);

            assert_relative_eq!(gt, val, epsilon = 1e-08, max_relative = 1e-05);
        }

        // visualize distribution
        for (key, value) in map.iter() {
            println!("{}: {}", key, value)
        }
    }
}
