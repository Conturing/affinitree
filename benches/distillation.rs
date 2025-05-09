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

use std::path::Path;

use affinitree::aff;
use affinitree::distill::builder::{afftree_from_layers, read_layers};
use affinitree::distill::schema;
use affinitree::linalg::affine::{AffFunc, Polytope};
use affinitree::pwl::afftree::AffTree;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::{arr1, arr2};

pub fn dd_apply_function(c: &mut Criterion) {
    let mut group = c.benchmark_group("dd-apply");
    group.sample_size(500);

    let dd = AffTree::<2>::new(5);
    let h = AffFunc::from_mats(
        arr2(&[
            [-1., 2., 3., 4., 5.],
            [2., -1., 3., -7., 9.],
            [0., 0., 3., 1., -2.],
            [1., -1., 1., -2., 2.],
            [0., 1., 0., 1., 0.],
        ]),
        arr1(&[-3., 5., -7., 1., -2.]),
    );

    group.bench_function("apply function", |b| {
        b.iter(|| dd.clone().apply_func(black_box(&h)))
    });
}

pub fn dd_add_node_function(c: &mut Criterion) {
    let mut group = c.benchmark_group("dd-add-node");
    group.sample_size(500);

    let h = AffFunc::from_mats(
        arr2(&[
            [-1., 2., 3., 4., 5.],
            [2., -1., 3., -7., 9.],
            [0., 0., 3., 1., -2.],
            [1., -1., 1., -2., 2.],
            [0., 1., 0., 1., 0.],
        ]),
        arr1(&[-3., 5., -7., 1., -2.]),
    );
    let dd = default_dd(h.to_owned(), 5, 4);

    group.bench_function("add node", |b| {
        b.iter(|| {
            dd.clone()
                .add_child_node(black_box(12894), black_box(0), black_box(h.to_owned()))
        })
    });
}

pub fn dd_is_feasible(c: &mut Criterion) {
    let mut group = c.benchmark_group("dd-feasible");
    group.sample_size(500);

    let mut dd = AffTree::<2>::from_aff(aff!([[2., 1.]] + [-1.]));

    dd.add_child_node(0, 1, aff!([[1., 2.]] + [-1.5])).unwrap();
    dd.add_child_node(1, 1, aff!([[0.5, 5.]] + [1.0])).unwrap();
    dd.add_child_node(2, 1, aff!([[3., -1.]] + [0.])).unwrap();
    dd.add_child_node(3, 1, aff!([[-1., -1.]] + [6.])).unwrap();
    dd.add_child_node(4, 0, aff!([[-1., 7.]] + [4.])).unwrap();
    dd.add_child_node(5, 1, aff!([[-2., -0.2]] + [3.])).unwrap();
    // feasible
    dd.add_child_node(6, 0, aff!([[0., 0.]] + [1.])).unwrap();
    // infeasible
    dd.add_child_node(6, 1, aff!([[0., 0.]] + [0.])).unwrap();

    group.bench_function("is_edge_feasible (calc)", |b| {
        b.iter(|| dd.clone().is_edge_feasible(black_box(6), black_box(8)))
    });

    dd.is_edge_feasible(6, 8);

    group.bench_function("is_edge_feasible (cache)", |b| {
        b.iter(|| dd.clone().is_edge_feasible(black_box(6), black_box(8)))
    });
}

#[inline]
fn default_dd(h: AffFunc, dim: usize, depth: i32) -> AffTree<2> {
    assert!(depth > 0);
    let mut relu_dd = schema::partial_ReLU(4, 0);
    for i in 1..dim {
        relu_dd.compose::<false, false>(&schema::partial_ReLU(4, i));
    }
    let mut dd = AffTree::<2>::from_aff(h.clone());
    for _ in 0..depth {
        dd.compose::<true, false>(&relu_dd);
        dd.apply_func(&h);
    }
    dd.compose::<true, false>(&relu_dd);
    dd
}

pub fn default_dd_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample-size-example");
    group.sample_size(25);

    let h = AffFunc::from_mats(
        arr2(&[[-1., 2., 3.], [2., -1., 3.], [3., 1., -2.]]),
        arr1(&[-3., 1., -2.]),
    );
    group.bench_function("default dd 2", |b| {
        b.iter(|| default_dd(black_box(h.clone()), black_box(3), black_box(2)))
    });
    group.bench_function("default dd 4", |b| {
        b.iter(|| default_dd(black_box(h.clone()), black_box(3), black_box(4)))
    });
    group.bench_function("default dd 6", |b| {
        b.iter(|| default_dd(black_box(h.clone()), black_box(3), black_box(6)))
    });
}

pub fn ecoli_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("net-to-dd-ecoli");
    group.sample_size(100);

    let layers = read_layers(&"res/nn/ecoli.npz").unwrap();
    group.bench_function("ecoli [7-5-5-4]", |b| {
        b.iter(|| afftree_from_layers(7, &layers, None))
    });
}

pub fn iris_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("net-to-dd-iris");
    group.sample_size(10);

    let layers = read_layers(&"res/nn/iris.npz").unwrap();
    group.bench_function("iris [4-10-8-6-4-3]", |b| {
        b.iter(|| afftree_from_layers(4, &layers, None))
    });
}

pub fn mnist_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("net-to-dd-mnist");
    group.sample_size(10);

    let layers = read_layers(&"res/nn/mnist-5-5.npz").unwrap();
    group.bench_function("mnist [7-5-5-5-5-10]", |b| {
        b.iter(|| afftree_from_layers(7, &layers, None))
    });
}

pub fn ecoli_argmax_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("compose-argmax-ecoli");
    group.sample_size(50);

    let layers = read_layers(&"res/nn/ecoli.npz").unwrap();
    let dd = afftree_from_layers(7, &layers, None);
    let argmax = schema::argmax(4);
    group.bench_function("ecoli argmax", |b| {
        b.iter(|| dd.clone().compose::<true, false>(&argmax))
    });
}

pub fn ecoli_relu_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("compose-relu-ecoli");
    group.sample_size(50);

    let layers = read_layers(&"res/nn/ecoli.npz").unwrap();
    let dd = afftree_from_layers(7, &layers, None);
    let mut relu = schema::partial_ReLU(4, 0);
    relu.compose::<false, false>(&schema::partial_ReLU(4, 1));
    relu.compose::<false, false>(&schema::partial_ReLU(4, 2));
    relu.compose::<false, false>(&schema::partial_ReLU(4, 3));
    group.bench_function("ecoli relu", |b| {
        b.iter(|| dd.clone().compose::<true, false>(&relu))
    });
}

pub fn compose_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("compose-benchmark");
    group.sample_size(10);

    let h = AffFunc::from_mats(
        arr2(&[[-1., 2., 3.], [2., -1., 3.], [3., 1., -2.]]),
        arr1(&[-3., 1., -2.]),
    );
    let dd = default_dd(h.clone(), 3, 4);
    let dd2 = default_dd(h, 3, 4);
    group.bench_function("compose bench", |b| {
        b.iter(|| dd.clone().compose::<true, false>(&dd2))
    });
}

pub fn infeasible_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("infeasible-mnist-60");
    group.sample_size(10);

    let poly = Polytope::hypercube(60, 0.03);
    let precondition = AffTree::<2>::from_poly(poly, AffFunc::identity(60), None).unwrap();
    let layers = read_layers(&Path::new("tests/mnist_60-4x10.npz")).unwrap();

    group.bench_function("mnist [60-10-10-10-10]", |b| {
        b.iter(|| afftree_from_layers(60, &layers, Some(precondition.clone())))
    });
}

criterion_group!(
    benches,
    dd_apply_function,
    default_dd_benchmark,
    ecoli_benchmark,
    iris_benchmark,
    mnist_benchmark,
    ecoli_argmax_benchmark,
    ecoli_relu_benchmark,
    compose_benchmark,
    dd_add_node_function,
    dd_is_feasible,
    infeasible_benchmark
);
criterion_main!(benches);
