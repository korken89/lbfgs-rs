# `L-BFGS` 

<a href="https://github.com/korken89/lbfgs-rs/actions?query=branch%3Amaster">
    <img alt="GHA continuous integration" src="https://github.com/korken89/lbfgs-rs/actions/workflows/rust_unit_tests.yml/badge.svg">
</a>

`L-BFGS` ([Low-memory Broyden–Fletcher–Goldfarb–Shanno](https://en.wikipedia.org/wiki/Limited-memory_BFGS)) is a library for doing
estimation and application of Hessians in numerical optimization while using
limited memory and never explicitly creating the Hessian. Only simple vector
operation are used, as specified by the L-BFGS algorithm.

The specific L-BFGS algorithm implemented here can be found in
[Algorithm 9.1 (L-BFGS two-loop recursion)](http://www.bioinfo.org.cn/~wangchao/maa/Numerical_Optimization.pdf).

Moreover, the condition for the Cautious-BFGS (C-BFGS) algorithm, specified in
[D.-H. Li and M. Fukushima, "On the global convergence of the BFGS method for
nonconvex unconstrained optimization problems"](https://pdfs.semanticscholar.org/5b90/45b7d27a53b1e3c3b3f0dc6aab908cc3e0b2.pdf),
is used to check the updates of the L-BFGS.

## Example

It is straightforward: you just need to create an LBFGS object like this...


```rust
// Problem size and the number of stored vectors in L-BFGS cannot be zero
let problem_size = 3;
let lbfgs_memory_size = 5;

// Create the L-BFGS instance with curvature and C-BFGS checks enabled
let mut lbfgs = Lbfgs::<f64>::new(problem_size, lbfgs_memory_size)
    .with_sy_epsilon(1e-8)     // L-BFGS acceptance condition on s'*y > sy_espsilon
    .with_cbfgs_alpha(1.0)     // C-BFGS condition:
    .with_cbfgs_epsilon(1e-4); // y'*s/||s||^2 > epsilon * ||grad(x)||^alpha
```

Then you can update the LBFGS cache as follows

```rust
// A proper update will be accepted returning `UpdateOk`
assert_eq!(
    lbfgs.update_hessian(&[-0.5, 0.6, -1.2], &[0.1, 0.2, -0.3]),
    UpdateStatus::UpdateOk
);
```

Lastly, you can apply the LFGS Hessian approximation to a vector as follows

```
let mut g = [-3.1, 1.5, 2.1];
lbfgs.apply_hessian(&mut g);
```

Note that this will update `g` in-place.

### Details

You can use either `f64` of `f32` num types.
If you write 
`Lbfgs::new(problem_size, lbfgs_memory_size)`, 
`f64` is implied.

The first direction provided is always accepted, but 
subsequent directions can be rejected either because 
the CBFGS condition is not satisfied or because the 
curvature condition is not satisfied.
Here is an example:

```rust
// Starting value is always accepted (no s or y vectors yet)
assert_eq!(
    lbfgs.update_hessian(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]),
    UpdateStatus::UpdateOk
);

// Rejected because of CBFGS condition
assert_eq!(
    lbfgs.update_hessian(&[-0.838, 0.260, 0.479], &[-0.5, 0.6, -1.2]),
    UpdateStatus::Rejection
);

// This will fail because y'*s == 0 (curvature condition)
assert_eq!(
    lbfgs.update_hessian(
        &[-0.5, 0.6, -1.2],
        &[0.419058177461747, 0.869843029576958, 0.260313940846084]
    ),
    UpdateStatus::Rejection
);
```


## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
