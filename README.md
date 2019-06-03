# `L-BFGS` [![Build Status](https://travis-ci.org/korken89/lbfgs-rs.svg?branch=master)](https://travis-ci.org/korken89/lbfgs-rs)

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
