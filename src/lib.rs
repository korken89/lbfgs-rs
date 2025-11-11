//! # lbfgs
//! ```
//! use lbfgs::*;
//!
//! fn main() {
//!     // Problem size and the number of stored vectors in L-BFGS cannot be zero
//!     let problem_size = 3;
//!     let lbfgs_memory_size = 5;
//!
//!     // Create the L-BFGS instance with curvature and C-BFGS checks enabled
//!     let mut lbfgs = Lbfgs::<f64>::new(problem_size, lbfgs_memory_size)
//!         .with_sy_epsilon(1e-8)     // L-BFGS acceptance condition on s'*y > sy_espsilon
//!         .with_cbfgs_alpha(1.0)     // C-BFGS condition:
//!         .with_cbfgs_epsilon(1e-4); // y'*s/||s||^2 > epsilon * ||grad(x)||^alpha
//!
//!     // Starting value is always accepted (no s or y vectors yet)
//!     assert_eq!(
//!         lbfgs.update_hessian(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]),
//!         UpdateStatus::UpdateOk
//!     );
//!
//!     // Rejected because of CBFGS condition
//!     assert_eq!(
//!         lbfgs.update_hessian(&[-0.838, 0.260, 0.479], &[-0.5, 0.6, -1.2]),
//!         UpdateStatus::Rejection
//!     );
//!
//!     // This will fail because y'*s == 0 (curvature condition)
//!     assert_eq!(
//!         lbfgs.update_hessian(
//!             &[-0.5, 0.6, -1.2],
//!             &[0.419058177461747, 0.869843029576958, 0.260313940846084]
//!         ),
//!         UpdateStatus::Rejection
//!     );
//!
//!     // A proper update that will be accepted
//!     assert_eq!(
//!         lbfgs.update_hessian(&[-0.5, 0.6, -1.2], &[0.1, 0.2, -0.3]),
//!         UpdateStatus::UpdateOk
//!     );
//!
//!     // Apply Hessian approximation on a gradient
//!     let mut g = [-3.1, 1.5, 2.1];
//!     let correct_dir = [-1.100601247872944, -0.086568349404424, 0.948633011911515];
//!
//!     lbfgs.apply_hessian(&mut g);
//!
//!     assert!((g[0] - correct_dir[0]).abs() < 1e-12);
//!     assert!((g[1] - correct_dir[1]).abs() < 1e-12);
//!     assert!((g[2] - correct_dir[2]).abs() < 1e-12);
//! }
//! ```
//!
//! # Errors
//!
//! `update_hessian` will give errors if the C-BFGS or L-BFGS curvature conditions are not met.
//!
//! # Panics
//!
//! `with_sy_epsilon`, `with_cbfgs_alpha`, and `with_cbfgs_epsilon` will panic if given negative
//! values.
//!
//! `update_hessian` and `apply_hessian` will panic if given slices which are not the same length
//! as the `problem_size`.
//!

use num_traits::Float;

pub mod vec_ops;

#[cfg(test)]
mod tests;

/// Precision is a trait extending `num_traits::Float` to provide type-specific constants
/// (the default sy tolerance).
pub trait LbfgsPrecision: Float {
    const DEFAULT_SY_TOLERANCE: Self;
    const ABS_TOL: Self;
    const REL_TOL: Self;
}

impl LbfgsPrecision for f64 {
    const DEFAULT_SY_TOLERANCE: f64 = 1e-10;
    const ABS_TOL: f64 = 1e-8;
    const REL_TOL: f64 = 1e-10;
}

impl LbfgsPrecision for f32 {
    const DEFAULT_SY_TOLERANCE: f32 = 1e-8;
    const ABS_TOL: f32 = 1e-6;
    const REL_TOL: f32 = 1e-6;
}


/// LBFGS Buffer
///
/// The Limited-memory BFGS algorithm is used to estimate curvature information for the
/// gradient of a function as well as other operators and is often used in numerical
/// optimization and numerical methods in general.
///
/// `Lbfgs` maintains a buffer of pairs `(s,y)` and values `rho` (inverse of inner products
/// of `s` and `y`)
///
///
#[derive(Debug)]
pub struct Lbfgs<T = f64>
where
    T: LbfgsPrecision + std::iter::Sum<T>,
{
    /// The number of vectors in s and y that are currently in use
    active_size: usize,
    /// Used to warm-start the Hessian estimation with H_0 = gamma * I
    gamma: T,
    /// s holds the vectors of state difference s_k = x_{k+1} - x_k, s_0 holds the most recent s
    s: Vec<Vec<T>>,
    /// y holds the vectors of the function g (usually cost function gradient) difference:
    /// y_k = g_{k+1} - g_k, y_0 holds the most recent y
    y: Vec<Vec<T>>,
    /// Intermediary storage for the forward L-BFGS pass
    alpha: Vec<T>,
    /// Intermediary storage for the forward L-BFGS pass
    rho: Vec<T>,
    /// The alpha parameter of the C-BFGS criterion
    cbfgs_alpha: T,
    /// The epsilon parameter of the C-BFGS criterion
    cbfgs_epsilon: T,
    /// Limit on the inner product s'*y for acceptance in the buffer
    sy_epsilon: T,
    /// Holds the state of the last `update_hessian`, used to calculate the `s_k` vectors
    old_state: Vec<T>,
    /// Holds the g of the last `update_hessian`, used to calculate the `y_k` vectors
    old_g: Vec<T>,
    /// Check to see if the `old_*` variables have valid data
    first_old: bool,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum UpdateStatus {
    /// The g and state was accepted to update the Hessian estimate
    UpdateOk,
    /// The g and state was rejected by the C-BFGS criteria
    Rejection,
}

impl<T> Lbfgs<T>
where
    T: LbfgsPrecision + std::iter::Sum<T> ,
{
    /// Create a new L-BFGS instance with a specific problem and L-BFGS buffer size
    pub fn new(problem_size: usize, buffer_size: usize) -> Lbfgs<T> {
        assert!(problem_size > 0);
        assert!(buffer_size > 0);

        Lbfgs {
            active_size: 0,
            gamma: T::one(),
            s: vec![vec![T::zero(); problem_size]; buffer_size + 1], // +1 for the temporary checking area
            y: vec![vec![T::zero(); problem_size]; buffer_size + 1], // +1 for the temporary checking area
            alpha: vec![T::zero(); buffer_size],
            rho: vec![T::zero(); buffer_size + 1],
            cbfgs_alpha: T::zero(),
            cbfgs_epsilon: T::zero(),
            sy_epsilon: T::DEFAULT_SY_TOLERANCE,
            old_state: vec![T::zero(); problem_size],
            old_g: vec![T::zero(); problem_size],
            first_old: true,
        }
    }

    /// Update the default C-BFGS alpha
    pub fn with_cbfgs_alpha(mut self, alpha: T) -> Self {
        assert!(alpha >= T::zero(), "Negative alpha");

        self.cbfgs_alpha = alpha;
        self
    }

    /// Update the default C-BFGS epsilon
    pub fn with_cbfgs_epsilon(mut self, epsilon: T) -> Self {
        assert!(epsilon >= T::zero(), "sy_epsilon must be non-negative");

        self.cbfgs_epsilon = epsilon;
        self
    }

    /// Update the default sy_epsilon
    pub fn with_sy_epsilon(mut self, sy_epsilon: T) -> Self {
        assert!(sy_epsilon >= T::zero(), "sy_epsilon must be non-negative");

        self.sy_epsilon = sy_epsilon;
        self
    }

    /// "Empties" the buffer
    ///
    /// This is a cheap operation as it amount to setting certain internal flags
    pub fn reset(&mut self) {
        self.active_size = 0;
        self.first_old = true;
    }

    /// Apply the current Hessian estimate to an input vector
    pub fn apply_hessian(&mut self, g: &mut [T]) {
        assert!(g.len() == self.old_g.len());

        if self.active_size == 0 {
            // No Hessian available, the g is the best we can do for now
            return;
        }

        let active_s = &self.s[0..self.active_size];
        let active_y = &self.y[0..self.active_size];
        let rho = &self.rho[0..self.active_size];
        let alpha = &mut self.alpha;

        let q = g;

        // Perform the forward L-BFGS algorithm
        for (s_k, (y_k, (rho_k, alpha_k))) in active_s
            .iter()
            .zip(active_y.iter().zip(rho.iter().zip(alpha.iter_mut())))
        {
            let a = *rho_k * vec_ops::inner_product(s_k, q);

            *alpha_k = a;

            vec_ops::inplace_vec_add(q, y_k, -a);
        }

        // Apply the initial Hessian estimate and form r = H_0 * q, where H_0 = gamma * I
        vec_ops::scalar_mult(q, self.gamma);
        let r = q;

        // Perform the backward L-BFGS algorithm
        for (s_k, (y_k, (rho_k, alpha_k))) in active_s
            .iter()
            .zip(active_y.iter().zip(rho.iter().zip(alpha.iter())))
            .rev()
        {
            let beta = *rho_k * vec_ops::inner_product(y_k, r);

            vec_ops::inplace_vec_add(r, s_k, *alpha_k - beta);
        }

        // The g with the Hessian applied is available in the input g
        // r = H_k * grad f
    }

    /// Check the validity of the newly added s and y vectors. Based on the condition in:
    /// D.-H. Li and M. Fukushima, "On the global convergence of the BFGS method for nonconvex
    /// unconstrained optimization problems," vol. 11, no. 4, pp. 1054–1064, jan 2001.
    fn new_s_and_y_valid(&mut self, g: &[T]) -> bool {
        let s = self.s.last().unwrap();
        let y = self.y.last().unwrap();
        let rho = self.rho.last_mut().unwrap();
        let ys = vec_ops::inner_product(s, y);
        let norm_s_squared = vec_ops::inner_product(s, s);

        *rho = T::one() / ys;

        if norm_s_squared <= T::min_positive_value()
            || (self.sy_epsilon > T::zero() && ys <= self.sy_epsilon)
        {
            // In classic L-BFGS, the buffer should be updated only if
            // y'*s is strictly positive and |s| is nonzero
            false
        } else if self.cbfgs_epsilon > T::zero() && self.cbfgs_alpha > T::zero() {
            // Check the CBFGS condition of Li and Fukushima
            // Condition: (y^T * s) / ||s||^2 > epsilon * ||grad(x)||^alpha
            let lhs_cbfgs = ys / norm_s_squared;
            let rhs_cbfgs = self.cbfgs_epsilon * vec_ops::norm2(g).powf(self.cbfgs_alpha);

            lhs_cbfgs > rhs_cbfgs
        } else {
            // The standard L-BFGS conditions are satisfied and C-BFGS is
            // not active (either cbfgs_epsilon <= 0.0 or cbfgs_alpha <= 0.0)
            true
        }
    }

    /// Saves vectors to update the Hessian estimate
    pub fn update_hessian(&mut self, g: &[T], state: &[T]) -> UpdateStatus {
        assert!(g.len() == self.old_state.len());
        assert!(state.len() == self.old_state.len());

        // First iteration, only save
        if self.first_old {
            self.first_old = false;

            self.old_state.copy_from_slice(state);
            self.old_g.copy_from_slice(g);

            return UpdateStatus::UpdateOk;
        }

        // Form the new s_k in the temporary area
        vec_ops::difference_and_save(self.s.last_mut().unwrap(), &state, &self.old_state);

        // Form the new y_k in the temporary area
        vec_ops::difference_and_save(self.y.last_mut().unwrap(), &g, &self.old_g);

        // Check that the s and y are valid to use
        if !self.new_s_and_y_valid(g) {
            return UpdateStatus::Rejection;
        }

        self.old_state.copy_from_slice(state);
        self.old_g.copy_from_slice(g);

        // Move the new s_0,  y_0 and rho_0 to the front
        self.s.rotate_right(1);
        self.y.rotate_right(1);
        self.rho.rotate_right(1);

        // Update the Hessian estimate
        self.gamma = (T::one() / self.rho[0]) / vec_ops::inner_product(&self.y[0], &self.y[0]);

        // Update the indexes and number of active, -1 comes from the temporary area used in
        // the end of s and y to check if they are valid
        self.active_size = (self.s.len() - 1).min(self.active_size + 1);

        UpdateStatus::UpdateOk
    }
}
