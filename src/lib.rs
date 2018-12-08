//! # lbfgs
//!
//! The `L-BFGS`
//!
//! # Examples
//!
//! ```
//! fn main() {
//! }
//! ```
//!
//! # Errors
//!
//!
//! # Panics
//!
//!

extern crate num;
use std::num::NonZeroUsize;

pub mod vec_ops;

#[cfg(test)]
mod tests;

const SY_EPSILON: f64 = 1e-10;

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
pub struct Lbfgs {
    /// The number of vectors in s and y that are currently in use
    active_size: usize,
    /// Used to warm-start the Hessian estimation with H_0 = gamma * I
    gamma: f64,
    /// s holds the vectors of state difference s_k = x_{k+1} - x_k, s_0 holds the most recent s
    s: Vec<Vec<f64>>,
    /// y holds the vectors of the function g (usually cost function gradient) difference:
    /// y_k = g_{k+1} - g_k, y_0 holds the most recent y
    y: Vec<Vec<f64>>,
    /// Intermediary storage for the forward L-BFGS pass
    alpha: Vec<f64>,
    /// Intermediary storage for the forward L-BFGS pass
    rho: Vec<f64>,
    /// The alpha parameter of the C-BFGS criterion
    cbfgs_alpha: f64,
    /// The epsilon parameter of the C-BFGS criterion
    cbfgs_epsilon: f64,
    /// limit on the inner product s'*y for acceptance in the buffer
    sy_epsilon: f64,
    /// Holds the state of the last `update_hessian`, used to calculate the `s_k` vectors
    old_state: Vec<f64>,
    /// Holds the g of the last `update_hessian`, used to calculate the `y_k` vectors
    old_g: Vec<f64>,
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

impl Lbfgs {
    /// Create a new L-BFGS instance with a specific problem and L-BFGS buffer size
    ///
    pub fn new(problem_size: NonZeroUsize, buffer_size: NonZeroUsize) -> Lbfgs {
        let problem_size = problem_size.get();
        let buffer_size = buffer_size.get();

        Lbfgs {
            active_size: 0,
            gamma: 1.0,
            s: vec![vec![0.0; problem_size]; buffer_size + 1], // +1 for the temporary checking area
            y: vec![vec![0.0; problem_size]; buffer_size + 1], // +1 for the temporary checking area
            alpha: vec![0.0; buffer_size],
            rho: vec![0.0; buffer_size + 1],
            cbfgs_alpha: 0.0,
            cbfgs_epsilon: 0.0,
            sy_epsilon: SY_EPSILON,
            old_state: vec![0.0; problem_size],
            old_g: vec![0.0; problem_size],
            first_old: true,
        }
    }

    /// Update the default C-BFGS alpha
    pub fn with_cbfgs_alpha(mut self, alpha: f64) -> Self {
        assert!(alpha >= 0.0);

        self.cbfgs_alpha = alpha;
        self
    }

    /// Update the default C-BFGS epsilon
    pub fn with_cbfgs_epsilon(mut self, epsilon: f64) -> Self {
        assert!(epsilon >= 0.0);

        self.cbfgs_epsilon = epsilon;
        self
    }

    /// Update the default sy_epsilon
    pub fn with_sy_epsilon(mut self, sy_epsilon: f64) -> Self {
        assert!(sy_epsilon >= 0.0);

        self.sy_epsilon = sy_epsilon;
        self
    }

    /// Apply the current Hessian estimate to an input vector
    pub fn apply_hessian(&mut self, g: &mut [f64]) {
        assert!(g.len() == self.old_g.len());

        if self.active_size == 0 {
            // No Hessian available, the g is the best we can do for now
            return;
        }

        let active_s = &self.s[0..self.active_size];
        let active_y = &self.y[0..self.active_size];
        let rho = &mut self.rho[0..self.active_size];
        let alpha = &mut self.alpha;

        let q = g;

        // Perform the forward L-BFGS algorithm
        for (s_k, (y_k, (rho_k, alpha_k))) in active_s
            .iter()
            .zip(active_y.iter().zip(rho.iter().zip(alpha.iter_mut())))
        {
            let a = rho_k * vec_ops::inner_product(s_k, q);

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
            let beta = rho_k * vec_ops::inner_product(y_k, r);

            vec_ops::inplace_vec_add(r, s_k, alpha_k - beta);
        }

        // The g with the Hessian applied is available in the input g
        // r = H_k * grad f
    }

    /// Check the validity of the newly added s and y vectors. Based on the condition in:
    /// D.-H. Li and M. Fukushima, "On the global convergence of the BFGS method for nonconvex
    /// unconstrained optimization problems," vol. 11, no. 4, pp. 1054–1064, jan 2001.
    fn new_s_and_y_valid(&mut self, g: &[f64]) -> bool {
        let s = self.s.last().unwrap();
        let y = self.y.last().unwrap();
        let rho = self.rho.last_mut().unwrap();
        let ys = vec_ops::inner_product(s, y);
        let norm_s_squared = vec_ops::inner_product(s, s);

        *rho = 1.0 / ys;

        if norm_s_squared <= std::f64::MIN_POSITIVE
            || (self.sy_epsilon > 0.0 && ys <= self.sy_epsilon)
        {
            // In classic L-BFGS, the buffer should be updated only if
            // y'*s is strictly positive and |s| is nonzero
            false
        } else if self.cbfgs_epsilon > 0.0 && self.cbfgs_alpha > 0.0 {
            // Check the CBFGS condition of Li and Fukushima
            // Condition: (y^T * s) / ||s||^2 > epsilon * ||grad(x)||^alpha
            let lhs_cbfgs = ys / norm_s_squared;
            let rhs_cbfgs = self.cbfgs_epsilon * vec_ops::norm2(g).powf(self.cbfgs_alpha);
            lhs_cbfgs > rhs_cbfgs && lhs_cbfgs.is_finite() && rhs_cbfgs.is_finite()
        } else {
            // The standard L-BFGS conditions are satisfied and C-BFGS is
            // not active (either cbfgs_epsilon <= 0.0 or cbfgs_alpha <= 0.0)
            true
        }
    }

    /// Saves vectors to update the Hessian estimate
    pub fn update_hessian(&mut self, g: &[f64], state: &[f64]) -> UpdateStatus {
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
        self.gamma = (1.0 / self.rho[0]) / vec_ops::inner_product(&self.y[0], &self.y[0]);

        // Update the indexes and number of active, -1 comes from the temporary area used in
        // the end of s and y to check if they are valid
        self.active_size = (self.s.len() - 1).min(self.active_size + 1);

        UpdateStatus::UpdateOk
    }
}
