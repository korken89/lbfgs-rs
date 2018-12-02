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

pub mod vec_ops;

#[derive(Debug)]
pub struct Estimator {
    /// The number of vectors in s and y that are currently in use
    active_size: usize,
    /// Used to warm-start the Hessian estimation with H_0 = gamma * I
    gamma: f64,
    /// s holds the vectors of state difference s_k = x_{k+1} - x_k, s_0 holds the most recent s
    s: Vec<Vec<f64>>,
    /// y holds the vectors of the function g (usually cost function gradient) difference:
    /// y_k = g_{k+1} - g_k, y_0 holds the most recent y
    y: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    rho: Vec<f64>,
    old_state: Vec<f64>,
    old_g: Vec<f64>,
}

#[derive(Debug, Copy, Clone)]
pub enum UpdateStatus {
    /// The g and state was accepted to update the Hessian estimate
    UpdateOk,
    /// The g and state was rejected by the C-BFGS criteria
    CBFGSRejection,
}

impl Estimator {
    pub fn new(problem_size: usize, buffer_size: usize) -> Estimator {
        assert!(problem_size > 0); // TODO: Replace this with core::nonzero when stable
        assert!(buffer_size > 0);

        Estimator {
            active_size: 0,
            gamma: 1.0,
            s: vec![vec![1.0; problem_size]; buffer_size + 1], // +1 for the temporary checking area
            y: vec![vec![0.0; problem_size]; buffer_size + 1], // +1 for the temporary checking area
            alpha: vec![0.0; buffer_size],
            rho: vec![0.0; buffer_size],
            old_state: vec![0.0; problem_size],
            old_g: vec![0.0; problem_size],
        }
    }

    /// Apply the current Hessian estimate to an input vector
    pub fn apply_hessian(&mut self, g: &mut [f64]) {
        assert!(g.len() == self.old_g.len());

        if self.active_size == 0 {
            // No Hessian available, the g is the best we can do for now
        } else {
            let active_s = &self.s[0..self.active_size];
            let active_y = &self.y[0..self.active_size];
            let rho = &mut self.rho;
            let alpha = &mut self.alpha;

            let q = g;

            // Perform the forward L-BFGS algorithm
            active_s
                .iter()
                .zip(active_y.iter())
                .enumerate()
                .for_each(|(idx, (s_k, y_k))| {
                    let r = 1.0 / vec_ops::inner_product(s_k, y_k);
                    let a = r * vec_ops::inner_product(s_k, q);

                    rho[idx] = r;
                    alpha[idx] = a;

                    vec_ops::inplace_vec_add(q, y_k, -a);
                });

            // Apply the initial Hessian estimate and form r = H_0 * q, where H_0 = gamma * I
            vec_ops::scalar_mult(q, self.gamma);
            let r = q;

            // Perform the backward L-BFGS algorithm
            active_s
                .iter()
                .rev()
                .zip(active_y.iter().rev())
                .enumerate()
                .rev()
                .for_each(|(idx, (s_k, y_k))| {
                    let beta = rho[idx] * vec_ops::inner_product(y_k, r);
                    vec_ops::inplace_vec_add(r, s_k, alpha[idx] - beta);
                });

            // The g with the Hessian applied is available in the input g
            // r = H_k * grad f
        }
    }

    /// Check the validity of the newly added s and y vectors. Based on the condition in:
    /// D.-H. Li and M. Fukushima, "On the global convergence of the BFGS method for nonconvex
    /// unconstrained optimization problems," vol. 11, no. 4, pp. 1054–1064, jan 2001.
    fn new_s_and_y_valid(&self, g: &[f64], cbfgs_alpha: f64, cbfgs_epsilon: f64) -> bool {
        if cbfgs_epsilon > 0.0 && cbfgs_alpha > 0.0 {
            let sy = vec_ops::inner_product(&self.s.last().unwrap(), &self.y.last().unwrap())
                / vec_ops::inner_product(&self.y.last().unwrap(), &self.y.last().unwrap());

            let ep = cbfgs_epsilon * vec_ops::norm2(g).powf(cbfgs_alpha);

            // Condition: (y^T * s) / ||s||^2 > epsilon * ||grad(x)||^alpha
            sy > ep && sy.is_finite() && ep.is_finite()
        } else {
            true
        }
    }

    /// Saves vectors to update the Hessian estimate
    pub fn update_hessian(
        &mut self,
        g: &[f64],
        state: &[f64],
        cbfgs_alpha: f64,
        cbfgs_epsilon: f64,
    ) -> UpdateStatus {
        assert!(g.len() == self.old_state.len());
        assert!(state.len() == self.old_state.len());

        // Form the new s_k in the temporary area
        vec_ops::difference_and_save(&mut self.s.last_mut().unwrap(), &state, &self.old_state);

        // Form the new y_k in the temporary area
        vec_ops::difference_and_save(&mut self.y.last_mut().unwrap(), &g, &self.old_g);

        // Check that the s and y are valid to use
        if self.new_s_and_y_valid(g, cbfgs_alpha, cbfgs_epsilon) {
            self.old_state.copy_from_slice(state);
            self.old_g.copy_from_slice(g);

            // Move the new s_0 and y_0 to the front
            self.s.rotate_right(1);
            self.y.rotate_right(1);

            // Update the Hessian estimate
            self.gamma = vec_ops::inner_product(&self.s[0], &self.y[0])
                / vec_ops::inner_product(&self.y[0], &self.y[0]);

            // Update the indexes and number of active, -1 comes from the temporary area used in
            // the end of s and y to check if they are valid
            self.active_size = (self.s.len() - 1).min(self.active_size + 1);

            UpdateStatus::UpdateOk
        } else {
            UpdateStatus::CBFGSRejection
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    #[should_panic]
    fn lbfgs_panic_problem_size() {
        let _ = Estimator::new(0, 5);
    }

    #[test]
    #[should_panic]
    fn lbfgs_panic_buffer_size() {
        let _ = Estimator::new(5, 0);
    }

    #[test]
    #[should_panic]
    fn lbfgs_panic_apply_size_grad() {
        let mut e = Estimator::new(5, 5);
        e.update_hessian(&vec![0.0; 4], &vec![0.0; 5], 1.0, 1e-12);
    }

    #[test]
    #[should_panic]
    fn lbfgs_panic_apply_state() {
        let mut e = Estimator::new(5, 5);
        e.update_hessian(&vec![0.0; 5], &vec![0.0; 4], 1.0, 1e-12);
    }
}
