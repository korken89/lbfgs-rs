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
    /// y holds the vectors of gradient difference y_k = df(x_{k+1}) - df(x_k), y_0 holds the most recent y
    y: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    rho: Vec<f64>, // Only needed inside apply
    q: Vec<f64>,
    old_state: Vec<f64>,
    old_gradient: Vec<f64>,
}

impl Estimator {
    pub fn new(problem_size: usize, buffer_size: usize) -> Estimator {
        assert!(problem_size > 0); // TODO: Replace this with core::nonzero when stable
        assert!(buffer_size > 0);

        Estimator {
            active_size: 0,
            gamma: 0.0,
            s: vec![vec![0.0; problem_size]; buffer_size],
            y: vec![vec![0.0; problem_size]; buffer_size],
            alpha: vec![0.0; buffer_size],
            rho: vec![0.0; buffer_size],
            q: vec![0.0; problem_size],
            old_state: vec![0.0; problem_size],
            old_gradient: vec![0.0; problem_size],
        }
    }

    // Apply the current Hessian estimate to a gradient
    pub fn apply_hessian(&mut self, gradient: &mut [f64]) {
        if self.active_size == 0 {
            // No Hessian available, only make it go towards minima
            vec_ops::scalar_mult(gradient, -1.0);
        } else {
            self.q.copy_from_slice(gradient);

            // Perform the forward / backward L-BFGS algorithm
        }
    }

    /// Check the validity of the newly added s and y vectors. Based on the condition in:
    /// "D.-H. Li and M. Fukushima, “On the global convergence of the BFGS method for nonconvex
    /// unconstrained optimization problems,” vol. 11, no. 4, pp. 1054–1064, jan 2001.
    fn new_s_and_y_valid(&self, gradient: &[f64]) -> bool {
        // TODO: Check if EPSILON should be changed
        const EPSILON: f64 = 1e-12;

        let sy = vec_ops::inner_product(&self.s.last().unwrap(), &self.y.last().unwrap())
            / vec_ops::inner_product(&self.y.last().unwrap(), &self.y.last().unwrap());

        let ep = EPSILON * vec_ops::norm2(gradient);

        // Condition: (y^T * s)/||s||^2 > epsilon * ||grad(x)||
        sy > ep
    }

    pub fn update_hessian(&mut self, gradient: &mut [f64], state: &[f64]) {
        assert!(gradient.len() == self.old_state.len());
        assert!(state.len() == self.old_state.len());

        // Form the new s_k and save the state
        vec_ops::difference_and_save(&mut self.s.last_mut().unwrap(), &state, &self.old_state);
        self.old_state.copy_from_slice(state);

        // Form the new y_k and save the gradient
        vec_ops::difference_and_save(
            &mut self.y.last_mut().unwrap(),
            &gradient,
            &self.old_gradient,
        );
        self.old_gradient.copy_from_slice(gradient);

        // Check that the s and y are valid to use
        if self.new_s_and_y_valid(gradient) == true {
            // Move the new s_0 and y_0 to the front
            self.s.rotate_right(1);
            self.y.rotate_right(1);

            // Update the Hessian estimate
            self.gamma = vec_ops::inner_product(&self.s[0], &self.y[0])
                / vec_ops::inner_product(&self.y[0], &self.y[0]);

            // Update the indexes and number of active
            self.active_size = self.s.len().min(self.active_size + 1);
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
        e.update_hessian(&mut vec![0.0; 4], &vec![0.0; 5]);
    }

    #[test]
    #[should_panic]
    fn lbfgs_panic_apply_state() {
        let mut e = Estimator::new(5, 5);
        e.update_hessian(&mut vec![0.0; 5], &vec![0.0; 4]);
    }

    #[test]
    fn lbfgs_test() {
        let mut e = Estimator::new(2, 3);
        println!("LBFGS instance: {:?}", e);
        e.update_hessian(&mut vec![1.0, 1.0], &vec![1.0, 1.0]);
        println!("LBFGS instance: {:?}", e);
        e.update_hessian(&mut vec![3.0, 3.0], &vec![3.0, 3.0]);
        println!("LBFGS instance: {:?}", e);
        e.update_hessian(&mut vec![6.0, 6.0], &vec![6.0, 6.0]);
        println!("LBFGS instance: {:?}", e);
        e.update_hessian(&mut vec![10.0, 10.0], &vec![10.0, 10.0]);
        println!("LBFGS instance: {:?}", e);
    }
}
