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
    /// The alpha parameter of the C-BFGS criterion
    cbfgs_alpha: f64,
    /// The epsilon parameter of the C-BFGS criterion
    cbfgs_epsilon: f64,
    old_state: Vec<f64>,
    old_g: Vec<f64>,
    first_old: bool,
}

#[derive(Debug, Copy, Clone)]
pub enum UpdateStatus {
    /// The g and state was accepted to update the Hessian estimate
    UpdateOk,
    /// The g and state was rejected by the C-BFGS criteria
    CBFGSRejection,
}

impl Estimator {
    pub fn new(problem_size: NonZeroUsize, buffer_size: NonZeroUsize) -> Estimator {
        let problem_size = problem_size.get();
        let buffer_size = buffer_size.get();

        Estimator {
            active_size: 0,
            gamma: 1.0,
            s: vec![vec![0.0; problem_size]; buffer_size + 1], // +1 for the temporary checking area
            y: vec![vec![0.0; problem_size]; buffer_size + 1], // +1 for the temporary checking area
            alpha: vec![0.0; buffer_size],
            rho: vec![0.0; buffer_size],
            cbfgs_alpha: 0.0,
            cbfgs_epsilon: 0.0,
            old_state: vec![0.0; problem_size],
            old_g: vec![0.0; problem_size],
            first_old: true,
        }
    }

    /// Update the default C-BFGS alpha
    pub fn with_cbfgs_alpha(&mut self, alpha: f64) {
        assert!(alpha >= 0.0);

        self.cbfgs_alpha = alpha;
    }

    /// Update the default C-BFGS epsilon
    pub fn with_cbfgs_epsilon(&mut self, epsilon: f64) {
        assert!(epsilon >= 0.0);

        self.cbfgs_epsilon = epsilon;
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
            for (idx, (s_k, y_k)) in active_s.iter().zip(active_y.iter()).enumerate() {
                let r = 1.0 / vec_ops::inner_product(s_k, y_k);

                if !r.is_finite() {
                    return;
                }

                let a = r * vec_ops::inner_product(s_k, q);

                rho[idx] = r;
                alpha[idx] = a;

                vec_ops::inplace_vec_add(q, y_k, -a);
            }

            // Apply the initial Hessian estimate and form r = H_0 * q, where H_0 = gamma * I
            vec_ops::scalar_mult(q, self.gamma);
            let r = q;

            // Perform the backward L-BFGS algorithm
            for (idx, (s_k, y_k)) in active_s.iter().zip(active_y.iter()).enumerate().rev() {
                let beta = rho[idx] * vec_ops::inner_product(y_k, r);

                vec_ops::inplace_vec_add(r, s_k, alpha[idx] - beta);
            }

            // The g with the Hessian applied is available in the input g
            // r = H_k * grad f
        }
    }

    /// Check the validity of the newly added s and y vectors. Based on the condition in:
    /// D.-H. Li and M. Fukushima, "On the global convergence of the BFGS method for nonconvex
    /// unconstrained optimization problems," vol. 11, no. 4, pp. 1054–1064, jan 2001.
    fn new_s_and_y_valid(&self, g: &[f64]) -> bool {
        if self.cbfgs_epsilon > 0.0 && self.cbfgs_alpha > 0.0 {
            let sy = vec_ops::inner_product(&self.s.last().unwrap(), &self.y.last().unwrap())
                / vec_ops::inner_product(&self.s.last().unwrap(), &self.s.last().unwrap());

            let ep = self.cbfgs_epsilon * vec_ops::norm2(g).powf(self.cbfgs_alpha);

            // Condition: (y^T * s) / ||s||^2 > epsilon * ||grad(x)||^alpha
            sy > ep && sy.is_finite() && ep.is_finite()
        } else {
            true
        }
    }

    /// Saves vectors to update the Hessian estimate
    pub fn update_hessian(&mut self, g: &[f64], state: &[f64]) -> UpdateStatus {
        assert!(g.len() == self.old_state.len());
        assert!(state.len() == self.old_state.len());

        // First iteration, only save
        if self.first_old == true {
            self.first_old = false;

            self.old_state.copy_from_slice(state);
            self.old_g.copy_from_slice(g);

            return UpdateStatus::UpdateOk;
        }

        // Form the new s_k in the temporary area
        vec_ops::difference_and_save(&mut self.s.last_mut().unwrap(), &state, &self.old_state);

        // Form the new y_k in the temporary area
        vec_ops::difference_and_save(&mut self.y.last_mut().unwrap(), &g, &self.old_g);

        // Check that the s and y are valid to use
        if self.new_s_and_y_valid(g) {
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
    use num::Float;

    fn vec_is_finite<T>(vec: &[T]) -> bool
    where
        T: Float,
    {
        for val in vec.iter() {
            if !val.is_finite() {
                return false;
            }
        }

        true
    }

    fn assert_ae(x: f64, y: f64, tol: f64, msg: &str) {
        if (x - y).abs() > tol {
            panic!("({}) {} != {} [log(tol) = {}]", msg, x, y, tol.log10());
        }
    }

    fn assert_array_ae(x: &[f64], y: &[f64], tol: f64, msg: &str) {
        x.iter()
            .zip(y.iter())
            .for_each(|(&xi, &yi)| assert_ae(xi, yi, tol, msg));
    }

    #[test]
    #[should_panic]
    fn lbfgs_panic_apply_size_grad() {
        let mut e = Estimator::new(NonZeroUsize::new(5).unwrap(), NonZeroUsize::new(5).unwrap());
        e.update_hessian(&vec![0.0; 4], &vec![0.0; 5]);
    }

    #[test]
    #[should_panic]
    fn lbfgs_panic_apply_state() {
        let mut e = Estimator::new(NonZeroUsize::new(5).unwrap(), NonZeroUsize::new(5).unwrap());
        e.update_hessian(&vec![0.0; 5], &vec![0.0; 4]);
    }

    #[test]
    fn lbfgs_buffer_storage() {
        let mut e = Estimator::new(NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(3).unwrap());
        e.update_hessian(&vec![1.0, 1.0], &vec![1.5, 1.5]);
        assert_eq!(e.active_size, 0);

        e.update_hessian(&vec![2.0, 2.0], &vec![2.5, 2.5]);
        assert_eq!(e.active_size, 1);
        assert_eq!(&e.s[0], &vec![1.0, 1.0]);

        assert_eq!(&e.y[0], &vec![1.0, 1.0]);

        e.update_hessian(&vec![-3.0, -3.0], &vec![-3.5, -3.5]);
        assert_eq!(e.active_size, 2);
        assert_eq!(&e.s[0], &vec![-6.0, -6.0]);
        assert_eq!(&e.s[1], &vec![1.0, 1.0]);

        assert_eq!(&e.y[0], &vec![-5.0, -5.0]);
        assert_eq!(&e.y[1], &vec![1.0, 1.0]);

        e.update_hessian(&vec![-4.0, -4.0], &vec![-4.5, -4.5]);
        assert_eq!(e.active_size, 3);
        assert_eq!(&e.s[0], &vec![-1.0, -1.0]);
        assert_eq!(&e.s[1], &vec![-6.0, -6.0]);
        assert_eq!(&e.s[2], &vec![1.0, 1.0]);

        assert_eq!(&e.y[0], &vec![-1.0, -1.0]);
        assert_eq!(&e.y[1], &vec![-5.0, -5.0]);
        assert_eq!(&e.y[2], &vec![1.0, 1.0]);

        e.update_hessian(&vec![5.0, 5.0], &vec![5.5, 5.5]);
        assert_eq!(e.active_size, 3);
        assert_eq!(&e.s[0], &vec![10.0, 10.0]);
        assert_eq!(&e.s[1], &vec![-1.0, -1.0]);
        assert_eq!(&e.s[2], &vec![-6.0, -6.0]);

        assert_eq!(&e.y[0], &vec![9.0, 9.0]);
        assert_eq!(&e.y[1], &vec![-1.0, -1.0]);
        assert_eq!(&e.y[2], &vec![-5.0, -5.0]);
    }

    #[test]
    fn lbfgs_apply_finite() {
        let mut e = Estimator::new(NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(3).unwrap());
        e.update_hessian(&vec![1.0, 1.0], &vec![1.5, 1.5]);

        let mut g = [1.0, 1.0];
        e.apply_hessian(&mut g);

        assert_eq!(vec_is_finite(&g), true);
    }

    #[test]
    fn correctneess_buff_empty() {
        let mut e = Estimator::new(NonZeroUsize::new(3).unwrap(), NonZeroUsize::new(3).unwrap());
        let mut g = [-3.1, 1.5, 2.1];
        e.update_hessian(&vec![0.0, 0.0, 0.0], &vec![0.0, 0.0, 0.0]);
        e.apply_hessian(&mut g);
        let correct_dir = [-3.1, 1.5, 2.1];
        assert_array_ae(&correct_dir, &g, 1e-10, "direction");
    }

    #[test]
    fn correctneess_buff_1() {
        let mut e = Estimator::new(NonZeroUsize::new(3).unwrap(), NonZeroUsize::new(3).unwrap());
        let mut g = [-3.1, 1.5, 2.1];

        e.update_hessian(&vec![0.0, 0.0, 0.0], &vec![0.0, 0.0, 0.0]);
        e.update_hessian(&[-0.5, 0.6, -1.2], &[0.1, 0.2, -0.3]);
        e.apply_hessian(&mut g);

        let correct_dir = [-1.100601247872944, -0.086568349404424, 0.948633011911515];
        let alpha_correct = -1.488372093023256;
        let rho_correct = 2.325581395348837;

        assert_ae(alpha_correct, e.alpha[0], 1e-10, "alpha");
        assert_ae(rho_correct, e.rho[0], 1e-10, "rho");
        assert_array_ae(&correct_dir, &g, 1e-10, "direction");
    }

    #[test]
    fn correctneess_buff_2() {
        let mut e = Estimator::new(NonZeroUsize::new(3).unwrap(), NonZeroUsize::new(3).unwrap());
        let mut g = [-3.1, 1.5, 2.1];

        e.update_hessian(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        e.update_hessian(&[-0.5, 0.6, -1.2], &[0.1, 0.2, -0.3]);
        e.update_hessian(&[-0.75, 0.9, -1.9], &[0.19, 0.19, -0.44]);

        e.apply_hessian(&mut g);

        let correct_dir = [-1.814749861477524, 0.895232314736337, 1.871795942557546];

        assert_array_ae(&correct_dir, &g, 1e-10, "direction");
    }

    #[test]
    fn correctneess_buff_overfull() {
        let mut e = Estimator::new(NonZeroUsize::new(3).unwrap(), NonZeroUsize::new(3).unwrap());
        let mut g = [-2.0, 0.2, -0.3];

        e.update_hessian(&vec![0.0, 0.0, 0.0], &vec![0.0, 0.0, 0.0]);
        e.update_hessian(&[-0.5, 0.6, -1.2], &[0.1, 0.2, -0.3]);
        e.update_hessian(&[-0.75, 0.9, -1.9], &[0.19, 0.19, -0.44]);
        e.update_hessian(&[-2.25, 3.5, -3.1], &[0.39, 0.39, -0.84]);
        e.update_hessian(&[-3.75, 6.3, -4.3], &[0.49, 0.59, -1.24]);

        e.apply_hessian(&mut g);

        println!("{:#.3?}", e);

        let gamma_correct = 0.077189939288812;
        let alpha_correct = [-0.044943820224719, -0.295345104333868, -1.899418829910887];
        let rho_correct = [1.123595505617978, 1.428571428571429, 13.793103448275861];
        let dir_correct = [-0.933604237447365, -0.078865807539102, 1.016318412551302];

        assert_ae(gamma_correct, e.gamma, 1e-10, "gamma");
        assert_array_ae(&alpha_correct, &e.alpha, 1e-10, "alpha");
        assert_array_ae(&rho_correct, &e.rho, 1e-10, "rho");
        assert_array_ae(&dir_correct, &g, 1e-10, "direction");
    }
}
