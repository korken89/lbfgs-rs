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
            rho: vec![0.0; buffer_size],
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
        let rho = &mut self.rho;
        let alpha = &mut self.alpha;

        let q = g;

        // Check so all rho_k are finite, else do not update the output
        for (s_k, (y_k, rho_k)) in active_s.iter().zip(active_y.iter().zip(rho.iter_mut())) {
            let r = 1.0 / vec_ops::inner_product(s_k, y_k);

            if !r.is_finite() {
                return;
            }

            *rho_k = r;
        }

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
    fn new_s_and_y_valid(&self, g: &[f64]) -> bool {
        let s = &self.s.last().unwrap();
        let y = &self.y.last().unwrap();
        let ys = vec_ops::inner_product(s, y);
        let norm_s_squared = vec_ops::inner_product(s, s);

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
        if !self.new_s_and_y_valid(g) {
            return UpdateStatus::Rejection;
        }
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
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    #[should_panic]
    fn lbfgs_panic_apply_size_grad() {
        let mut e = Lbfgs::new(NonZeroUsize::new(5).unwrap(), NonZeroUsize::new(5).unwrap());
        e.update_hessian(&[0.0; 4], &[0.0; 5]);
    }

    #[test]
    #[should_panic]
    fn lbfgs_panic_apply_state() {
        let mut e = Lbfgs::new(NonZeroUsize::new(5).unwrap(), NonZeroUsize::new(5).unwrap());
        e.update_hessian(&[0.0; 5], &[0.0; 4]);
    }

    #[test]
    #[should_panic]
    fn lbfgs_panic_cbfgs_alpha() {
        let mut _e = Lbfgs::new(NonZeroUsize::new(5).unwrap(), NonZeroUsize::new(5).unwrap())
            .with_cbfgs_alpha(-1.0);
    }

    #[test]
    #[should_panic]
    fn lbfgs_panic_cbfgs_epsilon() {
        let mut _e = Lbfgs::new(NonZeroUsize::new(5).unwrap(), NonZeroUsize::new(5).unwrap())
            .with_cbfgs_epsilon(-1.0);
    }

    #[test]
    fn lbfgs_buffer_storage() {
        let mut e = Lbfgs::new(NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(3).unwrap());
        e.update_hessian(&[1.0, 1.0], &[1.5, 1.5]);
        assert_eq!(e.active_size, 0);

        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[2.0, 2.0], &[2.5, 2.5])
        );
        assert_eq!(e.active_size, 1);
        assert_eq!(&e.s[0], &[1.0, 1.0]);

        assert_eq!(&e.y[0], &[1.0, 1.0]);

        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[-3.0, -3.0], &[-3.5, -3.5])
        );
        assert_eq!(e.active_size, 2);
        assert_eq!(&e.s[0], &[-6.0, -6.0]);
        assert_eq!(&e.s[1], &[1.0, 1.0]);

        assert_eq!(&e.y[0], &[-5.0, -5.0]);
        assert_eq!(&e.y[1], &[1.0, 1.0]);

        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[-4.0, -4.0], &[-4.5, -4.5])
        );
        assert_eq!(e.active_size, 3);
        assert_eq!(&e.s[0], &[-1.0, -1.0]);
        assert_eq!(&e.s[1], &[-6.0, -6.0]);
        assert_eq!(&e.s[2], &[1.0, 1.0]);

        assert_eq!(&e.y[0], &[-1.0, -1.0]);
        assert_eq!(&e.y[1], &[-5.0, -5.0]);
        assert_eq!(&e.y[2], &[1.0, 1.0]);

        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[5.0, 5.0], &[5.5, 5.5])
        );
        assert_eq!(e.active_size, 3);
        assert_eq!(&e.s[0], &[10.0, 10.0]);
        assert_eq!(&e.s[1], &[-1.0, -1.0]);
        assert_eq!(&e.s[2], &[-6.0, -6.0]);

        assert_eq!(&e.y[0], &[9.0, 9.0]);
        assert_eq!(&e.y[1], &[-1.0, -1.0]);
        assert_eq!(&e.y[2], &[-5.0, -5.0]);
    }

    #[test]
    fn lbfgs_apply_finite() {
        let mut e = Lbfgs::new(NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(3).unwrap());
        e.update_hessian(&[1.0, 1.0], &[1.5, 1.5]);

        let mut g = [1.0, 1.0];
        e.apply_hessian(&mut g);

        unit_test_utils::assert_is_finite_array(&g, "g");
    }

    #[test]
    fn correctneess_buff_empty() {
        let mut e = Lbfgs::new(NonZeroUsize::new(3).unwrap(), NonZeroUsize::new(3).unwrap());
        let mut g = [-3.1, 1.5, 2.1];
        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0])
        );
        e.apply_hessian(&mut g);
        let correct_dir = [-3.1, 1.5, 2.1];
        unit_test_utils::assert_nearly_equal_array(&correct_dir, &g, 1e-8, 1e-10, "direction");
    }

    #[test]
    fn correctneess_buff_1() {
        let mut e = Lbfgs::new(NonZeroUsize::new(3).unwrap(), NonZeroUsize::new(3).unwrap());
        let mut g = [-3.1, 1.5, 2.1];

        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0])
        );
        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[-0.5, 0.6, -1.2], &[0.1, 0.2, -0.3])
        );
        e.apply_hessian(&mut g);

        let correct_dir = [-1.100601247872944, -0.086568349404424, 0.948633011911515];
        let alpha_correct = -1.488372093023256;
        let rho_correct = 2.325581395348837;

        unit_test_utils::assert_nearly_equal(alpha_correct, e.alpha[0], 1e-8, 1e-10, "alpha");
        unit_test_utils::assert_nearly_equal(rho_correct, e.rho[0], 1e-8, 1e-10, "rho");
        unit_test_utils::assert_nearly_equal_array(&correct_dir, &g, 1e-8, 1e-10, "direction");
    }

    #[test]
    fn correctneess_buff_2() {
        let mut e = Lbfgs::new(NonZeroUsize::new(3).unwrap(), NonZeroUsize::new(3).unwrap());
        let mut g = [-3.1, 1.5, 2.1];

        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0])
        );
        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[-0.5, 0.6, -1.2], &[0.1, 0.2, -0.3])
        );
        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[-0.75, 0.9, -1.9], &[0.19, 0.19, -0.44])
        );

        e.apply_hessian(&mut g);

        let correct_dir = [-1.814749861477524, 0.895232314736337, 1.871795942557546];

        unit_test_utils::assert_nearly_equal_array(&correct_dir, &g, 1e-8, 1e-10, "direction");
    }

    #[test]
    fn correctneess_buff_overfull() {
        let mut e = Lbfgs::new(NonZeroUsize::new(3).unwrap(), NonZeroUsize::new(3).unwrap());
        let mut g = [-2.0, 0.2, -0.3];

        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0])
        );
        assert_eq!(
            UpdateStatus::Rejection,
            e.update_hessian(
                &[-0.5, 0.6, -1.2],
                &[0.419058177461747, 0.869843029576958, 0.260313940846084]
            )
        );
        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[-0.5, 0.6, -1.2], &[0.1, 0.2, -0.3])
        );
        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[-0.75, 0.9, -1.9], &[0.19, 0.19, -0.44])
        );

        for _i in 1..10 {
            assert_eq!(
                UpdateStatus::Rejection,
                e.update_hessian(
                    &[1., 2., 3.],
                    &[-0.534522483824849, 0.774541920588438, -0.338187119117343]
                )
            );
        }

        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[-2.25, 3.5, -3.1], &[0.39, 0.39, -0.84])
        );

        assert_eq!(
            UpdateStatus::UpdateOk,
            e.update_hessian(&[-3.75, 6.3, -4.3], &[0.49, 0.59, -1.24])
        );

        e.apply_hessian(&mut g);

        println!("{:#.3?}", e);

        let gamma_correct = 0.077189939288812;
        let alpha_correct = [-0.044943820224719, -0.295345104333868, -1.899418829910887];
        let rho_correct = [1.123595505617978, 1.428571428571429, 13.793103448275861];
        let dir_correct = [-0.933604237447365, -0.078865807539102, 1.016318412551302];

        unit_test_utils::assert_nearly_equal(gamma_correct, e.gamma, 1e-8, 1e-10, "gamma");
        unit_test_utils::assert_nearly_equal_array(&alpha_correct, &e.alpha, 1e-8, 1e-10, "alpha");
        unit_test_utils::assert_nearly_equal_array(&rho_correct, &e.rho, 1e-8, 1e-10, "rho");
        unit_test_utils::assert_nearly_equal_array(&dir_correct, &g, 1e-8, 1e-10, "direction");
    }

    #[test]
    fn reject_perpendicular_sy() {
        let n = NonZeroUsize::new(3).unwrap();
        let mem = NonZeroUsize::new(5).unwrap();
        let mut lbfgs = Lbfgs::new(n, mem).with_sy_epsilon(1e-8);

        assert_eq!(
            UpdateStatus::UpdateOk,
            lbfgs.update_hessian(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0])
        );
        assert_eq!(0, lbfgs.active_size);

        // this will fail because y's == 0
        assert_eq!(
            UpdateStatus::Rejection,
            lbfgs.update_hessian(
                &[-0.5, 0.6, -1.2],
                &[0.419058177461747, 0.869843029576958, 0.260313940846084]
            )
        );
        assert_eq!(0, lbfgs.active_size);

        assert_eq!(
            UpdateStatus::UpdateOk,
            lbfgs.update_hessian(&[-0.5, 0.6, -1.2], &[0.1, 0.2, -0.3])
        );
        assert_eq!(1, lbfgs.active_size);

        // this will fail because y's is negative
        assert_eq!(
            UpdateStatus::Rejection,
            lbfgs.update_hessian(&[1.1, 2., 3.], &[-0.5, 0.7, -0.3])
        );
        assert_eq!(1, lbfgs.active_size);

        assert_eq!(
            UpdateStatus::UpdateOk,
            lbfgs.update_hessian(&[-0.75, 0.9, -1.9], &[0.19, 0.19, -0.44])
        );
        assert_eq!(2, lbfgs.active_size);
    }

    #[test]
    fn reject_norm_s_zero() {
        let n = NonZeroUsize::new(3).unwrap();
        let mem = NonZeroUsize::new(5).unwrap();
        let mut lbfgs = Lbfgs::new(n, mem);

        assert_eq!(
            UpdateStatus::UpdateOk,
            lbfgs.update_hessian(&[1.0, 2.0, -1.0], &[5.0, 5.0, 5.0])
        );

        assert_eq!(
            UpdateStatus::Rejection,
            lbfgs.update_hessian(
                &[
                    1.0 + std::f64::MIN_POSITIVE,
                    2.0 + std::f64::MIN_POSITIVE,
                    -1.0 + std::f64::MIN_POSITIVE
                ],
                &[5.0, 5.0, 5.0]
            )
        );
    }

    #[test]
    fn reject_cfbs_condition() {
        let n = NonZeroUsize::new(3).unwrap();
        let mem = NonZeroUsize::new(5).unwrap();
        let mut lbfgs = Lbfgs::new(n, mem)
            .with_sy_epsilon(1e-8)
            .with_cbfgs_alpha(1.0)
            .with_cbfgs_epsilon(1e-4);

        assert_eq!(
            UpdateStatus::UpdateOk,
            lbfgs.update_hessian(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0])
        );

        // rejected because of CBFGS condition
        assert_eq!(
            UpdateStatus::Rejection,
            lbfgs.update_hessian(&[-0.838, 0.260, 0.479], &[-0.5, 0.6, -1.2])
        );
    }
}
