use lbfgs::{
    Lbfgs, UpdateStatus, vec_ops::{inner_product, norm2, scalar_mult}
};

// Constants for Wolfe Conditions
const WOLFE_C1: f64 = 1e-5; // Armijo constant
const WOLFE_C2: f64 = 0.99; // Curvature constant
const MAX_LS_STEPS: u32 = 12; // max LS steps
const ALPHA_MAX: f64 = 0.9995; // Max initial step size
const ALPHA_MIN: f64 = 1e-12; // Mix alpha
const LS_BETA: f64 = 0.5; // Step size reduction factor
const DEFAULT_TOLERANCE: f64 = 1e-6;
const DEFAULT_MAX_ITERATIONS: usize = 100;

/// Solver status
#[derive(Debug, PartialEq)]
pub enum LbfgsSolverStatus {
    Converged,
    ZoomIterationsExceeded,
    LineSearchIterationsExceeded,
    MaximumIterationsExceeded,
}

// Computes out = x + alpha * d
pub fn xpad(out: &mut [f64], x: &[f64], alpha: f64, d: &[f64]) {
    out.iter_mut()
        .zip(x.iter())
        .zip(d.iter())
        .for_each(|((out, xi), di)| *out = (*xi) + alpha * (*di));
}

/// LBFGS Solver with line search satisfying the strong Wolfe conditions
/// 
/// The cost function and its gradient are specified via a closure/function of the form 
/// `Fn(&[f64], &mut [f64]) -> f64`. The first argument is the input vector `x`,
/// the second argument is a mutable slice where the gradient `∇f(x)` will be stored,
/// and the return value is the function value `f(x)`.
/// 
/// 
struct LbfgsSolver<FunctionT>
where
     FunctionT: Fn(&[f64], &mut [f64]) -> f64,
{
    /// Cost function with its gradient
    func: FunctionT,
    /// Tolerance for the termination condition |∇f(x)| ≤ tolerance
    tolerance: f64,
    /// Maximum number of iterations
    max_iterations: usize,
    /// LBFGS instance
    lbfgs: Lbfgs<f64>,
    /// Whether to print details while solving
    verbose: bool,
    // -- Workspace -------------------------------
    d: Vec<f64>,
    x_ws_1: Vec<f64>,
    x_ws_2: Vec<f64>,
    x_new: Vec<f64>,
    search_dir: Vec<f64>,
    grad: Vec<f64>,
    grad_plus: Vec<f64>,
}

impl<FunctionT> LbfgsSolver<FunctionT>
where
    FunctionT: Fn(&[f64], &mut [f64]) -> f64,
{

    /// Creates a new LBFGS solver instance
    /// 
    /// # Arguments
    /// 
    /// * `func` - The cost function with its gradient
    /// * `n` - The problem size (number of variables)
    /// * `lbfgs_memory` - The LBFGS memory size
    ///
    /// # Returns
    /// 
    /// A new `LbfgsSolver` instance
    /// 
    pub fn new(func: FunctionT, n: usize, lbfgs_memory: usize) -> Self {
        let lbfgs = Lbfgs::<f64>::new(n, lbfgs_memory);
        LbfgsSolver {
            func,
            tolerance: DEFAULT_TOLERANCE,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            lbfgs,
            verbose: false,
            // WS
            d: vec![0.0; n],
            x_ws_1: vec![0.0; n],
            x_ws_2: vec![0.0; n],
            x_new: vec![0.0; n],
            search_dir: vec![0.0; n],
            grad: vec![0.0; n],
            grad_plus: vec![0.0; n],
        }
    }

    /// Sets the tolerance for the termination condition |∇f(x)| ≤ tolerance
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }

    /// Sets the maximum number of iterations
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

    /// Sets whether to print details while solving
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Zoom algorithm (internal)
    fn zoom(
        &mut self,
        x: &[f64],
        fx_0: f64,
        phi_prime_0: f64,
        mut alpha_low: f64,
        mut alpha_high: f64,
    ) -> Result<f64, LbfgsSolverStatus> {
        
        for _ in 0..MAX_LS_STEPS {
            let alpha_j = (alpha_low + alpha_high) / 2.0;
            xpad(&mut self.x_ws_1, x, alpha_j, &self.d);

            let fx_j  = (self.func)(&self.x_ws_1, &mut self.grad);
            let phi_prime_j = inner_product(&self.grad, &self.d); // ∇f(x_j)·d
            xpad(&mut self.x_ws_2, x, alpha_low, &self.d);
            if fx_j > fx_0 + WOLFE_C1 * alpha_j * phi_prime_0
                || fx_j >= (self.func)(&self.x_ws_2, &mut self.grad)
            {
                alpha_high = alpha_j;
            } else {
                if phi_prime_j.abs() <= WOLFE_C2 * phi_prime_0.abs() {
                    return Ok(alpha_j);
                }
                if phi_prime_j * (alpha_high - alpha_low) >= 0.0 {
                    alpha_high = alpha_low;
                }
                alpha_low = alpha_j;
            }
        }
        Err(LbfgsSolverStatus::ZoomIterationsExceeded)
    }

    /// Line search (internal)
    fn wolfe_line_search(
        &mut self,
        x: &[f64],
        fx_0: f64,
    ) -> Result<f64, LbfgsSolverStatus> {
        self.d.copy_from_slice(&self.search_dir);
        scalar_mult(&mut self.d, -1.0); 

        let phi_prime_0 = inner_product(&self.grad, &self.d); // ∇f(x)·d
        let mut alpha_prev = 0.0;
        let mut fx_prev = fx_0;
        let mut alpha_i = ALPHA_MAX;

        for i in 0..MAX_LS_STEPS {
            xpad(&mut self.x_ws_1, x, alpha_i, &self.d);
            let fx_i = (self.func)(&self.x_ws_1, &mut self.grad);
            let phi_prime_i = inner_product(&self.grad, &self.d); // ∇f(x_i)·d
            if fx_i > fx_0 + WOLFE_C1 * alpha_i * phi_prime_0 || (i > 0 && fx_i >= fx_prev) {
                return self.zoom(x, fx_0, phi_prime_0, alpha_prev, alpha_i);
            }

            // Strong Wolfe condition: |∇f(x_i)·d| <= c2 * |∇f(x)·d|
            if phi_prime_i.abs() <= WOLFE_C2 * phi_prime_0.abs() {
                return Ok(alpha_i);
            }

            if phi_prime_i >= 0.0 {
                let alpha_low = alpha_i.min(alpha_prev);
                let alpha_high = alpha_i.max(alpha_prev);
                return self.zoom(x, fx_0, phi_prime_0, alpha_low, alpha_high);
            }
            alpha_prev = alpha_i;
            fx_prev = fx_i;
            alpha_i *= LS_BETA;
        }

        Err(LbfgsSolverStatus::LineSearchIterationsExceeded)
    }

    /// Solves the optimization problem starting from initial guess x
    /// 
    /// # Arguments
    /// * `x` - Initial guess (will be modified to contain the solution)
    /// 
    /// # Returns
    /// * `LbfgsSolverStatus` - The status of the solver after completion
    ///
    pub fn solve(&mut self, x: &mut Vec<f64>) -> LbfgsSolverStatus {
        self.lbfgs.reset();
        let mut fx = (self.func)(&x, &mut self.grad);        
        assert_eq!(self.lbfgs.update_hessian(&x, &self.grad), UpdateStatus::UpdateOk);
        for k in 0..self.max_iterations {
            if norm2(&self.grad) < self.tolerance {
                if self.verbose {
                    println!("\nCONVERGED! (|∇f| < {:.0e})", self.tolerance);
                }
                break;
            }
            self.search_dir.copy_from_slice(&self.grad);
            self.lbfgs.apply_hessian(&mut self.search_dir); // search_dir = H * grad
            let alpha = self
                .wolfe_line_search(&x, fx)
                .unwrap_or(ALPHA_MIN);
            xpad(&mut self.x_new, x, alpha, &self.d);
            let fx_new = (self.func)(&self.x_new, &mut self.grad_plus);
            let update_status = self.lbfgs.update_hessian(&self.grad_plus, &self.x_new);

            x.copy_from_slice(&self.x_new);
            fx = fx_new;
            self.grad.copy_from_slice(&self.grad_plus);

            if self.verbose {
                println!(
                    "Iter {:>3}: f(x) = {:10.3e}, |∇f| = {:10.3e}, alpha = {:7.6} / {:?}",
                    k + 1,
                    fx,
                    norm2(&self.grad),
                    alpha,
                    update_status
                );
            }

            if k + 1 == self.max_iterations {
                return LbfgsSolverStatus::MaximumIterationsExceeded;
            }
        }
        return LbfgsSolverStatus::Converged;
    }
}

// The user provides an implementaiton of f(x) and ∇f(x)
// in a single function; this is the Rosenbrock function
fn f_and_df(x: &[f64], g: &mut [f64]) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    let p = 10.;

    // Function Value: f = (1-x)^2 + p(y - x^2)^2
    let t1 = 1.0 - x0;
    let t2 = x1 - x0 * x0;
    let f = t1 * t1 + p * t2 * t2;

    g[0] = -2.0 * t1 - 4. * p * t2 * x0;
    g[1] = 2. * p * t2;

    f
}

fn main() {
    let mem = 10; // LBFGS buffer length
    let mut x: Vec<f64> = vec![-1.5, 1.5]; // Initial guess
    let mut solver = LbfgsSolver::new(f_and_df, 2, mem);
    solver.set_tolerance(1e-4);
    solver.set_max_iterations(100);
    solver.set_verbose(true);
    assert_eq!(solver.solve(&mut x), LbfgsSolverStatus::Converged);
    println!("x* = {:.3?}", x);
}
