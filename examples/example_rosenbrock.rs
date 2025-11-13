use lbfgs::{
    vec_ops::{inner_product, norm2},
    Lbfgs, UpdateStatus,
};


// Constants for Wolfe Conditions
const WOLFE_C1: f64 = 1e-5; // Armijo constant
const WOLFE_C2: f64 = 0.99; // Curvature constant
const MAX_LS_STEPS: u32 = 15; // max LS steps
const ALPHA_MAX: f64 = 0.9995; // Max initial step size
const ALPHA_MIN: f64 = 1e-12; // Mix alpha
const LS_BETA: f64 = 0.5; // Step size reduction factor


// The user provides an implementaiton of f(x) and ∇f(x)
// in a single function
fn f_and_df(x: &[f64]) -> (f64, Vec<f64>) {
    let x0 = x[0];
    let x1 = x[1];
    let p = 10.;

    // Function Value: f = (1-x)^2 + p(y - x^2)^2
    let t1 = 1.0 - x0;
    let t2 = x1 - x0 * x0;
    let f = t1 * t1 + p * t2 * t2;

    let mut g = vec![0.0; 2];
    g[0] = -2.0 * t1 - 4. * p * t2 * x0;
    g[1] = 2. * p * t2;

    (f, g)
}


/// Computes the gradient projection onto the search direction: ∇f_k · d_k
fn directional_derivative(grad: &[f64], search_dir: &[f64]) -> f64 {
    inner_product(grad, search_dir)
}

/// Compute x + alpha*d
pub fn compute_x_alpha_d(x: &[f64], d: &[f64], alpha: f64) -> Vec<f64> {
    x.iter()
        .zip(d.iter())
        .map(|(xi, di)| xi + alpha * di)
        .collect()
}

fn zoom(
    x: &[f64],
    d: &[f64],
    fx_0: f64,
    phi_prime_0: f64,
    mut alpha_low: f64,
    mut alpha_high: f64,
) -> f64 {
    for _ in 0..MAX_LS_STEPS {
        let alpha_j = (alpha_low + alpha_high) / 2.0;
        let x_j = compute_x_alpha_d(x, d, alpha_j);
        let (fx_j, grad_j) = f_and_df(&x_j);
        let phi_prime_j = directional_derivative(&grad_j, d); // ∇f(x_j)·d
        if fx_j > fx_0 + WOLFE_C1 * alpha_j * phi_prime_0
            || fx_j >= f_and_df(&compute_x_alpha_d(x, d, alpha_low)).0
        {
            alpha_high = alpha_j;
        } else {
            if phi_prime_j.abs() <= WOLFE_C2 * phi_prime_0.abs() {
                return alpha_j;
            }
            if phi_prime_j * (alpha_high - alpha_low) >= 0.0 {
                alpha_high = alpha_low;
            }
            alpha_low = alpha_j;
        }
    }
    0.0
}

fn wolfe_line_search(x: &[f64], d: &[f64], fx_0: f64, grad_0: &[f64]) -> f64 {
    let phi_prime_0 = directional_derivative(grad_0, d); // ∇f(x)·d
    let mut alpha_prev = 0.0;
    let mut fx_prev = fx_0;
    let mut alpha_i = ALPHA_MAX;

    for i in 0..MAX_LS_STEPS {
        let x_i = compute_x_alpha_d(x, d, alpha_i);
        let (fx_i, grad_i) = f_and_df(&x_i);
        let phi_prime_i = directional_derivative(&grad_i, d); // ∇f(x_i)·d
        if fx_i > fx_0 + WOLFE_C1 * alpha_i * phi_prime_0 || (i > 0 && fx_i >= fx_prev) {
            return zoom(x, d, fx_0, phi_prime_0, alpha_prev, alpha_i);
        }

        // Strong Wolfe condition: |∇f(x_i)·d| <= c2 * |∇f(x)·d|
        if phi_prime_i.abs() <= WOLFE_C2 * phi_prime_0.abs() {
            return alpha_i;
        }

        if phi_prime_i >= 0.0 {
            let alpha_low = alpha_i.min(alpha_prev);
            let alpha_high = alpha_i.max(alpha_prev);
            return zoom(x, d, fx_0, phi_prime_0, alpha_low, alpha_high);
        }
        alpha_prev = alpha_i;
        fx_prev = fx_i;
        alpha_i *= LS_BETA;
    }

    0.0
}

fn main() {
    // Problem Configuration
    let problem_size = 2; // x is a 2D vector
    let lbfgs_memory_size = 10;
    let max_iterations = 100;
    let tolerance = 1e-6;
    let mut x: Vec<f64> = vec![-0.9, 1.5]; // Initial guess

    let mut lbfgs = Lbfgs::new(problem_size, lbfgs_memory_size);
    let (mut fx, mut grad) = f_and_df(&x);
    assert_eq!(lbfgs.update_hessian(&x, &grad), UpdateStatus::UpdateOk);

    for k in 0..max_iterations {
        if norm2(&grad) < tolerance {
            println!("\nCONVERGED! (|∇f| < {:.0e})", tolerance);
            break;
        }
        let mut search_dir = grad.clone();
        lbfgs.apply_hessian(&mut search_dir); 
        let d: Vec<f64> = search_dir.iter().map(|v| -v).collect();
        let mut alpha = wolfe_line_search(&x, &d, fx, &grad);
        if alpha <= ALPHA_MIN {
            println!("\nLine search stalled (alpha < {:.2e})", ALPHA_MIN);
            alpha = 1e-3;
        }
        let mut x_new = vec![0.0; problem_size];
        for i in 0..problem_size {
            x_new[i] = x[i] + alpha * d[i];
        }

        let (fx_new, grad_new) = f_and_df(&x_new);
        let update_status = lbfgs.update_hessian(&grad_new, &x_new);

        x = x_new;
        fx = fx_new;
        grad = grad_new;

        println!(
            "Iter {:>3}: f(x) = {:10.3e}, |∇f| = {:10.3e}, α = {:7.6} / {:?}",
            k + 1, fx, norm2(&grad), alpha, update_status
        );

        if k + 1 == max_iterations {
            println!("Max iterations reached.");
        }
    }

    println!("x* = {:.3?}", x);
}
