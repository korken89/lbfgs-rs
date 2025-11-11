use std::fmt::Display;

use crate::*;

// Casts f64 to T
fn c<T>(v: f64) -> T
where
    T: LbfgsPrecision + std::iter::Sum<T>,
{
    T::from(v).unwrap()
}

// This function converts a slice of f64 values into a fixed-size array [T; N]
// where T is the target float type (f32 or f64) and N is the size.
// It requires the input slice to have exactly N elements.
fn ca<T, const N: usize>(f64_values: &[f64]) -> [T; N]
where
    T: LbfgsPrecision + std::iter::Sum<T>,
{
    // array::from_fn creates a new array by calling the provided closure N times.
    // The closure takes the index (i) and returns the element at that index.
    std::array::from_fn(|i| {
        // Apply the conversion closure 'c' to the f64 value at index i
        c::<T>(f64_values[i])
    })
}

#[test]
#[should_panic]
fn lbfgs_panic_zero_n() {
    let mut _e = Lbfgs::<f64>::new(0, 1);
}

#[test]
#[should_panic]
fn lbfgs_panic_zero_mem() {
    let mut _e = Lbfgs::<f64>::new(1, 0);
}

#[test]
#[should_panic]
fn lbfgs_panic_apply_size_grad() {
    let mut e = Lbfgs::<f64>::new(5, 5);
    e.update_hessian(&[0.0; 4], &[0.0; 5]);
}

#[test]
#[should_panic]
fn lbfgs_panic_apply_state() {
    let mut e = Lbfgs::<f64>::new(5, 5);
    e.update_hessian(&[0.0; 5], &[0.0; 4]);
}

#[test]
#[should_panic]
fn lbfgs_panic_cbfgs_alpha() {
    let mut _e = Lbfgs::<f64>::new(5, 5).with_cbfgs_alpha(-1.0);
}

#[test]
#[should_panic]
fn lbfgs_panic_cbfgs_epsilon() {
    let mut _e = Lbfgs::<f64>::new(5, 5).with_cbfgs_epsilon(-1.0);
}

#[test]
fn lbfgs_buffer_storage() {
    let mut e = Lbfgs::<f64>::new(2, 3);
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
fn lbfgs_buffer_storage_f32() {
    let mut e = Lbfgs::<f32>::new(2, 3);
    e.update_hessian(&[1.0f32, 1.0f32], &[1.5f32, 1.5f32]);
    assert_eq!(e.active_size, 0);

    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(&[2.0f32, 2.0f32], &[2.5f32, 2.5f32])
    );
    assert_eq!(e.active_size, 1);
    assert_eq!(&e.s[0], &[1.0f32, 1.0f32]);

    assert_eq!(&e.y[0], &[1.0f32, 1.0f32]);

    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(&[-3.0f32, -3.0f32], &[-3.5f32, -3.5f32])
    );
    assert_eq!(e.active_size, 2);
    assert_eq!(&e.s[0], &[-6.0f32, -6.0f32]);
    assert_eq!(&e.s[1], &[1.0f32, 1.0f32]);

    assert_eq!(&e.y[0], &[-5.0f32, -5.0f32]);
    assert_eq!(&e.y[1], &[1.0f32, 1.0f32]);

    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(&[-4.0f32, -4.0f32], &[-4.5f32, -4.5f32])
    );
    assert_eq!(e.active_size, 3);
    assert_eq!(&e.s[0], &[-1.0f32, -1.0f32]);
    assert_eq!(&e.s[1], &[-6.0f32, -6.0f32]);
    assert_eq!(&e.s[2], &[1.0f32, 1.0f32]);

    assert_eq!(&e.y[0], &[-1.0f32, -1.0f32]);
    assert_eq!(&e.y[1], &[-5.0f32, -5.0f32]);
    assert_eq!(&e.y[2], &[1.0f32, 1.0f32]);

    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(&[5.0f32, 5.0f32], &[5.5f32, 5.5f32])
    );
    assert_eq!(e.active_size, 3);
    assert_eq!(&e.s[0], &[10.0f32, 10.0f32]);
    assert_eq!(&e.s[1], &[-1.0f32, -1.0f32]);
    assert_eq!(&e.s[2], &[-6.0f32, -6.0f32]);

    assert_eq!(&e.y[0], &[9.0f32, 9.0f32]);
    assert_eq!(&e.y[1], &[-1.0f32, -1.0f32]);
    assert_eq!(&e.y[2], &[-5.0f32, -5.0f32]);
}

#[test]
fn lbfgs_apply_finite() {
    let mut e = Lbfgs::<f64>::new(2, 3);
    e.update_hessian(&[1.0, 1.0], &[1.5, 1.5]);

    let mut g = [1.0, 1.0];
    e.apply_hessian(&mut g);

    unit_test_utils::assert_none_is_nan(&g, "g");
}

#[test]
fn correctneess_buff_empty() {
    let mut e = Lbfgs::<f64>::new(3, 3);
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
    let mut e = Lbfgs::<f64>::new(3, 3);
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
    let mut e = Lbfgs::<f64>::new(3, 3);
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

fn _correctneess_buff_overfull<T>()
where
    T: LbfgsPrecision + std::iter::Sum<T>,
{
    let mut e = Lbfgs::<T>::new(3, 3);
    let mut g = ca::<T, 3>(&[-2.0, 0.2, -0.3]);

    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(&ca::<T, 3>(&[0.0, 0.0, 0.0]), &ca::<T, 3>(&[0.0, 0.0, 0.0]))
    );
    assert_eq!(
        UpdateStatus::Rejection,
        e.update_hessian(
            &ca::<T, 3>(&[-0.5, 0.6, -1.2]),
            &ca::<T, 3>(&[0.419058177461747, 0.869843029576958, 0.260313940846084])
        )
    );
    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(
            &ca::<T, 3>(&[-0.5, 0.6, -1.2]),
            &ca::<T, 3>(&[0.1, 0.2, -0.3])
        )
    );
    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(
            &ca::<T, 3>(&[-0.75, 0.9, -1.9]),
            &ca::<T, 3>(&[0.19, 0.19, -0.44])
        )
    );

    for _i in 1..10 {
        assert_eq!(
            UpdateStatus::Rejection,
            e.update_hessian(
                &ca::<T, 3>(&[1., 2., 3.]),
                &ca::<T, 3>(&[-0.534522483824849, 0.774541920588438, -0.338187119117343])
            )
        );
    }

    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(
            &ca::<T, 3>(&[-2.25, 3.5, -3.1]),
            &ca::<T, 3>(&[0.39, 0.39, -0.84])
        )
    );

    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(
            &ca::<T, 3>(&[-3.75, 6.3, -4.3]),
            &ca::<T, 3>(&[0.49, 0.59, -1.24])
        )
    );

    e.apply_hessian(&mut g);

    let gamma_correct: T = c(0.077189939288812);
    let alpha_correct = ca::<T, 3>(&[-0.044943820224719, -0.295345104333868, -1.899418829910887]);
    let rho_correct = ca::<T, 3>(&[1.123595505617978, 1.428571428571429, 13.793103448275861]);
    let dir_correct = ca::<T, 3>(&[-0.933604237447365, -0.078865807539102, 1.016318412551302]);

    unit_test_utils::assert_nearly_equal(gamma_correct, e.gamma, T::REL_TOL, T::ABS_TOL, "gamma");
    unit_test_utils::assert_nearly_equal_array(
        &alpha_correct,
        &e.alpha,
        T::REL_TOL,
        T::ABS_TOL,
        "alpha",
    );
    unit_test_utils::assert_nearly_equal_array(
        &rho_correct,
        &e.rho[0..3],
        T::REL_TOL,
        T::ABS_TOL,
        "rho",
    );
    unit_test_utils::assert_nearly_equal_array(
        &dir_correct,
        &g,
        T::REL_TOL,
        T::ABS_TOL,
        "direction",
    );
}

#[test]
fn correctneess_buff_overfull() {
    _correctneess_buff_overfull::<f64>();
    _correctneess_buff_overfull::<f32>();
}

fn _correctneess_reset<T>()
where
    T: LbfgsPrecision + std::iter::Sum<T>,
{
    let mut e = Lbfgs::<T>::new(3, 3);
    let mut g = ca::<T, 3>(&[-3.1, 1.5, 2.1]);

    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(&ca::<T, 3>(&[0.0, 0.0, 0.0]), &ca::<T, 3>(&[0.0, 0.0, 0.0]))
    );
    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(
            &ca::<T, 3>(&[-0.5, 0.6, -1.2]),
            &ca::<T, 3>(&[0.1, 0.2, -0.3])
        )
    );
    e.apply_hessian(&mut g);

    let correct_dir = ca::<T, 3>(&[-1.100601247872944, -0.086568349404424, 0.948633011911515]);
    let alpha_correct = c::<T>(-1.488372093023256);
    let rho_correct = c::<T>(2.325581395348837);

    unit_test_utils::assert_nearly_equal(
        alpha_correct,
        e.alpha[0],
        T::REL_TOL,
        T::ABS_TOL,
        "alpha",
    );
    unit_test_utils::assert_nearly_equal(rho_correct, e.rho[0], T::REL_TOL, T::ABS_TOL, "rho");
    unit_test_utils::assert_nearly_equal_array(
        &correct_dir,
        &g,
        T::REL_TOL,
        T::ABS_TOL,
        "direction",
    );

    e.reset();

    let mut g = ca::<T, 3>(&[-3.1, 1.5, 2.1]);

    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(&ca::<T, 3>(&[0.0, 0.0, 0.0]), &ca::<T, 3>(&[0.0, 0.0, 0.0]))
    );
    assert_eq!(
        UpdateStatus::UpdateOk,
        e.update_hessian(
            &ca::<T, 3>(&[-0.5, 0.6, -1.2]),
            &ca::<T, 3>(&[0.1, 0.2, -0.3])
        )
    );
    e.apply_hessian(&mut g);

    unit_test_utils::assert_nearly_equal(
        alpha_correct,
        e.alpha[0],
        T::REL_TOL,
        T::ABS_TOL,
        "alpha",
    );
    unit_test_utils::assert_nearly_equal(rho_correct, e.rho[0], T::REL_TOL, T::ABS_TOL, "rho");
    unit_test_utils::assert_nearly_equal_array(
        &correct_dir,
        &g,
        T::REL_TOL,
        T::ABS_TOL,
        "direction",
    );
}

#[test]
fn correctneess_reset() {
    _correctneess_reset::<f64>();
    _correctneess_reset::<f32>();
}

fn _reject_perpendicular_sy<T>()
where
    T: LbfgsPrecision + std::iter::Sum<T>,
{
    let n = 3;
    let mem = 5;
    let mut lbfgs = Lbfgs::<T>::new(n, mem);

    assert_eq!(
        UpdateStatus::UpdateOk,
        lbfgs.update_hessian(&ca::<T, 3>(&[0.0, 0.0, 0.0]), &ca::<T, 3>(&[0.0, 0.0, 0.0]))
    );
    assert_eq!(0, lbfgs.active_size);

    // this will fail because y's == 0
    assert_eq!(
        UpdateStatus::Rejection,
        lbfgs.update_hessian(
            &ca::<T, 3>(&[-0.5, 0.6, -1.2]),
            &ca::<T, 3>(&[0.419058177461747, 0.869843029576958, 0.260313940846084])
        )
    );
    assert_eq!(0, lbfgs.active_size);

    assert_eq!(
        UpdateStatus::UpdateOk,
        lbfgs.update_hessian(
            &ca::<T, 3>(&[-0.5, 0.6, -1.2]),
            &ca::<T, 3>(&[0.1, 0.2, -0.3])
        )
    );
    assert_eq!(1, lbfgs.active_size);

    // this will fail because y's is negative
    assert_eq!(
        UpdateStatus::Rejection,
        lbfgs.update_hessian(&ca::<T, 3>(&[1.1, 2., 3.]), &ca::<T, 3>(&[-0.5, 0.7, -0.3]))
    );
    assert_eq!(1, lbfgs.active_size);

    assert_eq!(
        UpdateStatus::UpdateOk,
        lbfgs.update_hessian(
            &ca::<T, 3>(&[-0.75, 0.9, -1.9]),
            &ca::<T, 3>(&[0.19, 0.19, -0.44])
        )
    );
    assert_eq!(2, lbfgs.active_size);
}

#[test]
fn reject_perpendicular_sy() {
    _reject_perpendicular_sy::<f64>();
    _reject_perpendicular_sy::<f32>();
}

/// Tests the rejection mechanism when the step length vector 's' is zero,
/// which violates the L-BFGS curvature condition `||s||^2 > 0`.
/// This ensures the update is rejected with `UpdateStatus::Rejection` due to
/// the near-zero norm of 's'.
fn _reject_norm_s_zero<T>()
where
    T: LbfgsPrecision + std::iter::Sum<T>,
{
    let n = 3;
    let mem = 5;
    let mut lbfgs = Lbfgs::<T>::new(n, mem);

    assert_eq!(
        UpdateStatus::UpdateOk,
        lbfgs.update_hessian(
            &ca::<T, 3>(&[1.0, 2.0, -1.0]),
            &ca::<T, 3>(&[5.0, 5.0, 5.0])
        )
    );

    assert_eq!(
        UpdateStatus::Rejection,
        lbfgs.update_hessian(
            &ca::<T, 3>(&[
                1.0 + std::f64::MIN_POSITIVE,
                2.0 + std::f64::MIN_POSITIVE,
                -1.0 + std::f64::MIN_POSITIVE
            ]),
            &ca::<T, 3>(&[5., 5., 5.])
        )
    );
}

#[test]
fn reject_norm_s_zero() {
    _reject_norm_s_zero::<f64>();
    _reject_norm_s_zero::<f32>();
}

/// Tests the rejection mechanism based on the C-BFGS (Curved-BFGS) criterion,
/// which ensures the Hessian update is only performed when the curvature of
/// the function is sufficiently strong, guarding against updates based on noisy
/// or flat steps in non-convex regions.
fn _reject_cfbs_condition<T>()
where
    T: LbfgsPrecision + std::iter::Sum<T>,
{
    let n = 3;
    let mem = 5;
    let mut lbfgs = Lbfgs::<T>::new(n, mem)
        .with_sy_epsilon(c(1e-8))
        .with_cbfgs_alpha(c(1.0))
        .with_cbfgs_epsilon(c(1e-4));

    assert_eq!(
        UpdateStatus::UpdateOk,
        lbfgs.update_hessian(&ca::<T, 3>(&[0., 0., 0.]), &ca::<T, 3>(&[0., 0., 0.]))
    );

    // rejected because of CBFGS condition
    assert_eq!(
        UpdateStatus::Rejection,
        lbfgs.update_hessian(
            &ca::<T, 3>(&[-0.838, 0.260, 0.479]),
            &ca::<T, 3>(&[-0.5, 0.6, -1.2])
        )
    );
}

#[test]
fn reject_cfbs_condition() {
    // Testing both f64 and f32
    _reject_cfbs_condition::<f64>();
    _reject_cfbs_condition::<f32>();
}
