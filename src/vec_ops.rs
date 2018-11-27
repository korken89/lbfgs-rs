//! # vec_ops
//!
//! Matrix operations used by the optimization algorithm.
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

use num::{Float, Zero};
use std::iter::Sum;
use std::ops::Mul;

/// Calculate the inner product of two slices
#[inline(always)]
pub fn inner_product<T>(a: &[T], b: &[T]) -> T
where
    T: Float + Sum<T> + Mul<T, Output = T>,
{
    assert!(a.len() == b.len());

    a.iter().zip(b.iter()).map(|(x, y)| (*x) * (*y)).sum()
}

/// Calculate the 2-norm of a slice
#[inline(always)]
pub fn norm1<T>(a: &[T]) -> T
where
    T: Float + Sum<T>,
{
    a.iter().map(|x| x.abs()).sum()
}

/// Calculate the squared 2-norm of a slice
#[inline(always)]
pub fn norm2_sq<T>(a: &[T]) -> T
where
    T: Float + Sum<T>,
{
    inner_product(a, a)
}

/// Calculate the 2-norm of a slice
#[inline(always)]
pub fn norm2<T>(a: &[T]) -> T
where
    T: Float + Sum<T>,
{
    norm2_sq(a).sqrt()
}

/// Calculate the infinity-norm of a slice
#[inline(always)]
pub fn norm_inf<T>(a: &[T]) -> T
where
    T: Float + Zero,
{
    a.iter()
        .fold(T::zero(), |current_max, x| x.abs().max(current_max))
}

/// Calculates the difference of two vectors and saves it in the third
#[inline(always)]
pub fn difference_and_save<T>(out: &mut [T], a: &[T], b: &[T])
where
    T: Float,
{
    assert!(a.len() == b.len());
    assert!(out.len() == a.len());

    out.iter_mut()
        .zip(a.iter().zip(b.iter()))
        .for_each(|(out, (a, b))| *out = (*a) - (*b));
}

/// Calculates a scalar times vector
#[inline(always)]
pub fn scalar_mult<T>(a: &mut [T], s: T)
where
    T: Float,
{
    a.iter_mut().for_each(|out| *out = s * (*out));
}

/// Calculates out = out - s * a
#[inline(always)]
pub fn inplace_vec_sub<T>(out: &mut [T], a: &[T], s: T)
where
    T: Float,
{
    assert!(out.len() == a.len());

    out.iter_mut()
        .zip(a.iter())
        .for_each(|(out, a)| *out = (*out) - s * (*a));
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn inner_product_test() {
        assert_eq!(
            vec_ops::inner_product(&vec![1.0, 2.0, 3.0], &vec![1.0, 2.0, 3.0]),
            14.0
        );
    }

    #[test]
    #[should_panic]
    fn inner_product_test_panic() {
        vec_ops::inner_product(&vec![2.0, 3.0], &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic]
    fn diff_and_save_test_panic1() {
        let mut out = vec![0.0];
        vec_ops::difference_and_save(&mut out, &vec![3.0, 4.0], &vec![1.0, 1.0]);
    }

    #[test]
    #[should_panic]
    fn diff_and_save_test_panic2() {
        let mut out = vec![0.0, 0.0];
        vec_ops::difference_and_save(&mut out, &vec![4.0], &vec![1.0, 1.0]);
    }

    #[test]
    #[should_panic]
    fn diff_and_save_test_panic3() {
        let mut out = vec![0.0, 0.0];
        vec_ops::difference_and_save(&mut out, &vec![3.0, 4.0], &vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn inplace_vec_sub_panic() {
        let mut out = vec![0.0, 0.0];
        vec_ops::inplace_vec_sub(&mut out, &vec![1.0], 1.0);
    }

    #[test]
    fn norm1_test() {
        assert_eq!(vec_ops::norm1(&vec![1.0, -2.0, -3.0]), 6.0);
    }

    #[test]
    fn norm2_sq_test() {
        assert_eq!(vec_ops::norm2_sq(&vec![3.0, 4.0]), 25.0);
    }

    #[test]
    fn norm2_test() {
        assert_eq!(vec_ops::norm2(&vec![3.0, 4.0]), 5.0);
    }

    #[test]
    fn norm_inf_test() {
        assert_eq!(vec_ops::norm_inf(&vec![1.0, -2.0, -3.0]), 3.0);
        assert_eq!(vec_ops::norm_inf(&vec![1.0, -8.0, -3.0, 0.0]), 8.0);
    }

    #[test]
    fn diff_and_save_test() {
        let mut out = vec![0.0, 0.0];
        let out_result = vec![2.0, 3.0];

        vec_ops::difference_and_save(&mut out, &vec![3.0, 4.0], &vec![1.0, 1.0]);

        assert_eq!(&out, &out_result);
    }

    #[test]
    fn scalar_vector_test() {
        let mut out = vec![1.0, 1.0];
        let out_result = vec![2.0, 2.0];
        let out_result2 = vec![4.0, 4.0];

        vec_ops::scalar_mult(&mut out, 2.0);
        assert_eq!(out, out_result);

        vec_ops::scalar_mult(&mut out, 2.0);
        assert_eq!(out, out_result2);
    }

    #[test]
    fn inplace_vec_sub_test() {
        let mut out = vec![1.0, 1.0];
        let input = vec![1.0, 1.0];
        let out_result = vec![-1.0, -1.0];
        vec_ops::inplace_vec_sub(&mut out, &input, 2.0);

        assert_eq!(out, out_result);
    }
}
