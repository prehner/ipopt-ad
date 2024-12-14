//! # IPOPT-AD - Blackbox NLP solver using IPOPT and automatic differentiation
//! This crate provides a simple interface to the [IPOPT](https://coin-or.github.io/Ipopt/index.html)
//! nonlinear program (NLP) solver. By evaluating gradients and Hessians using automatic differentiation
//! provided by the [num-dual] crate, it "converts" IPOPT into a blackbox solver without introducing
//! numerical errors through forward differences that could diminish the robustness and accuracy. The
//! only limitation is that the evaluation of the objective and constraints has to be implemented
//! generically for values implementing the [DualNum] trait.
//!
//! ## Example: problem 71 from the Hock-Schittkowsky test-suite
//! This example demonstrates how to solve the problem that is also used as example in the
//! [documentation](https://coin-or.github.io/Ipopt/INTERFACES.html) of IPOPT. Because the
//! objective value and constraints are simple functions of x, the [SimpleADProblem] interface
//! can be used.
//! ```
//! use ipopt_ad::{ADProblem, BasicADProblem, SimpleADProblem};
//! use ipopt_ad::ipopt::Ipopt;
//! use num_dual::DualNum;
//! use approx::assert_relative_eq;
//!
//! struct HS071;
//!
//! impl BasicADProblem<4> for HS071 {
//!     fn bounds(&self) -> ([f64; 4], [f64; 4]) {
//!         ([1.0; 4], [5.0; 4])
//!     }
//!     fn initial_point(&self) -> [f64; 4] {
//!         [1.0, 5.0, 5.0, 1.0]
//!     }
//!     fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
//!         (vec![25.0, 40.0], vec![f64::INFINITY, 40.0])
//!     }
//! }
//!
//! impl SimpleADProblem<4> for HS071 {
//!     fn objective<D: DualNum<f64> + Copy>(&self, x: [D; 4]) -> D {
//!         let [x1, x2, x3, x4] = x;
//!         x1 * x4 * (x1 + x2 + x3) + x3
//!     }
//!     fn constraint_values<D: DualNum<f64> + Copy>(&self, x: [D; 4]) -> Vec<D> {
//!         let [x1, x2, x3, x4] = x;
//!         vec![x1 * x2 * x3 * x4, x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4]
//!     }
//! }
//!
//! let problem = ADProblem::new(HS071);
//! let mut ipopt = Ipopt::new(problem).unwrap();
//! ipopt.set_option("print_level", 0);
//! let res = ipopt.solve();
//! let x = res.solver_data.solution.primal_variables;
//! let x_lit = &[1f64, 4.74299963, 3.82114998, 1.37940829] as &[f64];
//! assert_relative_eq!(x, x_lit, max_relative = 1e-8);
//! ```
//!
//! For more complex NLPs, it can be advantageous to evaluate the objective function and constraints
//! simultaneously, which is possible with the [CachedADProblem] interface. For the HS071 problem,
//! this looks like this:
//! ```
//! # use ipopt_ad::{ADProblem, BasicADProblem, CachedADProblem};
//! # use ipopt_ad::ipopt::Ipopt;
//! # use num_dual::DualNum;
//! # use approx::assert_relative_eq;
//! # use std::convert::Infallible;
//! # struct HS071;
//! # impl BasicADProblem<4> for HS071 {
//! #     fn bounds(&self) -> ([f64; 4], [f64; 4]) {
//! #         ([1.0; 4], [5.0; 4])
//! #     }
//! #     fn initial_point(&self) -> [f64; 4] {
//! #         [1.0, 5.0, 5.0, 1.0]
//! #     }
//! #     fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
//! #         (vec![25.0, 40.0], vec![f64::INFINITY, 40.0])
//! #     }
//! # }
//! impl CachedADProblem<4> for HS071 {
//!     type Error = Infallible;
//!     fn evaluate<D: DualNum<f64> + Copy>(&self, x: [D; 4]) -> Result<(D, Vec<D>), Infallible> {
//!         let [x1, x2, x3, x4] = x;
//!         Ok((
//!             x1 * x4 * (x1 + x2 + x3) + x3,
//!             vec![x1 * x2 * x3 * x4, x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4],
//!         ))
//!     }
//! }
//!
//! let problem = ADProblem::new_cached(HS071).unwrap();
//! let mut ipopt = Ipopt::new(problem).unwrap();
//! ipopt.set_option("print_level", 0);
//! let res = ipopt.solve();
//! let x = res.solver_data.solution.primal_variables;
//! let x_lit = &[1f64, 4.74299963, 3.82114998, 1.37940829] as &[f64];
//! assert_relative_eq!(x, x_lit, max_relative = 1e-8);
//! ```
use ipopt::{BasicProblem, ConstrainedProblem};
use nalgebra::{OVector, SVector, U1};
use num_dual::{
    gradient, hessian, jacobian, try_hessian, Derivative, Dual2Vec, DualNum, DualVec, DualVec64,
    HyperDualVec, HyperDualVec64,
};
use std::cell::RefCell;
use std::convert::Infallible;

pub mod ipopt {
    //! Re-export of all functionalities in [ipopt-rs].
    pub use ipopt::*;
}

/// The basic information that needs to be provided for every optimization problem.
pub trait BasicADProblem<const X: usize> {
    /// Return lower and upper bounds for all variables.
    fn bounds(&self) -> ([f64; X], [f64; X]);

    /// Return an initial guess for all variables.
    fn initial_point(&self) -> [f64; X];

    /// Return the lower and upper bounds for all constraints.
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>);
}

/// The interface for a simple NLP in which the objective function and constraint values are
/// determined separate from each other.
pub trait SimpleADProblem<const X: usize>: BasicADProblem<X> {
    /// Return the objective value for the given x.
    fn objective<D: DualNum<f64> + Copy>(&self, x: [D; X]) -> D;

    /// Return the values of all constraints for the given x.
    fn constraint_values<D: DualNum<f64> + Copy>(&self, x: [D; X]) -> Vec<D>;
}

/// The interface for an NLP in which it is more efficient to evaluate the objective and
/// constraint value in the same function.
///
/// Internally, the results are cached for repeated calls with the same x so that the number
/// of function evaluations is kept to a minimum.
pub trait CachedADProblem<const X: usize>: BasicADProblem<X> {
    /// The error type used in the evaluation.
    type Error;

    /// Return the objective value and values for all constraints simultaneously.
    fn evaluate<D: DualNum<f64> + Copy>(&self, x: [D; X]) -> Result<(D, Vec<D>), Self::Error>;
}

/// Wrapper struct that handles caching and sparsity detection to interface between the blackbox
/// model and IPOPT.
#[expect(clippy::type_complexity)]
pub struct ADProblem<T, const X: usize, const CACHE: bool> {
    problem: T,
    con_jac_row_vec: Vec<i32>,
    con_jac_col_vec: Vec<i32>,
    hess_row_vec: Vec<i32>,
    hess_col_vec: Vec<i32>,
    cache: RefCell<Option<Option<(f64, Vec<f64>)>>>,
    grad_cache: RefCell<Option<Option<([f64; X], Vec<[f64; X]>)>>>,
}

impl<T: BasicADProblem<X>, const X: usize, const CACHE: bool> ADProblem<T, X, CACHE> {
    fn new_impl<
        Err,
        G: Fn(&T, [DualVec64<U1>; X]) -> Result<Vec<DualVec64<U1>>, Err>,
        E: Fn(
            &T,
            [HyperDualVec64<U1, U1>; X],
        ) -> Result<(HyperDualVec64<U1, U1>, Vec<HyperDualVec64<U1, U1>>), Err>,
    >(
        problem: T,
        constraints: G,
        evaluate: E,
    ) -> Result<Self, Err> {
        let x = problem.initial_point();
        let mut con_jac_row_vec = Vec::new();
        let mut con_jac_col_vec = Vec::new();
        for i in 0..x.len() {
            let mut x_dual: [DualVec64<U1>; X] = x.map(DualVec::from_re);
            x_dual[i].eps = Derivative::derivative_generic(U1, U1, 0);
            let con = constraints(&problem, x_dual)?;
            for (j, c) in con.into_iter().enumerate() {
                if c.eps != Derivative::none() {
                    con_jac_row_vec.push(j as i32);
                    con_jac_col_vec.push(i as i32);
                }
            }
        }

        let mut hess_row_vec = Vec::new();
        let mut hess_col_vec = Vec::new();
        for row in 0..x.len() {
            for col in 0..=row {
                let mut x_dual: [HyperDualVec64<U1, U1>; X] = x.map(HyperDualVec::from_re);
                x_dual[row].eps1 = Derivative::derivative_generic(U1, U1, 0);
                x_dual[col].eps2 = Derivative::derivative_generic(U1, U1, 0);
                let (mut f, con) = evaluate(&problem, x_dual)?;
                for g in con {
                    f += g;
                }
                if f.eps1eps2 != Derivative::none() {
                    hess_row_vec.push(row as i32);
                    hess_col_vec.push(col as i32);
                }
            }
        }

        Ok(Self {
            problem,
            con_jac_row_vec,
            con_jac_col_vec,
            hess_row_vec,
            hess_col_vec,
            cache: RefCell::new(None),
            grad_cache: RefCell::new(None),
        })
    }

    fn update_cache(&self, new_x: bool) {
        if new_x {
            *self.cache.borrow_mut() = None;
            *self.grad_cache.borrow_mut() = None;
        }
    }
}

impl<T: SimpleADProblem<X>, const X: usize> ADProblem<T, X, false> {
    /// Initialize an NLP using the `SimpleADProblem` interface.
    pub fn new(problem: T) -> Self {
        Self::new_impl(
            problem,
            |problem, x| Ok::<_, Infallible>(problem.constraint_values(x)),
            |problem, x| Ok((problem.objective(x), problem.constraint_values(x))),
        )
        .unwrap()
    }
}

impl<T: CachedADProblem<X>, const X: usize> ADProblem<T, X, true> {
    /// Initialize an NLP using the `CachedADProblem` interface.
    pub fn new_cached(problem: T) -> Result<Self, T::Error> {
        Self::new_impl(
            problem,
            |problem, x| problem.evaluate(x).map(|(_, g)| g),
            |problem, x| problem.evaluate(x),
        )
    }

    #[expect(clippy::type_complexity)]
    fn evaluate_gradients(&self, x: [f64; X]) -> Result<([f64; X], Vec<[f64; X]>), T::Error> {
        let mut x = SVector::from(x).map(DualVec::from_re);
        let (r, c) = x.shape_generic();
        for (i, xi) in x.iter_mut().enumerate() {
            xi.eps = Derivative::derivative_generic(r, c, i);
        }
        let (f, g) = self.problem.evaluate(x.data.0[0])?;
        Ok((
            f.eps.unwrap_generic(r, c).data.0[0],
            g.into_iter()
                .map(|g| g.eps.unwrap_generic(r, c).data.0[0])
                .collect(),
        ))
    }
}

impl<T: SimpleADProblem<X>, const X: usize> BasicProblem for ADProblem<T, X, false> {
    fn num_variables(&self) -> usize {
        X
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) -> bool {
        let (lbnd, ubnd) = self.problem.bounds();
        x_l.copy_from_slice(&lbnd);
        x_u.copy_from_slice(&ubnd);
        true
    }

    fn initial_point(&self, x: &mut [f64]) -> bool {
        let init = self.problem.initial_point();
        x.copy_from_slice(&init);
        true
    }

    fn objective(&self, x: &[f64], _: bool, obj: &mut f64) -> bool {
        *obj = self.problem.objective(x.try_into().unwrap());
        true
    }

    fn objective_grad(&self, x: &[f64], _: bool, grad_f: &mut [f64]) -> bool {
        let x = SVector::from_column_slice(x);
        let (_, grad) = gradient(|x| self.problem.objective(x.data.0[0]), x);
        grad_f.copy_from_slice(&grad.data.0[0]);
        true
    }
}

impl<T: SimpleADProblem<X>, const X: usize> ConstrainedProblem for ADProblem<T, X, false> {
    fn num_constraints(&self) -> usize {
        self.problem.constraint_bounds().0.len()
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        self.con_jac_col_vec.len()
    }

    fn constraint(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g.copy_from_slice(&self.problem.constraint_values(x.try_into().unwrap()));
        true
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) -> bool {
        let (lbnd, ubnd) = self.problem.constraint_bounds();
        g_l.copy_from_slice(&lbnd);
        g_u.copy_from_slice(&ubnd);
        true
    }

    fn constraint_jacobian_indices(&self, rows: &mut [i32], cols: &mut [i32]) -> bool {
        rows.copy_from_slice(&self.con_jac_row_vec);
        cols.copy_from_slice(&self.con_jac_col_vec);
        true
    }

    fn constraint_jacobian_values(&self, x: &[f64], _: bool, vals: &mut [f64]) -> bool {
        let x = SVector::from_column_slice(x);
        let (_, jac) = jacobian(
            |x| OVector::from(self.problem.constraint_values(x.data.0[0])),
            x,
        );
        for ((v, &r), &c) in vals
            .iter_mut()
            .zip(&self.con_jac_row_vec)
            .zip(&self.con_jac_col_vec)
        {
            *v = jac[(r as usize, c as usize)];
        }
        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        self.hess_col_vec.len()
    }

    fn hessian_indices(&self, rows: &mut [i32], cols: &mut [i32]) -> bool {
        rows.copy_from_slice(&self.hess_row_vec);
        cols.copy_from_slice(&self.hess_col_vec);
        true
    }

    fn hessian_values(
        &self,
        x: &[f64],
        _new_x: bool,
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        let (_, _, hess) = hessian(
            |x| {
                let f = self.problem.objective(x.data.0[0]);
                let g = self.problem.constraint_values(x.data.0[0]);
                f * obj_factor
                    + g.into_iter()
                        .zip(lambda)
                        .map(|(g, &l)| g * l)
                        .sum::<Dual2Vec<_, _, _>>()
            },
            SVector::from_column_slice(x),
        );
        for ((v, &r), &c) in vals
            .iter_mut()
            .zip(&self.hess_row_vec)
            .zip(&self.hess_col_vec)
        {
            *v = hess[(r as usize, c as usize)];
        }
        true
    }
}

impl<T: CachedADProblem<X>, const X: usize> BasicProblem for ADProblem<T, X, true> {
    fn num_variables(&self) -> usize {
        X
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) -> bool {
        let (lbnd, ubnd) = self.problem.bounds();
        x_l.copy_from_slice(&lbnd);
        x_u.copy_from_slice(&ubnd);
        true
    }

    fn initial_point(&self, x: &mut [f64]) -> bool {
        let init = self.problem.initial_point();
        x.copy_from_slice(&init);
        true
    }

    fn objective(&self, x: &[f64], new_x: bool, obj: &mut f64) -> bool {
        self.update_cache(new_x);
        let mut cache = self.cache.borrow_mut();
        let Some((f, _)) = cache.get_or_insert_with(|| {
            let x = SVector::from_column_slice(x);
            self.problem.evaluate(x.data.0[0]).ok()
        }) else {
            return false;
        };
        *obj = *f;
        true
    }

    fn objective_grad(&self, x: &[f64], new_x: bool, grad_f: &mut [f64]) -> bool {
        self.update_cache(new_x);
        let mut cache = self.grad_cache.borrow_mut();
        let Some((grad, _)) = cache.get_or_insert_with(|| {
            let x = SVector::from_column_slice(x);
            self.evaluate_gradients(x.data.0[0]).ok()
        }) else {
            return false;
        };
        grad_f.copy_from_slice(&*grad);
        true
    }
}

impl<T: CachedADProblem<X>, const X: usize> ConstrainedProblem for ADProblem<T, X, true> {
    fn num_constraints(&self) -> usize {
        self.problem.constraint_bounds().0.len()
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        self.con_jac_col_vec.len()
    }

    fn constraint(&self, x: &[f64], new_x: bool, g: &mut [f64]) -> bool {
        self.update_cache(new_x);
        let mut cache = self.cache.borrow_mut();
        let Some((_, con)) = cache.get_or_insert_with(|| {
            let x = SVector::from_column_slice(x);
            self.problem.evaluate(x.data.0[0]).ok()
        }) else {
            return false;
        };
        g.copy_from_slice(&*con);
        true
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) -> bool {
        let (lbnd, ubnd) = self.problem.constraint_bounds();
        g_l.copy_from_slice(&lbnd);
        g_u.copy_from_slice(&ubnd);
        true
    }

    fn constraint_jacobian_indices(&self, rows: &mut [i32], cols: &mut [i32]) -> bool {
        rows.copy_from_slice(&self.con_jac_row_vec);
        cols.copy_from_slice(&self.con_jac_col_vec);
        true
    }

    fn constraint_jacobian_values(&self, x: &[f64], new_x: bool, vals: &mut [f64]) -> bool {
        self.update_cache(new_x);
        let mut cache = self.grad_cache.borrow_mut();
        let Some((_, jac)) = cache.get_or_insert_with(|| {
            let x = SVector::from_column_slice(x);
            self.evaluate_gradients(x.data.0[0]).ok()
        }) else {
            return false;
        };
        for ((v, &r), &c) in vals
            .iter_mut()
            .zip(&self.con_jac_row_vec)
            .zip(&self.con_jac_col_vec)
        {
            *v = jac[r as usize][c as usize];
        }
        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        self.hess_col_vec.len()
    }

    fn hessian_indices(&self, rows: &mut [i32], cols: &mut [i32]) -> bool {
        rows.copy_from_slice(&self.hess_row_vec);
        cols.copy_from_slice(&self.hess_col_vec);
        true
    }

    fn hessian_values(
        &self,
        x: &[f64],
        _new_x: bool,
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        let Ok((_, _, hess)) = try_hessian(
            |x| {
                self.problem.evaluate(x.data.0[0]).map(|(f, g)| {
                    f * obj_factor
                        + g.into_iter()
                            .zip(lambda)
                            .map(|(g, &l)| g * l)
                            .sum::<Dual2Vec<_, _, _>>()
                })
            },
            SVector::from_column_slice(x),
        ) else {
            return false;
        };
        for ((v, &r), &c) in vals
            .iter_mut()
            .zip(&self.hess_row_vec)
            .zip(&self.hess_col_vec)
        {
            *v = hess[(r as usize, c as usize)];
        }
        true
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ipopt::Ipopt;
    use approx::assert_relative_eq;
    use std::convert::Infallible;

    struct HS071;

    impl BasicADProblem<4> for HS071 {
        fn bounds(&self) -> ([f64; 4], [f64; 4]) {
            ([1.0; 4], [5.0; 4])
        }

        fn initial_point(&self) -> [f64; 4] {
            [1.0, 5.0, 5.0, 1.0]
        }

        fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
            (vec![25.0, 40.0], vec![f64::INFINITY, 40.0])
        }
    }

    impl SimpleADProblem<4> for HS071 {
        fn objective<D: DualNum<f64> + Copy>(&self, x: [D; 4]) -> D {
            let [x1, x2, x3, x4] = x;
            x1 * x4 * (x1 + x2 + x3) + x3
        }

        fn constraint_values<D: DualNum<f64> + Copy>(&self, x: [D; 4]) -> Vec<D> {
            let [x1, x2, x3, x4] = x;
            vec![x1 * x2 * x3 * x4, x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4]
        }
    }

    impl CachedADProblem<4> for HS071 {
        type Error = Infallible;
        fn evaluate<D: DualNum<f64> + Copy>(&self, x: [D; 4]) -> Result<(D, Vec<D>), Infallible> {
            let [x1, x2, x3, x4] = x;
            Ok((
                x1 * x4 * (x1 + x2 + x3) + x3,
                vec![x1 * x2 * x3 * x4, x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4],
            ))
        }
    }

    #[test]
    fn test_problem() {
        let problem = ADProblem::new(HS071);
        let mut ipopt = Ipopt::new(problem).unwrap();
        ipopt.set_option("print_level", 0);
        let res = ipopt.solve();
        let x = res.solver_data.solution.primal_variables;
        println!("{x:?}");
        let x_lit = &[1f64, 4.74299963, 3.82114998, 1.37940829] as &[f64];
        assert_relative_eq!(x, x_lit, max_relative = 1e-8);
    }

    #[test]
    fn test_problem_cached() {
        let problem = ADProblem::new_cached(HS071).unwrap();
        let mut ipopt = Ipopt::new(problem).unwrap();
        ipopt.set_option("print_level", 0);
        let res = ipopt.solve();
        let x = res.solver_data.solution.primal_variables;
        println!("{x:?}");
        let x_lit = &[1f64, 4.74299963, 3.82114998, 1.37940829] as &[f64];
        assert_relative_eq!(x, x_lit, max_relative = 1e-8);
    }
}
