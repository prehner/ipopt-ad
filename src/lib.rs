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
//! # #[cfg(feature = "ripopt")]
//! # {
//! use ipopt_ad::{ADProblem, BasicADProblem, SimpleADProblem};
//! use ripopt::solve;
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
//!     fn constraints<D: DualNum<f64> + Copy>(&self, x: [D; 4]) -> Vec<D> {
//!         let [x1, x2, x3, x4] = x;
//!         vec![x1 * x2 * x3 * x4, x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4]
//!     }
//! }
//!
//! let problem = ADProblem::new(HS071);
//! let res = solve(&problem, &Default::default());
//! let x_lit = &[1f64, 4.74299963, 3.82114998, 1.37940829] as &[f64];
//! assert_relative_eq!(&res.x as &[f64], x_lit, max_relative = 1e-7);
//! # }
//! ```
//!
//! For more complex NLPs, it can be advantageous to evaluate the objective function and constraints
//! simultaneously, which is possible with the [CachedADProblem] interface. For the HS071 problem,
//! this looks like this:
//! ```
//! # #[cfg(feature = "ripopt")]
//! # {
//! # use ipopt_ad::{ADProblem, BasicADProblem, CachedADProblem};
//! # use ripopt::solve;
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
//! let res = solve(&problem, &Default::default());
//! let x_lit = &[1f64, 4.74299963, 3.82114998, 1.37940829] as &[f64];
//! assert_relative_eq!(&res.x as &[f64], x_lit, max_relative = 1e-7);
//! # }
//! ```
use nalgebra::{SVector, U1};
use num::FromPrimitive;
use num_dual::{Derivative, DualNum, DualVec, DualVec64, HyperDualVec, HyperDualVec64};
use std::cell::RefCell;
use std::convert::Infallible;

#[cfg(feature = "ipopt")]
pub mod ipopt;
#[cfg(feature = "ripopt")]
pub mod ripopt;

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
    fn constraints<D: DualNum<f64> + Copy>(&self, x: [D; X]) -> Vec<D>;
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
#[allow(unused)]
pub struct ADProblem<T, I, const X: usize, const CACHE: bool> {
    problem: T,
    con_jac_row_vec: Vec<I>,
    con_jac_col_vec: Vec<I>,
    hess_row_vec: Vec<I>,
    hess_col_vec: Vec<I>,
    cache: RefCell<Option<Option<(f64, Vec<f64>)>>>,
    grad_cache: RefCell<Option<Option<([f64; X], Vec<[f64; X]>)>>>,
}

impl<T: BasicADProblem<X>, I: FromPrimitive, const X: usize, const CACHE: bool>
    ADProblem<T, I, X, CACHE>
{
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
                    con_jac_row_vec.push(I::from_usize(j).unwrap());
                    con_jac_col_vec.push(I::from_usize(i).unwrap());
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
                    hess_row_vec.push(I::from_usize(row).unwrap());
                    hess_col_vec.push(I::from_usize(col).unwrap());
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

    #[allow(unused)]
    fn update_cache(&self, new_x: bool) {
        if new_x {
            *self.cache.borrow_mut() = None;
            *self.grad_cache.borrow_mut() = None;
        }
    }
}

impl<T: SimpleADProblem<X>, I: FromPrimitive, const X: usize> ADProblem<T, I, X, false> {
    /// Initialize an NLP using the `SimpleADProblem` interface.
    pub fn new(problem: T) -> Self {
        Self::new_impl(
            problem,
            |problem, x| Ok::<_, Infallible>(problem.constraints(x)),
            |problem, x| Ok((problem.objective(x), problem.constraints(x))),
        )
        .unwrap()
    }
}

impl<T: CachedADProblem<X>, I: FromPrimitive, const X: usize> ADProblem<T, I, X, true> {
    /// Initialize an NLP using the `CachedADProblem` interface.
    pub fn new_cached(problem: T) -> Result<Self, T::Error> {
        Self::new_impl(
            problem,
            |problem, x| problem.evaluate(x).map(|(_, g)| g),
            |problem, x| problem.evaluate(x),
        )
    }

    #[expect(clippy::type_complexity)]
    #[allow(unused)]
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
