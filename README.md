# IPOPT-AD - Blackbox NLP solver using IPOPT and automatic differentiation
This crate provides a simple interface to the [IPOPT](https://coin-or.github.io/Ipopt/index.html)
nonlinear program (NLP) solver. By evaluating gradients and Hessians using automatic differentiation
provided by the [num-dual](https://github.com/itt-ustutt/num-dual) crate, it "converts" IPOPT into a blackbox solver without introducing
numerical errors through forward differences that could diminish the robustness and accuracy. The
only limitation is that the evaluation of the objective and constraints has to be implemented
generically for values implementing the [DualNum](https://docs.rs/num-dual/latest/num_dual/trait.DualNum.html) trait.

## Example: problem 71 from the Hock-Schittkowsky test-suite
This example demonstrates how to solve the problem that is also used as example in the
[documentation](https://coin-or.github.io/Ipopt/INTERFACES.html) of IPOPT. Because the
objective value and constraints are simple functions of x, the `SimpleADProblem` interface
can be used.
```rust
use ipopt_ad::{ADProblem, BasicADProblem, SimpleADProblem};
use ipopt_ad::ipopt::Ipopt;
use num_dual::DualNum;
use approx::assert_relative_eq;

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

let problem = ADProblem::new(HS071);
let mut ipopt = Ipopt::new(problem).unwrap();
ipopt.set_option("print_level", 0);
let res = ipopt.solve();
let x = res.solver_data.solution.primal_variables;
let x_lit = &[1f64, 4.74299963, 3.82114998, 1.37940829] as &[f64];
assert_relative_eq!(x, x_lit, max_relative = 1e-8);
```

For more complex NLPs, it can be advantageous to evaluate the objective function and constraints
simultaneously, which is possible with the `CachedADProblem` interface. For the HS071 problem,
this looks like this:
```rust
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

let problem = ADProblem::new_cached(HS071).unwrap();
let mut ipopt = Ipopt::new(problem).unwrap();
ipopt.set_option("print_level", 0);
let res = ipopt.solve();
let x = res.solver_data.solution.primal_variables;
let x_lit = &[1f64, 4.74299963, 3.82114998, 1.37940829] as &[f64];
assert_relative_eq!(x, x_lit, max_relative = 1e-8);
```