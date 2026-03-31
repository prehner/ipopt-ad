use super::{ADProblem, CachedADProblem, SimpleADProblem};
use nalgebra::{OVector, SVector};
use num_dual::{gradient, hessian, jacobian, Dual2Vec};
use ripopt::NlpProblem;

impl<T: SimpleADProblem<X>, const X: usize> NlpProblem for ADProblem<T, usize, X, false> {
    fn num_variables(&self) -> usize {
        X
    }

    fn num_constraints(&self) -> usize {
        self.problem.constraint_bounds().0.len()
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        let (lbnd, ubnd) = self.problem.bounds();
        x_l.copy_from_slice(&lbnd);
        x_u.copy_from_slice(&ubnd);
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        let (lbnd, ubnd) = self.problem.constraint_bounds();
        g_l.copy_from_slice(&lbnd);
        g_u.copy_from_slice(&ubnd);
    }

    fn initial_point(&self, x0: &mut [f64]) {
        let init = self.problem.initial_point();
        x0.copy_from_slice(&init);
    }

    fn objective(&self, x: &[f64], _: bool, obj: &mut f64) -> bool {
        *obj = self.problem.objective(x.try_into().unwrap());
        true
    }

    fn gradient(&self, x: &[f64], _: bool, grad: &mut [f64]) -> bool {
        let x = SVector::from_column_slice(x);
        let (_, g) = gradient(|x| self.problem.objective(x.data.0[0]), &x);
        grad.copy_from_slice(&g.data.0[0]);
        true
    }

    fn constraints(&self, x: &[f64], _: bool, grad: &mut [f64]) -> bool {
        grad.copy_from_slice(&self.problem.constraints(x.try_into().unwrap()));
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.con_jac_row_vec.clone(), self.con_jac_col_vec.clone())
    }

    fn jacobian_values(&self, x: &[f64], _: bool, vals: &mut [f64]) -> bool {
        let x = SVector::from_column_slice(x);
        let (_, jac) = jacobian(|x| OVector::from(self.problem.constraints(x.data.0[0])), &x);
        for ((v, &r), &c) in vals
            .iter_mut()
            .zip(&self.con_jac_row_vec)
            .zip(&self.con_jac_col_vec)
        {
            *v = jac[(r, c)];
        }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.hess_row_vec.clone(), self.hess_col_vec.clone())
    }

    fn hessian_values(
        &self,
        x: &[f64],
        _: bool,
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        let (_, _, hess) = hessian(
            |x| {
                let f = self.problem.objective(x.data.0[0]);
                let g = self.problem.constraints(x.data.0[0]);
                f * obj_factor
                    + g.into_iter()
                        .zip(lambda)
                        .map(|(g, &l)| g * l)
                        .sum::<Dual2Vec<_, _, _>>()
            },
            &SVector::from_column_slice(x),
        );
        for ((v, &r), &c) in vals
            .iter_mut()
            .zip(&self.hess_row_vec)
            .zip(&self.hess_col_vec)
        {
            *v = hess[(r, c)];
        }
        true
    }
}

impl<T: CachedADProblem<X>, const X: usize> NlpProblem for ADProblem<T, usize, X, true> {
    fn num_variables(&self) -> usize {
        X
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        let (lbnd, ubnd) = self.problem.bounds();
        x_l.copy_from_slice(&lbnd);
        x_u.copy_from_slice(&ubnd);
    }

    fn initial_point(&self, x: &mut [f64]) {
        let init = self.problem.initial_point();
        x.copy_from_slice(&init);
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

    fn gradient(&self, x: &[f64], new_x: bool, grad_f: &mut [f64]) -> bool {
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

    fn num_constraints(&self) -> usize {
        self.problem.constraint_bounds().0.len()
    }

    fn constraints(&self, x: &[f64], new_x: bool, g: &mut [f64]) -> bool {
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

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        let (lbnd, ubnd) = self.problem.constraint_bounds();
        g_l.copy_from_slice(&lbnd);
        g_u.copy_from_slice(&ubnd);
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.con_jac_row_vec.clone(), self.con_jac_col_vec.clone())
    }

    fn jacobian_values(&self, x: &[f64], new_x: bool, vals: &mut [f64]) -> bool {
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
            *v = jac[r][c];
        }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.hess_row_vec.clone(), self.hess_col_vec.clone())
    }

    fn hessian_values(
        &self,
        x: &[f64],
        _new_x: bool,
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        let Ok((_, _, hess)) = hessian(
            |x| {
                self.problem.evaluate(x.data.0[0]).map(|(f, g)| {
                    f * obj_factor
                        + g.into_iter()
                            .zip(lambda)
                            .map(|(g, &l)| g * l)
                            .sum::<Dual2Vec<_, _, _>>()
                })
            },
            &SVector::from_column_slice(x),
        ) else {
            return false;
        };
        for ((v, &r), &c) in vals
            .iter_mut()
            .zip(&self.hess_row_vec)
            .zip(&self.hess_col_vec)
        {
            *v = hess[(r, c)];
        }
        true
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::BasicADProblem;
    use approx::assert_relative_eq;
    use num_dual::DualNum;
    use ripopt::solve;
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

        fn constraints<D: DualNum<f64> + Copy>(&self, x: [D; 4]) -> Vec<D> {
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
        let res = solve(&problem, &Default::default());
        // let mut ipopt = Ipopt::new(problem).unwrap();
        // ipopt.set_option("print_level", 0);
        // let res = ipopt.solve();
        let x = &res.x as &[f64];
        println!("{x:?}");
        let x_lit = &[1f64, 4.74299963, 3.82114998, 1.37940829] as &[f64];
        assert_relative_eq!(x, x_lit, max_relative = 1e-7);
    }

    #[test]
    fn test_problem_cached() {
        let problem = ADProblem::new_cached(HS071).unwrap();
        let res = solve(&problem, &Default::default());
        // let mut ipopt = Ipopt::new(problem).unwrap();
        // ipopt.set_option("print_level", 0);
        // let res = ipopt.solve();
        let x = &res.x as &[f64];
        println!("{x:?}");
        let x_lit = &[1f64, 4.74299963, 3.82114998, 1.37940829] as &[f64];
        assert_relative_eq!(x, x_lit, max_relative = 1e-7);
    }
}
