// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::{SMatrix, SVector};

use crate::solvers::runge_kutta::rk_common::{RKStepper, RKTableau};

pub struct ExplicitStepper<Tableau, const S: usize, const E: usize>
    where Tableau: RKTableau<f64,1,S,E,1> 
{
    _marker: std::marker::PhantomData<Tableau>
}

impl<Tableau, const S: usize, const E: usize> RKStepper<Tableau,f64,1,S,E,1> for ExplicitStepper<Tableau,S,E>
    where Tableau: RKTableau<f64,1,S,E,1> 
{
    fn step<F, const D: usize>(
        k: &mut [SVector<f64,D>; S],
        e: &mut [SVector<f64,D>; E],
        ode: &F,
        t_n: f64,
        y_n: &SVector<f64,D>,
        f_n: &SVector<f64,D>,
        h: f64
    ) -> SVector<f64,D>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        k[0] = *f_n;
        for i in 1..S {
            let mut dy = SVector::<f64,D>::zeros();
            for j in 0..i {
                dy += k[j] * h * Tableau::A[0][i][j];
            }
            k[i] = ode(Tableau::C[0][i] * h + t_n,&(y_n + dy));
        }
        let mut o = SVector::<f64,D>::zeros();
        e.fill(SVector::<f64,D>::zeros());
        for j in 0..S {
            let kj = k[j];
            o += kj * Tableau::B[0][j] * h;
            for i in 0..E {
                e[i] += kj * Tableau::E_B[0][i][j] * h;
            }
        }
        o += y_n;
        o
    }
}