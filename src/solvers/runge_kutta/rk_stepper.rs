// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::marker::PhantomData;

use nalgebra::SVector;

use crate::solvers::runge_kutta::rk_method::RKMethod;

pub struct RKStepper<Method, const S: usize, const O: usize> 
    where Method: RKMethod<S,O> 
{
    _marker: PhantomData<Method>
}

impl<Method, const S: usize, const E: usize> RKStepper<Method, S, E> 
    where Method: RKMethod<S,E> 
{
    pub fn step<F, const D: usize>(
        k: &mut [SVector<f64,D>; S],
        e: &mut [SVector<f64,D>; E],
        ode: &F,
        t_n: f64,
        y_n: &SVector<f64,D>,
        f_n: &SVector<f64,D>,
        h: f64
    ) -> SVector<f64,D>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>,
    {
        k[0] = *f_n;
        for i in 1..S {
            let mut dy = SVector::<f64, D>::zeros();
            for j in 0..i {
                dy += k[j] * Method::A[i][j] * h;
            }
            k[i] = ode(t_n + h * Method::C[i],&(y_n + dy));
        }
        let mut o = SVector::<f64,D>::zeros();
        e.fill(SVector::<f64, D>::zeros());
        for j in 0..S {
            let kj = k[j];
            o += kj * Method::B[j];
            for i in 0..E {
                e[i] += kj * Method::E_B[i][j] * h;
            }
        }
        o *= h;
        o += y_n;
        o
    }
}

