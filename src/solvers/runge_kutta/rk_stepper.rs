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

const fn compute_e<const S: usize, const O: usize>(b: [[f64; S]; O]) -> [[f64; S]; O] 
{
    let mut e = [[0.0; S]; O];
    let mut i = 1;
    while i < O {
        let mut j = 0;
        while j < S {
            e[i][j] = b[0][j] - b[i][j];
            j += 1;
        }
        i += 1;
    }
    e
}

impl<Method, const S: usize, const O: usize> RKStepper<Method,S,O>
    where Method: RKMethod<S,O> 
{
    const E: [[f64; S]; O] = compute_e(Method::B);
}

impl<Method, const S: usize, const O: usize> RKStepper<Method, S, O> 
    where Method: RKMethod<S,O> 
{
    pub fn step<F, const D: usize>(
        k: &mut [SVector<f64,D>; S],
        o: &mut [SVector<f64,D>; O],
        ode: &F,
        t_n: f64,
        y_n: &SVector<f64,D>,
        f_n: &SVector<f64,D>,
        h: f64
    ) -> ()
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>,
    {
        k[0] = *f_n;
        for i in 1..S {
            let mut dy = SVector::<f64, D>::zeros();
            for j in 0..i {
                dy += k[j] * Method::A[i][j];
            }
            dy *= h;
            k[i] = ode(t_n + h * Method::C[i],&(y_n + dy));
        }
        o.fill(SVector::zeros());
        for j in 0..S {
            let kj = k[j];
            o[0] += kj * Method::B[0][j];
            for i in 1..O {
                o[i] += kj * Self::E[i][j] * h;
            }
        }
        o[0] *= h;
        o[0] += y_n;
    }
}

