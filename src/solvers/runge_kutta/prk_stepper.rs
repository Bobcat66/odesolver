// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::marker::PhantomData;

use nalgebra::{ComplexField, SMatrix};

use crate::{solvers::runge_kutta::prk_method::PRKMethod};
use std::ops::{MulAssign, AddAssign};

pub struct PRKStepper<Method,const P: usize, const S: usize, const E: usize> 
    where Method: PRKMethod<P,S,E> 
{
    _marker: PhantomData<Method>
}

impl<Method, const P: usize, const S: usize, const E: usize> PRKStepper<Method,P, S, E> 
    where Method: PRKMethod<P,S,E> 
{
    pub fn step<F, const D: usize>(
        k: &mut [SMatrix<f64,D,P>; S],
        e: &mut [SMatrix<f64,D,P>; E],
        ode: &F,
        t_n: f64,
        y_n: &SMatrix<f64,D,P>,
        f_n: &SMatrix<f64,D,P>,
        h: f64
    ) -> SMatrix<f64,D,P>
        where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
    {
        k[0] = *f_n;
        for i in 1..S {
            let mut dy = SMatrix::<f64,D,P>::zeros();
            for j in 0..i {
                let mut tmp = k[j];
                for p in 0..P {
                    tmp.column_mut(p).mul_assign(h * Method::A[p][i][j]);
                }
                dy += tmp;
            }
            k[i] = ode(t_n + h * Method::C[i],&(y_n + dy));
        }
        let mut o = SMatrix::<f64,D,P>::zeros();
        e.fill(SMatrix::<f64,D,P>::zeros());
        for j in 0..S {
            let kj = &k[j];
            for p in 0..P {
                o.column_mut(p).add_assign(kj.column(p) * Method::B[p][j] * h);
                for i in 0..E {
                    e[i].column_mut(p).add_assign(kj.column(p) * Method::E_B[p][i][j] * h);
                }
            }
        }
        o += y_n;
        o
    }
}

