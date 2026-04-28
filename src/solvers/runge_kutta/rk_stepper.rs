

// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::marker::PhantomData;

use nalgebra::{ArrayStorage, SMatrix, SVector};

use crate::{solvers::runge_kutta::rk_common::PRKMethod};
use std::ops::{MulAssign, AddAssign};

pub struct PRKStepper<Method,const P: usize, const S: usize, const E: usize> 
    where Method: PRKMethod<P,S,E> 
{
    _marker: PhantomData<Method>
}

const fn compute_time_vectors<const P: usize, const S: usize>(c: [[f64; S]; P]) -> [SVector<f64, P>; S]
{
    let mut c_rawvecs = [[0.0; P]; S];
    let (mut i, mut j) = (0, 0);
    while i < S {
        while j < P {
            c_rawvecs[i][j] = c[j][i];
            j += 1;
        }
        i += 1;
        j = 0;
    }
    i = 0;
    let mut c_vecs = [SVector::<f64,P>::from_data(ArrayStorage([[0.0; P]; 1])); S];
    while i < S {
        c_vecs[i] = SVector::<f64,P>::from_data(ArrayStorage([c_rawvecs[i]; 1]));
        i += 1;
    }
    c_vecs
}

impl<Method, const P: usize, const S: usize, const E: usize> PRKStepper<Method,P, S, E> 
    where Method: PRKMethod<P,S,E> 
{
    const C_VECS: [SVector<f64, P>; S] = compute_time_vectors(Method::C);

    pub fn step_partitioned<F, const D: usize>(
        k: &mut [SMatrix<f64,D,P>; S],
        e: &mut [SMatrix<f64,D,P>; E],
        ode: &F,
        t_n: f64,
        y_n: &SMatrix<f64,D,P>,
        f_n: &SMatrix<f64,D,P>,
        h: f64
    ) -> SMatrix<f64,D,P>
        where F: Fn(&SVector<f64,P>,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
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
            k[i] = ode(&(Self::C_VECS[i] * h).add_scalar(t_n),&(y_n + dy));
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

    // Part is which partition of the tableau to use
    pub fn step<F, const Part: usize, const D: usize>(
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
                dy += k[j] * h * Method::A[Part][i][j];
            }
            k[i] = ode(Method::C[Part][i] * h + t_n,&(y_n + dy));
        }
        let mut o = SVector::<f64,D>::zeros();
        e.fill(SVector::<f64,D>::zeros());
        for j in 0..S {
            let kj = k[j];
            o += kj * Method::B[Part][j] * h;
            for i in 0..E {
                e[i] += kj * Method::E_B[Part][i][j] * h;
            }
        }
        o += y_n;
        o
    }
}



