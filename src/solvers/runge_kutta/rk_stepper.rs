// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::marker::PhantomData;

use nalgebra::SVector;

// S is number of stages, O is the number of solutions. B[0] is always the main solution, while B[1..n] are embedded solutions
pub trait ButchersTableau<const S: usize, const O: usize> {
    const C: [f64; S]; // time coefficients
    const A: [[f64; S]; S]; // stage coefficients
    const B: [[f64; S]; O]; // weights
    const ORDERS: [usize; O]; // order of solutions
    const FSAL: bool;
}

pub struct RKStepper<Tableau, const S: usize, const O: usize> 
    where Tableau: ButchersTableau<S,O> 
{
    _marker: PhantomData<Tableau>
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

impl<Tableau, const S: usize, const O: usize> RKStepper<Tableau,S,O>
    where Tableau: ButchersTableau<S,O> 
{
    const E: [[f64; S]; O] = compute_e(Tableau::B);
}

impl<Tableau, const S: usize, const O: usize> RKStepper<Tableau, S, O> 
    where Tableau: ButchersTableau<S,O> 
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
                dy += k[j] * Tableau::A[i][j];
            }
            dy *= h;
            k[i] = ode(t_n + h * Tableau::C[i],&(y_n + dy));
        }
        o.fill(SVector::zeros());
        for j in 0..S {
            let kj = k[j];
            o[0] += kj * Tableau::B[0][j];
            for i in 1..O {
                o[i] += kj * Self::E[i][j] * h;
            }
        }
        o[0] *= h;
        o[0] += y_n;
    }
}

