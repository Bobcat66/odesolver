// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::{ComplexField, SMatrix, SVector};

use crate::{fields::Real, solvers::{DenseInterpolant, DenseOutput}};

// T is the type of the time variable. Usually f64, but can be a vector of f64 for partitioned methods and additive methods.
pub trait RKController<T, const E: usize, const P: usize>
{

    type Config: Default + Copy;
    // returns the next timestep and whether or not the step should be accepted. o should be a buffer containing output, with y1 in o[0] and errors in o[1..n]
    fn get_next_step<const D: usize>(y1: &SMatrix<f64,D,P>, e: &[SMatrix<f64,D,P>; E], y0: &SMatrix<f64,D,P>, h: f64, cfg: &Self::Config) -> (bool, f64);
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SMatrix<f64,D,P>, f0: &SMatrix<f64,D,P>, cfg: &Self::Config) -> f64
        where F: Fn(T,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>;
}
pub trait RKInterpolator<T, const S: usize, const P: usize> 
{

    type InterpolantType<const D: usize>: DenseInterpolant<D,P>;

    fn interpolate_stage<F, const D: usize>(ode: &F, t0: f64, t1: f64, y0: &SMatrix<f64,D,P>, y1: &SMatrix<f64,D,P>, stage: &[SMatrix<f64,D,P>; S]) -> Self::InterpolantType<D>
        where F: Fn(T,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>;
    // Points are always evaluated at a single time, so the time variable is always f64, even for multi-tableaux methods.
    fn interpolate_dense<F, const D: usize>(ode: &F, points: &Vec<(f64,SMatrix<f64,D,P>)>, stages: &Vec<[SMatrix<f64,D,P>; S]>) -> DenseOutput<Self::InterpolantType<D>,D,P>
        where F: Fn(T,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
    {
        let steps = stages.len();
        let mut segments: Vec<Self::InterpolantType<D>> = Vec::new();
        for i in 0..(steps-1) {
            segments.push(
                Self::interpolate_stage(ode,points[i].0,points[i + 1].0, &points[i].1, &points[i+1].1, &stages[i])
            );
        }

        DenseOutput::<Self::InterpolantType<D>,D,P>::new(segments)
    }
}

pub trait RKStepper<Tableau, T, const N: usize, const S: usize, const E: usize, const P: usize>
    where Tableau: RKTableau<T,N,S,E,P>
{
    fn step<F, const D: usize>(k: &mut [SMatrix<f64,D,P>; S], e: &mut [SMatrix<f64,D,P>; E], ode: &F, t_n: f64, y_n: &SMatrix<f64,D,P>, f_n: &SMatrix<f64,D,P>, h: f64) -> SMatrix<f64,D,P>
        where F: Fn(T,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>;
}

// N is the number of tableaux, P is the number of state partitions, S is number of stages, E is the number of errors. B[0] is always the main solution, while B[1..n] are embedded solutions
pub trait RKTableau<T,const N: usize, const S: usize, const E: usize, const P: usize>
{
    // Butcher's tableau
    const C: [[f64; S]; N]; // time coefficients
    const A: [[[f64; S]; S]; N]; // stage coefficients
    const B: [[f64; S]; N]; // weights
    const E_B: [[[f64; S]; E]; N]; // error weights

    const ORDER: usize; // order of solution
    const FSAL: bool;
    const ERR_ORDER: usize; // order of error estimator
}