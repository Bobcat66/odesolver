// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::{ComplexField, SMatrix};

use crate::{fields::Real, solvers::{DenseInterpolant, DenseOutput}};

pub trait PRKController<const P: usize, const E: usize>
{

    type Config: Default + Copy;
    // returns the next timestep and whether or not the step should be accepted. o should be a buffer containing output, with y1 in o[0] and errors in o[1..n]
    fn get_next_step<const D: usize>(y1: &SMatrix<f64,D,P>, e: &[SMatrix<f64,D,P>; E], y0: &SMatrix<f64,D,P>, h: f64, cfg: &Self::Config) -> (bool, f64);
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SMatrix<f64,D,P>, f0: &SMatrix<f64,D,P>, cfg: &Self::Config) -> f64
        where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>;
}
pub trait PRKInterpolator<const P: usize, const S: usize> 
{

    type InterpolantType<const D: usize>: DenseInterpolant<P,D>;

    fn interpolate_stage<F, const D: usize>(ode: &F, t0: f64, t1: f64, y0: &SMatrix<f64,D,P>, y1: &SMatrix<f64,D,P>, stage: &[SMatrix<f64,D,P>; S]) -> Self::InterpolantType<D>
        where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>;
    fn interpolate_dense<F, const D: usize>(ode: &F, points: &Vec<(f64,SMatrix<f64,D,P>)>, stages: &Vec<[SMatrix<f64,D,P>; S]>) -> DenseOutput<Self::InterpolantType<D>,P,D>
        where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
    {
        let steps = stages.len();
        let mut segments: Vec<Self::InterpolantType<D>> = Vec::new();
        for i in 0..(steps-1) {
            segments.push(
                Self::interpolate_stage(ode,points[i].0.clone(),points[i + 1].0.clone(), &points[i].1, &points[i+1].1, &stages[i])
            );
        }

        DenseOutput::<Self::InterpolantType<D>,P,D>::new(segments)
    }
}

// P is the number of partitions, S is number of stages, E is the number of errors. B[0] is always the main solution, while B[1..n] are embedded solutions
pub trait PRKMethod<const P: usize, const S: usize, const E: usize>
{
    
    type Interpolator: PRKInterpolator<P, S>;
    type Controller: PRKController<P, E>;
    
    // Butcher's tableau
    const C: [f64; S]; // time coefficients
    const A: [[[f64; S]; S]; P]; // stage coefficients
    const B: [[f64; S]; P]; // weights
    const E_B: [[[f64; S]; E]; P]; // error weights

    const ORDER: usize; // order of solution
    const FSAL: bool;
    const ERR_ORDER: usize; // order of error estimator
}

pub type PRKMethodController<M,const P: usize, const S: usize, const E: usize> = <M as PRKMethod<P,S,E>>::Controller;
pub type PRKMethodConfig<M,const P: usize, const S: usize, const E: usize> = <<M as PRKMethod<P,S,E>>::Controller as PRKController<P,E>>::Config;
pub type PRKMethodInterpolator<M,const P: usize, const S: usize, const E: usize> = <M as PRKMethod<P,S,E>>::Interpolator;
pub type PRKMethodInterpolant<M,const P: usize, const S: usize, const E: usize, const D: usize> = <<M as PRKMethod<P,S,E>>::Interpolator as PRKInterpolator<P,S>>::InterpolantType<D>;