// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SVector;

use crate::solvers::{DenseInterpolant, DenseOutput};

pub trait RKController<const E: usize> {

    type Config: Default + Copy;
    // returns the next timestep and whether or not the step should be accepted. o should be a buffer containing output, with y1 in o[0] and errors in o[1..n]
    fn get_next_step<const D: usize>(y1: &SVector<f64, D>, e: &[SVector<f64,D>; E], y0: &SVector<f64,D>, h: f64, cfg: &Self::Config) -> (bool, f64);
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SVector<f64,D>, f0: &SVector<f64,D>, cfg: &Self::Config) -> f64
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;
}
pub trait RKInterpolator<const S: usize> {

    type InterpolantType<const D: usize>: DenseInterpolant<D>;

    fn interpolate_stage<F, const D: usize>(ode: &F, t0: f64, t1: f64, y0: &SVector<f64,D>, y1: &SVector<f64,D>, stage: &[SVector<f64,D>; S]) -> Self::InterpolantType<D>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;
    fn interpolate_dense<F, const D: usize>(ode: &F, points: &Vec<(f64,SVector<f64,D>)>, stages: &Vec<[SVector<f64,D>; S]>) -> DenseOutput<Self::InterpolantType<D>,D>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        let steps = stages.len();
        let mut segments: Vec<Self::InterpolantType<D>> = Vec::new();
        for i in 0..(steps-1) {
            segments.push(
                Self::interpolate_stage(ode,points[i].0,points[i + 1].0, &points[i].1, &points[i+1].1, &stages[i])
            );
        }

        DenseOutput::<Self::InterpolantType<D>,D>::new(segments)
    }
}

// S is number of stages, E is the number of errors. B[0] is always the main solution, while B[1..n] are embedded solutions
pub trait RKMethod<const S: usize, const E: usize> {
    
    type Interpolator: RKInterpolator<S>;
    type Controller: RKController<E>;
    
    // Butcher's tableau
    const C: [f64; S]; // time coefficients
    const A: [[f64; S]; S]; // stage coefficients
    const B: [f64; S]; // weights
    const E_B: [[f64; S]; E]; // error weights

    const ORDER: usize; // order of solution
    const FSAL: bool;
    const ERR_ORDER: usize; // order of error estimator
}

pub type MethodController<M,const S: usize, const E: usize> = <M as RKMethod<S,E>>::Controller;
pub type MethodConfig<M,const S: usize, const E: usize> = <<M as RKMethod<S,E>>::Controller as RKController<E>>::Config;
pub type MethodInterpolator<M,const S: usize, const E: usize> = <M as RKMethod<S,E>>::Interpolator;
pub type MethodInterpolant<M,const S: usize, const E: usize, const D: usize> = <<M as RKMethod<S,E>>::Interpolator as RKInterpolator<S>>::InterpolantType<D>;