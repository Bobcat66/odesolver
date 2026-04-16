// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SVector;

use crate::solvers::dense::DenseOutput;

pub trait RKController<const O: usize> {

    type Config: Default;
    // returns the next timestep and whether or not the step should be accepted. o should be a buffer containing output, with y1 in o[0] and errors in o[1..n]
    fn get_next_step<const D: usize>(o: &[SVector<f64,D>; O], y0: &SVector<f64,D>, t0: f64, h: f64, t_end: f64, cfg: &Self::Config) -> (bool, f64);
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SVector<f64,D>, f0: &SVector<f64,D>, cfg: &Self::Config) -> f64
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;
}
pub trait RKInterpolator<const S: usize> {
    fn interpolate_dense<const D: usize>(points: &Vec<(f64,SVector<f64,D>)>, stages: &Vec<[SVector<f64,D>; S]>) -> DenseOutput<D>;
}

// S is number of stages, O is the number of solutions. B[0] is always the main solution, while B[1..n] are embedded solutions
pub trait RKMethod<const S: usize, const O: usize> {
    
    type Interpolator: RKInterpolator<S>;
    type Controller: RKController<O>;
    
    // Butcher's tableau
    const C: [f64; S]; // time coefficients
    const A: [[f64; S]; S]; // stage coefficients
    const B: [[f64; S]; O]; // weights

    const ORDERS: [usize; O]; // order of solutions
    const FSAL: bool;
}

pub type MethodController<M,const S: usize, const O: usize> = <M as RKMethod<S,O>>::Controller;
pub type MethodConfig<M,const S: usize, const O: usize> = <<M as RKMethod<S,O>>::Controller as RKController<O>>::Config;
pub type MethodInterpolator<M,const S: usize, const O: usize> = <M as RKMethod<S,O>>::Interpolator;