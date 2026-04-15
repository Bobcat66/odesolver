// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SVector;

pub trait RKController<const O: usize> {
    type Config: Default;
    // returns the next timestep and whether or not the step should be accepted. o should be a buffer containing output, with y1 in o[0] and errors in o[1..n]
    fn get_next_step<const D: usize>(o: &[SVector<f64,D>; O], y0: &SVector<f64,D>, t1: f64, t0: f64, h: f64, t_end: f64, cfg: &Self::Config) -> (bool, f64);
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SVector<f64,D>, f0: &SVector<f64,D>, cfg: &Self::Config) -> f64
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;
}