// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SVector;

#[derive(Clone, Copy)]
pub struct PointState<const D: usize> {
    pub t: f64, // time (t)
    pub y: SVector<f64, D>, // state vector (y)
    pub f: SVector<f64, D>, // state derivative (dy/dt)
}

impl<const D: usize> Default for PointState<D> {
    fn default() -> Self {
        Self {
            t: 0.0,
            y: SVector::<f64,D>::zeros(),
            f: SVector::<f64,D>::zeros()
        }
    }
}

pub struct SolverState<const D: usize> {
    pub point: PointState<D>,
    pub h: f64,
    pub points: Vec<(f64,SVector<f64,D>)>,
    pub steps: usize
}

// Returns initial guess for h. shamelessly stolen from scipy
pub fn select_initial_timestep<F, const D: usize>(ode: &F, y0: &SVector<f64,D>, t0: f64, f0: &SVector<f64,D>, atol: f64, rtol: f64, err_order: usize) -> f64
    where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
{   
    let scale = (y0.abs() * rtol).add_scalar(atol);
    let d0 = y0.component_div(&scale).norm();
    let d1 = f0.component_div(&scale).norm();

    // Create initial guess
    let h0: f64 =  if d0 < 1e-5 || d1 < 1e-5 {1e-6} else {0.01 * d0 / d1};

    // Take euler step to estimate curvature
    let y1 = y0 + h0 * f0;
    let f1 = ode(t0 + h0, &y1);
    let d2 = ((f1 - f0).component_div(&scale) / h0).norm();
    
    (h0 * 100.0).min((0.01 / d1.max(d2)).powf(1.0/(err_order as f64 + 1.0)))
}