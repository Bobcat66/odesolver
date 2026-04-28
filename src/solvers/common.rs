// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::{ComplexField, SMatrix, SVector};


// Returns initial guess for h. shamelessly stolen from scipy
pub fn select_initial_timestep<T, F, TimeType, const D: usize, const P: usize>(ode: &F, time_convert: &T, y0: &SMatrix<f64,D,P>, t0: f64, f0: &SMatrix<f64,D,P>, atol: f64, rtol: f64, err_order: usize) -> f64
    where F: Fn(TimeType,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>,
    T: Fn(f64) -> TimeType
{   
    let scale = (y0.abs() * rtol).add_scalar(atol);
    let d0 =norm(y0, &scale);
    let d1= norm(f0, &scale);

    // Create initial guess
    let h0 =  if d0 < 1e-5 || d1 < 1e-5 {1e-6} else {0.01 * d0 / d1};
    // Take euler step to estimate curvature
    let y1 = y0 + f0 * h0;
    let f1 = ode(time_convert(t0 + h0), &y1);
    let d2 = norm(&(f1 - f0), &scale)/h0;
    
    (h0 * 100.0).min((0.01 / d1.max(d2)).powf(1.0/(err_order as f64 + 1.0)))
}

pub fn norm<const R: usize,const C: usize>(mat: &SMatrix<f64,R,C>,scale: &SMatrix<f64,R,C>) -> f64
{
    (mat.component_div(&scale).norm_squared() / ((C * R) as f64)).sqrt()
}

pub fn convert_t<const P: usize>(t: f64) -> SVector<f64,P> {
    SVector::<f64,P>::repeat(t)
}
