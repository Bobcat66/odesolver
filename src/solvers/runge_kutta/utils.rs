// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
use nalgebra::SVector;

// Normalizes vec using RMS, scaled to y
pub fn scale_norm<const D: usize>(vec: &SVector<f64,D>, y: &SVector<f64,D>, atol: f64, rtol: f64) -> f64
{
    let mut vec_norm = 0.0;
    for i in 0..D {
        let scale = atol + rtol * y[i].abs();
        vec_norm += (vec[i]/scale).powi(2);
    }
    (vec_norm / D as f64).sqrt()
}

// Returns initial guess for h
pub fn guess_timestep<F, const D: usize>(ode: &F, y0: &SVector<f64,D>, t0: f64, atol: f64, rtol: f64) -> f64
    where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
{
    let d0 = scale_norm(y0, y0, atol, rtol);
    let ydot_0 = ode(t0,y0);
    let d1 = scale_norm(&ydot_0,y0, atol, rtol);

    // Create initial guess
    let h0: f64 =  if d0 < 1e-5 || d1 < 1e-5 {1e-6} else {0.01 * d0 / d1};

    // Take euler step to estimate curvature
    let y1 = y0 + h0 * ydot_0;
    let ydot_1 = ode(t0+h0,&y1);
    let d2 = scale_norm(&(ydot_1 - ydot_0), y0, atol, rtol);
    
    (h0 * 100.0).min((0.01/d1.max(d2)).powf(0.2))
}