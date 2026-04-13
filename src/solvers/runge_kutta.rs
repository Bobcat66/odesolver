// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SVector;

// This file implements explicit runge-kutta steppers
pub fn euler_step<F, const D: usize>(ode: &F, y0: &SVector<f64,D>, t0: f64, h: f64) -> SVector<f64,D> 
    where F: Fn(&SVector<f64,D>,f64) -> SVector<f64,D>
{
    y0 + (h * ode(y0,t0))
}

pub fn rk4_step<F, const D: usize>(ode: &F, y0: &SVector<f64,D>, t0: f64, h: f64) -> SVector<f64,D>
    where F: Fn(&SVector<f64,D>,f64) -> SVector<f64,D>
{
    let half_h = h/2.0;
    let half_t = t0 + half_h;
    let k1 = ode(y0, t0);
    let k2 = ode(&(y0 + (k1 * half_h)), half_t);
    let k3 = ode(&(y0 + (k2 * half_h)), half_t);
    let k4 = ode(&(y0 + (k3 * h)), t0 + h);
    y0 + ((h / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4))
}
