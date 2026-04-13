// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SVector;

pub fn euler_step_impl<F, const D: usize>(ode: &F, y0: &SVector<f64,D>, t0: f64, h: f64) -> SVector<f64,D> 
    where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
{
    y0 + (h * ode(t0, y0))
}

pub fn rk4_step_impl<F, const D: usize>(ode: &F, y0: &SVector<f64,D>, t0: f64, h: f64) -> SVector<f64,D>
    where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
{
    let half_h = h/2.0;
    let half_t = t0 + half_h;
    let k1 = ode(t0, y0);
    let k2 = ode(half_t, &(y0 + (k1 * half_h)));
    let k3 = ode(half_t, &(y0 + (k2 * half_h)));
    let k4 = ode(t0 + h, &(y0 + (k3 * h)));
    y0 + ((h / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4))
}

// Returns result, error, and first step of next stage
/* 
    fn step_old<F>(&self, ode: &F, y0: &SVector<f64,D>, k1: &SVector<f64,D>, t0: f64, h: f64) -> (SVector<f64,D>,SVector<f64,D>,SVector<f64,D>) 
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        
        let k2 = ode(t0 + h * (1.0/5.0), &(y0 + h * (k1 * (1.0/5.0))));
        let k3 = ode(t0 + h * (3.0/10.0), &(y0 + h * (k1 * (3.0/40.0) + k2 * (9.0/40.0))));
        let k4 = ode(t0 + h * (4.0/5.0), &(y0 + h * (k1 * (44.0/45.0) + k2 * (-56.0/15.0) + k3 * (32.0/9.0))));
        let k5 = ode(t0 + h * (8.0/9.0), &(y0 + h * (k1 * (19372.0/6561.0) + k2 * (-25360.0/2187.0) + k3 * (64448.0/6561.0) + k4 * (-212.0/729.0))));
        let k6 = ode(t0 + h,&(y0 + h * (k1 * (9017.0/3168.0) + k2 * (-355.0/33.0) + k3 * (46732.0/5247.0) + k4 * (49.0/176.0) + k5 * (-5103.0/18656.0))));
        let k7 = ode(t0 + h,&(y0 + h * (k1 * (35.0/384.0) + k3 * (500.0/1113.0) + k4 * (125.0/192.0) + k5 * (-2187.0/6784.0) + k6 * (11.0/84.0))));
        let y1 = y0 + h * (k1 * (35.0/384.0) + k3 * (500.0/1113.0) + k4 * (125.0/192.0) + k5 * (-2187.0/6784.0) + k6 * (11.0/84.0));
        let e1 = h * (k1 * (35.0/384.0 - 5179.0/57600.0) + k3 * (500.0/1113.0 - 7571.0/16695.0) + k4 * (125.0/192.0 - 393.0/640.0) + k5 * (-2187.0/6784.0 + 92097.0/339200.0) + k6 * (11.0/84.0 - 187.0/2100.0) + k7 * (-1.0/40.0));
        (y1,e1,k7)
    }
*/
