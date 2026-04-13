// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use crate::solvers::solver::Solver;
use nalgebra::SVector;
pub struct DormandPrinceSolver<const D: usize> 
{
    pub atol: f64, // absolute tolerance
    pub rtol: f64, // normalized tolerance
    pub safety: f64, // safety value to reduce overshoot
    pub min_clamp: f64, // minimum timestep clamp
    pub max_clamp: f64 // maximum timestep clamp
}

impl<const D: usize> Default for DormandPrinceSolver<D> {
    fn default() -> Self {
        Self {
            atol: 1e-3,
            rtol: 1e-6,
            safety: 0.9,
            min_clamp: 1e-12,
            max_clamp: f64::MAX,
        }
    }
}

impl<const D: usize> DormandPrinceSolver<D> {
    // Returns result, error, and first step of next stage
    fn step<F>(&self, ode: &F, y0: &SVector<f64,D>, k1: &SVector<f64,D>, t0: f64, h: f64) -> (SVector<f64,D>,SVector<f64,D>,SVector<f64,D>) 
        where F: Fn(&SVector<f64,D>,f64) -> SVector<f64,D>
    {
        
        let k2 = ode(&(y0 + h * (k1 * (1.0/5.0))),t0 + h * (1.0/5.0));
        let k3 = ode(&(y0 + h * (k1 * (3.0/40.0) + k2 * (9.0/40.0))),t0 + h * (3.0/10.0));
        let k4 = ode(&(y0 + h * (k1 * (44.0/45.0) + k2 * (-56.0/15.0) + k3 * (32.0/9.0))), t0 + h * (4.0/5.0));
        let k5 = ode(&(y0 + h * (k1 * (19372.0/6561.0) + k2 * (-25360.0/2187.0) + k3 * (64448.0/6561.0) + k4 * (-212.0/729.0))), t0 + h * (8.0/9.0));
        let k6 = ode(&(y0 + h * (k1 * (9017.0/3168.0) + k2 * (-355.0/33.0) + k3 * (46732.0/5247.0) + k4 * (49.0/176.0) + k5 * (-5103.0/18656.0))), t0 + h);
        let k7 = ode(&(y0 + h * (k1 * (35.0/384.0) + k3 * (500.0/1113.0) + k4 * (125.0/192.0) + k5 * (-2187.0/6784.0) + k6 * (11.0/84.0))),t0 + h);
        let y1 = y0 + h * (k1 * (35.0/384.0) + k3 * (500.0/1113.0) + k4 * (125.0/192.0) + k5 * (-2187.0/6784.0) + k6 * (11.0/84.0));
        let e1 = h * (k1 * (35.0/384.0 - 5179.0/57600.0) + k3 * (500.0/1113.0 - 7571.0/16695.0) + k4 * (125.0/192.0 - 393.0/640.0) + k5 * (-2187.0/6784.0 + 92097.0/339200.0) + k6 * (11.0/84.0 - 187.0/2100.0) + k7 * (-1.0/40.0));
        (y1,e1,k7)
    }

    // Normalizes vec using RMS, scaled to y
    fn scale_norm(&self, vec: &SVector<f64,D>, y: &SVector<f64,D>) -> f64
    {
        let mut vec_norm = 0.0;
        for i in 0..D {
            let scale = self.atol + self.rtol * y[i].abs();
            vec_norm += (vec[i]/scale).powi(2);
        }
        (vec_norm / D as f64).sqrt()
    }

    // Returns initial guess for h
    fn guess_timestep<F>(&self,ode: &F, y0: &SVector<f64,D>, t0: f64) -> f64
        where F: Fn(&SVector<f64,D>,f64) -> SVector<f64,D>
    {
        let d0 = self.scale_norm(y0, y0);
        let ydot_0 = ode(y0,t0);
        let d1 = self.scale_norm(&ydot_0,y0);

        // Create initial guess
        let h0: f64 =  if d0 < 1e-5 || d1 < 1e-5 {1e-6} else {0.01 * d0 / d1};

        // Take euler step to estimate curvature
        let y1 = y0 + h0 * ydot_0;
        let ydot_1 = ode(&y1,t0+h0);
        let d2 = self.scale_norm(&(ydot_1 - ydot_0), y0);
        
        (h0 * 100.0).min((0.01/d1.max(d2)).powf(0.2))
    }
}

impl<const D: usize> Solver<D> for DormandPrinceSolver<D> {
    fn solve<F>(&self, ode: &F, y0: &SVector<f64,D>, t_start: f64, t_end: f64) -> (SVector<f64,D>,f64)
        where F: Fn(&SVector<f64,D>,f64) -> SVector<f64,D> 
    {
        let mut t = t_start;
        let mut y = *y0;
        let mut k1 = ode(y0,t);
        let mut h = self.guess_timestep(ode, y0, t_start);
        while t < t_end {
            let res = self.step(&ode, &y, &k1, t, h);

            // compute error
            let err_norm = self.scale_norm(&res.1,&y);

            // recalculate stepsize
            let new_h = (h * self.safety * err_norm.powf(-0.2)).clamp(self.min_clamp,self.max_clamp);

            // Accept or reject step
            if err_norm <= 1.0 {
                y = res.0;
                k1 = res.2;
                t += h;
            }
            h = new_h
        }
        (y,t)
    }

    fn solve_dense<F>(&self, ode: &F, y0: &SVector<f64,D>, t_start: f64, t_end: f64) -> Vec<(SVector<f64,D>, f64)> 
        where F: Fn(&SVector<f64,D>,f64) -> SVector<f64,D>
    {
        let mut t = t_start;
        let mut y = *y0;
        let mut k1 = ode(y0,t);
        let mut h = self.guess_timestep(ode, y0, t_start);
        let mut points: Vec<(SVector<f64,D>, f64)> = Vec::new();
        points.push((y,t));
        while t < t_end {
            let res = self.step(&ode, &y, &k1, t, h);

            // compute error
            let err_norm = self.scale_norm(&res.1,&y);

            // recalculate stepsize
            let new_h = (h * self.safety * err_norm.powf(-0.2)).clamp(self.min_clamp,self.max_clamp);

            // Accept or reject step
            if err_norm <= 1.0 {
                y = res.0;
                k1 = res.2;
                t += h;
                points.push((y,t));
            }
            h = new_h
        }
        points
    }
}