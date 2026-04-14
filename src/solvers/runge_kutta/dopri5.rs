// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use crate::solvers::solver::Solver;
use crate::solvers::runge_kutta::utils::*;
use nalgebra::SVector;

// Tableau
const A: [[f64; 7]; 7] = [
    [           0.0,             0.0,            0.0,          0.0,             0.0,       0.0, 0.0],
    [       1.0/5.0,             0.0,            0.0,          0.0,             0.0,       0.0, 0.0],
    [      3.0/40.0,        9.0/40.0,            0.0,          0.0,             0.0,       0.0, 0.0],
    [     44.0/45.0,      -56.0/15.0,       32.0/9.0,          0.0,             0.0,       0.0, 0.0],
    [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0,             0.0,       0.0, 0.0],
    [ 9017.0/3168.0,     -355.0/33.0, 46732.0/5247.0,   49.0/176.0, -5103.0/18656.0,       0.0, 0.0],
    [    35.0/384.0,             0.0,   500.0/1113.0,  125.0/192.0,  -2187.0/6784.0, 11.0/84.0, 0.0]
];
const C: [f64; 7] = [0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0];

const B5: [f64; 7] = [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0];
const B4: [f64; 7] = [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0];

pub struct DOPRI5Solver<const D: usize> 
{
    pub atol: f64, // absolute tolerance
    pub rtol: f64, // normalized tolerance
    pub safety: f64, // safety value to reduce overshoot
    pub min_clamp: f64, // minimum timestep clamp
    pub max_clamp: f64, // maximum timestep clamp
    k: [SVector<f64,D>; 7]
}

impl<const D: usize> Default for DOPRI5Solver<D> {
    fn default() -> Self {
        Self {
            atol: 1e-3,
            rtol: 1e-6,
            safety: 0.9,
            min_clamp: 1e-12,
            max_clamp: f64::MAX,
            k: [SVector::<f64,D>::zeros(); 7]
        }
    }
}

impl<const D: usize> DOPRI5Solver<D> {

    // Returns a tuple of solutions in the form (high_order,error)
    fn step<F>(&mut self, ode: &F, y0: &SVector<f64,D>, t0: f64, h: f64) -> (SVector<f64,D>,SVector<f64,D>) 
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        self.k[1] = ode(t0 + h * C[1], &(y0 + h * (self.k[0] * A[1][0])));
        self.k[2] = ode(t0 + h * C[2], &(y0 + h * (self.k[0] * A[2][0] + self.k[1] * A[2][1])));
        self.k[3] = ode(t0 + h * C[3], &(y0 + h * (self.k[0] * A[3][0] + self.k[1] * A[3][1] + self.k[2] * A[3][2])));
        self.k[4] = ode(t0 + h * C[4], &(y0 + h * (self.k[0] * A[4][0] + self.k[1] * A[4][1] + self.k[2] * A[4][2] + self.k[3] * A[4][3])));
        self.k[5] = ode(t0 + h * C[5], &(y0 + h * (self.k[0] * A[5][0] + self.k[1] * A[5][1] + self.k[2] * A[5][2] + self.k[3] * A[5][3] + self.k[4] * A[5][4])));
        self.k[6] = ode(t0 + h * C[6], &(y0 + h * (self.k[0] * A[6][0] + self.k[1] * A[6][1] + self.k[2] * A[6][2] + self.k[3] * A[6][3] + self.k[4] * A[6][4] + self.k[5] * A[6][5])));
        let mut y1 = SVector::<f64,D>::zeros();
        let mut err = SVector::<f64,D>::zeros();
        for i in 0..7 {
            y1 += self.k[i] * B5[i];
            err += self.k[i] * (B5[i] - B4[i]);
        }
        y1 *= h;
        y1 += y0;
        err *= h;
        (y1,err)
    }
}

impl<const D: usize> Solver<D> for DOPRI5Solver<D> {
    fn solve<F>(&mut self, ode: &F, y0: &SVector<f64,D>, t_start: f64, t_end: f64) -> Vec<(f64,SVector<f64,D>)> 
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        let mut t = t_start;
        let mut y = *y0;
        let mut h = guess_timestep(ode, y0, t_start, self.atol, self.rtol);
        let mut points: Vec<(f64,SVector<f64,D>)> = Vec::new();
        points.push((t,y));
        self.k[0] = ode(t_start, y0);
        while t < t_end {
            let res = self.step(&ode, &y, t, h);

            // compute error
            let err_norm = scale_norm(&res.1,&y, self.atol, self.rtol);

            // recalculate stepsize
            let new_h = (h * self.safety * err_norm.powf(-0.2)).clamp(self.min_clamp,self.max_clamp);

            // Accept or reject step
            if err_norm <= 1.0 {
                y = res.0;
                self.k[0] = self.k[6];
                t += h;
                points.push((t,y));
            }
            h = new_h;
        }
        points
    }
}