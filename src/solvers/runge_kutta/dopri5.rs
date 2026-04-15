// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use crate::solvers::runge_kutta::butcher::{ButchersTableau, ExtendedButchersTableau};
use crate::solvers::runge_kutta::rk_solver::RKSolver;
use crate::solvers::solver::Solver;
use crate::solvers::runge_kutta::rkimpl::*;
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

const P: [[f64; 4]; 7] = [
    [1.0,   -8048581381.0/2820520608.0,     8663915743.0/2820520608.0,  -12715105075.0/11282082432.0],
    [0.0,                          0.0,                           0.0,                           0.0],
    [0.0, 131558114200.0/32700410799.0,  -68118460800.0/10900136933.0,   87487479700.0/32700410799.0],
    [0.0,    -1754552775.0/470086768.0,    14199869525.0/1410260304.0,   -10690763975.0/1880347072.0],
    [0.0, 127303824393.0/49829197408.0, -318862633887.0/49829197408.0, 701980252875.0/199316789632.0],
    [0.0,     -282668133.0/205662961.0,      2019193451.0/616988883.0,     -1453857185.0/822651844.0],
    [0.0,        40617522.0/29380423.0,       -110615467.0/29380423.0,         69997945.0/29380423.0]
];

pub struct DOPRI5 {}

impl ButchersTableau<7> for DOPRI5 {
    fn c() -> &'static [f64; 7] {&C}
    fn a() -> &'static [[f64; 7]; 7] {&A}
    fn b() -> &'static [f64; 7] {&B5}
}

impl ExtendedButchersTableau<7,4> for DOPRI5 {
    fn b_low() -> &'static [f64; 7] {&B4}
    fn p() -> &'static [[f64; 4]; 7] {&P}
}

pub type DOPRI5Solver<const D: usize> = RKSolver<DOPRI5,7, 4, D>;

/* 
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
            let res = rk_step_impl(&C, &A, &B5, &B4, ode, t, &y, &(self.k[0].clone()), h, &mut self.k);

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
*/