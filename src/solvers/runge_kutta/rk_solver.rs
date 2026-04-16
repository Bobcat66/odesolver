// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::default::Default;

use nalgebra::SVector;

use crate::solvers::{dense::DenseOutput, runge_kutta::{rk_method::{MethodConfig, MethodController, MethodInterpolator, RKController, RKInterpolator, RKMethod}, rk_stepper::RKStepper}, solver::{DenseSolver, Solver}};

// T is the RKKernel config type. The RKKernel is a stateless construct that only implements functions for a solver
pub struct RKSolver<Method, const S: usize, const O: usize, const D: usize> 
    where Method: RKMethod<S,O>
{
    pub cfg: MethodConfig<Method,S,O>,
    k: [SVector<f64,D>; S],
    o: [SVector<f64,D>; O],
}

impl<Method, const S: usize, const O: usize, const D: usize> RKSolver<Method,S,O,D>
    where Method: RKMethod<S,O>
{
    
    pub fn new(cfg: MethodConfig<Method,S,O>) -> Self {
        Self {
            cfg: cfg,
            k: [SVector::<f64,D>::zeros(); S],
            o: [SVector::<f64,D>::zeros(); O],
        }
    }

    // Returns (steps,Vec(t,y))
    fn solve_impl<F,C>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64, stage_consumer: &mut C) -> Vec<(f64,SVector<f64,D>)>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>,
        C: FnMut(&[SVector<f64,D>; S]) -> ()
    {
        let k = &mut self.k;
        let o = &mut self.o;
        let cfg = &self.cfg;

        let mut t = t_start;
        let mut y = *y_start;
        let mut f = ode(t_start,y_start);
        let mut h = MethodController::<Method,S,O>::select_initial_timestep(ode, t_start, y_start, &f, cfg);
        let mut points: Vec<(f64,SVector<f64,D>)> = Vec::new();
        points.push((t_start,*y_start));
        while t < t_end {
            h = (t_end - t).min(h);
            let new_t = t + h;
            RKStepper::<Method,S,O>::step(k, o, ode, t, &y, &f, h);
            let time_control = MethodController::<Method,S,O>::get_next_step(o, &y, t, h, t_end, cfg);
            if time_control.0 {
                y = o[0];
                t = new_t;
                f = if Method::FSAL {k[S - 1]} else {ode(t,&y)};
                points.push((t,y));
                stage_consumer(k);
            }
            h = time_control.1;
        }
        points
    }
}

impl<Method, const S: usize, const O: usize, const D: usize> Default for RKSolver<Method,S,O,D>
    where Method: RKMethod<S,O>
{
    fn default() -> Self {
        Self::new(MethodConfig::<Method,S,O>::default())
    }
}

impl<Method, const S: usize, const O: usize, const D: usize> Solver<D> for RKSolver<Method,S,O,D>
    where Method: RKMethod<S,O>,
{
    fn solve<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64) -> Vec<(f64,SVector<f64,D>)> 
            where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        self.solve_impl(ode, y_start, t_start, t_end, &mut (|_stage: &[SVector<f64,D>; S]| ()))
    }
}

impl<Method, const S: usize, const O: usize, const D: usize> DenseSolver<D> for RKSolver<Method,S,O,D>
    where Method: RKMethod<S,O>
{
    fn solve_dense<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64) -> (Vec<(f64,SVector<f64,D>)>,DenseOutput<D>) 
            where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        let mut stages: Vec<[SVector<f64,D>; S]> = Vec::new();
        let points = self.solve_impl(ode, y_start, t_start, t_end, &mut (|stage: &[SVector<f64,D>; S]| stages.push(*stage)));
        let dense = MethodInterpolator::<Method,S,O>::interpolate_dense(&points, &stages);
        (points,dense)
    }
}