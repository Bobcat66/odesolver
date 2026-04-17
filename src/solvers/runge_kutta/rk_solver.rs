// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::default::Default;

use nalgebra::SVector;

use crate::solvers::{dense::{DenseInterpolant, DenseOutput}, runge_kutta::{rk_method::{MethodConfig, MethodController, MethodInterpolator, RKController, RKInterpolator, RKMethod}, rk_stepper::RKStepper}, solver::{DenseSolver, Solver}};

// T is the RKKernel config type. The RKKernel is a stateless construct that only implements functions for a solver
pub struct RKSolver<Method, const S: usize, const E: usize, const D: usize> 
    where Method: RKMethod<S,E>
{
    pub cfg: MethodConfig<Method,S,E>,
    k: [SVector<f64,D>; S],
    e: [SVector<f64,D>; E],
}

impl<Method, const S: usize, const E: usize, const D: usize> RKSolver<Method,S,E,D>
    where Method: RKMethod<S,E>
{
    
    pub fn new(cfg: MethodConfig<Method,S,E>) -> Self {
        Self {
            cfg: cfg,
            k: [SVector::<f64,D>::zeros(); S],
            e: [SVector::<f64,D>::zeros(); E],
        }
    }

    // Returns (steps,Vec(t,y))
    fn solve_impl<F,P,C>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64, point_consumer: &mut P, stage_consumer: &mut C) -> ()
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>,
        P: FnMut(f64,SVector<f64,D>) -> (),
        C: FnMut(&[SVector<f64,D>; S]) -> ()
    {
        let k = &mut self.k;
        let e = &mut self.e;
        let cfg = &self.cfg;

        let mut t = t_start;
        let mut y = *y_start;
        let mut f = ode(t_start,y_start);
        let mut h = MethodController::<Method,S,E>::select_initial_timestep(ode, t_start, y_start, &f, cfg);
        point_consumer(t_start,*y_start);
        while t < t_end {
            h = (t_end - t).min(h);
            let new_t = t + h;
            let res = RKStepper::<Method,S,E>::step(k, e, ode, t, &y, &f, h);
            let time_control = MethodController::<Method,S,E>::get_next_step(&res,e, &y, h, cfg);
            if time_control.0 {
                y = res;
                t = new_t;
                f = if Method::FSAL {k[S-1]} else {ode(t,&y)};
                point_consumer(t,y);
                stage_consumer(k);
            }
            h = time_control.1;
        }
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
        let mut points: Vec<(f64,SVector<f64,D>)> = Vec::new();
        self.solve_impl(ode, y_start, t_start, t_end, &mut (|t,y| points.push((t,y))), &mut (|_stage: &[SVector<f64,D>; S]| ()));
        points
    }
    fn solve_stream<F,C>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64, consumer: &mut C) -> ()
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>,
        C: FnMut(f64,SVector<f64,D>) -> ()
    {
        self.solve_impl(ode, y_start, t_start, t_end, consumer, &mut (|_stage: &[SVector<f64,D>; S]| ()));
    }
}

impl<Method, const S: usize, const O: usize, const D: usize> DenseSolver<D> for RKSolver<Method,S,O,D>
    where Method: RKMethod<S,O>
{
    fn solve_dense<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64) -> (Vec<(f64,SVector<f64,D>)>,DenseOutput<D>) 
            where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        let mut stages: Vec<[SVector<f64,D>; S]> = Vec::new();
        let mut points: Vec<(f64,SVector<f64,D>)> = Vec::new();
        self.solve_impl(ode, y_start, t_start, t_end, &mut (|t,y| points.push((t,y))), &mut (|stage: &[SVector<f64,D>; S]| stages.push(*stage)));
        let dense = MethodInterpolator::<Method,S,O>::interpolate_dense(ode, &points, &stages);
        (points,dense)
    }
}