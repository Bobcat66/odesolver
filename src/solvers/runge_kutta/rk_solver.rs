// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::default::Default;

use nalgebra::SVector;

use crate::solvers::{dense::{DenseInterpolant, DenseOutput}, runge_kutta::{rk_method::{MethodConfig, MethodController, MethodInterpolator, MethodInterpolant, RKController, RKInterpolator, RKMethod}, rk_stepper::RKStepper}, solver::{DenseSolver, LazyDenseSolution, LazySolution, Solver}};

// Lazy solutions
pub struct RKLazySolution<F, Method, const S: usize, const E: usize, const D: usize> 
    where Method: RKMethod<S, E>,
    F:  Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
{
    cfg: MethodConfig<Method,S,E>,
    ode: F,
    y: SVector<f64, D>,
    t: f64,
    h: f64,
    f: SVector<f64, D>,
    k: [SVector<f64,D>; S],
    e: [SVector<f64,D>; E]
}

impl<F, Method, const S: usize, const E: usize, const D: usize> RKLazySolution<F, Method, S, E, D>
    where Method: RKMethod<S, E>,
    F:  Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
{
    pub fn new(cfg: MethodConfig<Method,S,E>,ode: F, t0: f64, y0: SVector<f64, D>, f0: SVector<f64, D>) -> Self
    {
        let h = MethodController::<Method, S, E>::select_initial_timestep(&ode, t0, &y0, &f0, &cfg);
        Self {
            cfg: cfg,
            ode: ode,
            t: t0,
            y: y0,
            f: f0,
            h: h,
            k: [SVector::<f64,D>::zeros(); S],
            e: [SVector::<f64,D>::zeros(); E]
        }
    }
}

impl<F, Method, const S: usize, const E: usize, const D: usize> LazySolution<D> for RKLazySolution<F, Method,S,E,D>
    where Method: RKMethod<S, E>,
    F:  Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
{

    fn next(&mut self) -> (f64,SVector<f64, D>)
    {
        let mut accepted = false;
        let mut new_t = self.t;
        let mut new_y = self.y;
        while !accepted {
            new_t = self.t + self.h;
            new_y = RKStepper::<Method,S,E>::step(&mut self.k, &mut self.e, &self.ode, self.t, &self.y, &self.f, self.h);
            let time_control = MethodController::<Method,S,E>::get_next_step(&new_y, &self.e, &self.y, self.h, &self.cfg);
            accepted = time_control.0;
            self.h = time_control.1; 
        }
        self.y = new_y;
        self.t = new_t;
        self.f = if Method::FSAL {self.k[S-1]} else {(self.ode)(self.t,&self.y)};
        (self.t,self.y)
    }
}

pub struct RKLazyDenseSolution<F, Method, const S: usize, const E: usize, const D: usize> 
    where Method: RKMethod<S, E>,
    F:  Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
{
    cfg: MethodConfig<Method,S,E>,
    ode: F,
    y: SVector<f64, D>,
    t: f64,
    h: f64,
    f: SVector<f64, D>,
    k: [SVector<f64,D>; S],
    e: [SVector<f64,D>; E]
}

impl<F, Method, const S: usize, const E: usize, const D: usize> RKLazyDenseSolution<F, Method, S, E, D>
    where Method: RKMethod<S, E>,
    F:  Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
{
    pub fn new(cfg: MethodConfig<Method,S,E>,ode: F, t0: f64, y0: SVector<f64, D>, f0: SVector<f64, D>) -> Self
    {
        let h = MethodController::<Method, S, E>::select_initial_timestep(&ode, t0, &y0, &f0, &cfg);
        Self {
            cfg: cfg,
            ode: ode,
            t: t0,
            y: y0,
            f: f0,
            h: h,
            k: [SVector::<f64,D>::zeros(); S],
            e: [SVector::<f64,D>::zeros(); E]
        }
    }
}

impl<F, Method, const S: usize, const E: usize, const D: usize> LazyDenseSolution<MethodInterpolant<Method, S, E, D>, D> for RKLazyDenseSolution<F,Method,S,E,D>
    where Method: RKMethod<S, E>,
    F:  Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
{
    fn next(&mut self) -> MethodInterpolant<Method, S, E, D>
    {
        let mut accepted = false;
        let mut new_t = self.t;
        let mut new_y = self.y;
        while !accepted {
            new_t = self.t + self.h;
            new_y = RKStepper::<Method,S,E>::step(&mut self.k, &mut self.e, &self.ode, self.t, &self.y, &self.f, self.h);
            let time_control = MethodController::<Method,S,E>::get_next_step(&new_y, &self.e, &self.y, self.h, &self.cfg);
            accepted = time_control.0;
            self.h = time_control.1; 
        }
        let out = MethodInterpolator::<Method, S, E>::interpolate_stage(&self.ode, self.t, new_t, &self.y, &new_y, &self.k);
        self.y = new_y;
        self.t = new_t;
        self.f = if Method::FSAL {self.k[S-1]} else {(self.ode)(self.t,&self.y)};
        out
    }
}

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
            e: [SVector::<f64,D>::zeros(); E]
        }
    }

    // Returns (steps,Vec(t,y))
    fn solve_impl<F,C>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64, stage_consumer: &mut C, verbose: bool) -> Vec<(f64,SVector<f64,D>)>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>,
        C: FnMut(&[SVector<f64,D>; S]) -> ()
    {
        let k = &mut self.k;
        let e = &mut self.e;
        let cfg = &self.cfg;

        let mut t = t_start;
        let mut y = *y_start;
        let mut f = ode(t_start,y_start);
        let mut h = MethodController::<Method,S,E>::select_initial_timestep(ode, t_start, y_start, &f, cfg);
        let mut points: Vec<(f64,SVector<f64,D>)> = Vec::new();
        points.push((t_start,*y_start));
        while t < t_end {
            h = (t_end - t).min(h);
            let new_t = t + h;
            if verbose {
                println!("t={}, h={}, evaluating at t={}",t,h,new_t);
            }
            let res = RKStepper::<Method,S,E>::step(k, e, ode, t, &y, &f, h);
            if verbose {
                println!("y(t)={:?}",res)
            }
            let time_control = MethodController::<Method,S,E>::get_next_step(&res,e, &y, h, cfg);
            if time_control.0 {
                y = res;
                t = new_t;
                f = if Method::FSAL {k[S-1]} else {ode(t,&y)};
                points.push((t,y));
                stage_consumer(k);
            }
            h = time_control.1;
            if verbose {
                println!("{}",if time_control.0 {"ACCEPTED"} else {"REJECTED"})
            }
        }
        points
    }
}

impl<Method, const S: usize, const E: usize, const D: usize> Default for RKSolver<Method,S,E,D>
    where Method: RKMethod<S,E>
{
    fn default() -> Self {
        Self::new(MethodConfig::<Method,S,E>::default())
    }
}

impl<Method, const S: usize, const E: usize, const D: usize> Solver<D> for RKSolver<Method,S,E,D>
    where Method: RKMethod<S,E>,
{
    fn solve<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64, verbose: bool) -> Vec<(f64,SVector<f64,D>)> 
            where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        self.solve_impl(ode, y_start, t_start, t_end, &mut (|_stage: &[SVector<f64,D>; S]| ()), verbose)
    }
    fn solve_lazy<F,C>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64) -> impl LazySolution<D>
            where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D> 
    {
        RKLazySolution::<_,Method, S, E, D>::new(self.cfg, ode, t_start, *y_start, ode(t_start, y_start))
    }
}

impl<Method, const S: usize, const E: usize, const D: usize> DenseSolver<D> for RKSolver<Method,S,E,D>
    where Method: RKMethod<S,E>
{
    type InterpolantType = MethodInterpolant<Method,S,E,D>;
    fn solve_dense<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64, verbose: bool) -> (Vec<(f64,SVector<f64,D>)>,DenseOutput<Self::InterpolantType,D>) 
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        let mut stages: Vec<[SVector<f64,D>; S]> = Vec::new();
        let points = self.solve_impl(ode, y_start, t_start, t_end, &mut (|stage: &[SVector<f64,D>; S]| stages.push(*stage)), verbose);
        let dense = MethodInterpolator::<Method,S,E>::interpolate_dense(ode, &points, &stages);
        (points,dense)
    }
    fn solve_dense_lazy<F,C>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64) -> impl LazyDenseSolution<Self::InterpolantType, D>
            where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D> 
    {
        RKLazyDenseSolution::<_,Method, S, E, D>::new(self.cfg, ode, t_start, *y_start, ode(t_start, y_start))
    }
}