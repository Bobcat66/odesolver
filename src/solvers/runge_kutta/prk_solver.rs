// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::default::Default;
use std::array;

use nalgebra::SMatrix;

use crate::{solvers::{DenseOutput, DenseSolver, LazyDenseSolution, LazySolution, Solver, runge_kutta::{prk_method::{PRKController, PRKInterpolator, PRKMethod, PRKMethodConfig, PRKMethodController, PRKMethodInterpolant, PRKMethodInterpolator}, prk_stepper::PRKStepper}}};

// Lazy solutions
pub struct PRKLazySolution<F,Method,const P: usize,const S: usize,const E: usize,const D: usize> 
    where Method: PRKMethod<P,S,E>,
    F:  Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
{
    cfg: PRKMethodConfig<Method,P,S,E>,
    ode: F,
    y: SMatrix<f64,D,P>,
    t: f64,
    h: f64,
    f: SMatrix<f64,D,P>,
    k: [SMatrix<f64,D,P>; S],
    e: [SMatrix<f64,D,P>; E]
}

impl<F, Method, const P: usize, const S: usize, const E: usize, const D: usize> PRKLazySolution<F, Method, P, S, E, D>
    where Method: PRKMethod<P,S,E>,
    F:  Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
{
    pub fn new(cfg: PRKMethodConfig<Method,P,S,E>,ode: F, t0: f64, y0: SMatrix<f64,D,P>, f0: SMatrix<f64,D,P>) -> Self
    {
        let h = PRKMethodController::<Method,P,S,E>::select_initial_timestep(&ode, t0, &y0, &f0, &cfg);
        Self {
            cfg: cfg,
            ode: ode,
            t: t0,
            y: y0,
            f: f0,
            h: h,
            k: array::from_fn(|_i|(SMatrix::<f64,D,P>::zeros())),
            e: array::from_fn(|_i|(SMatrix::<f64,D,P>::zeros()))
        }
    }
}

impl<F, Method, const P: usize, const S: usize, const E: usize, const D: usize> LazySolution<P,D> for PRKLazySolution<F,Method,P,S,E,D>
    where Method: PRKMethod<P, S, E>,
    F:  Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
{

    fn next(&mut self) -> (f64,SMatrix<f64,D,P>)
    {
        let mut accepted = false;
        let mut new_t = self.t;
        let mut new_y = self.y;
        while !accepted {
            new_t = self.t + self.h;
            new_y = PRKStepper::<Method,P,S,E>::step(&mut self.k, &mut self.e, &self.ode, self.t, &self.y, &self.f, self.h);
            let time_control = PRKMethodController::<Method,P,S,E>::get_next_step(&new_y, &self.e, &self.y, self.h, &self.cfg);
            accepted = time_control.0;
            self.h = time_control.1; 
        }
        self.y = new_y;
        self.t = new_t;
        self.f = if Method::FSAL {self.k[S-1]} else {(self.ode)(self.t,&self.y)};
        (self.t,self.y)
    }
}

pub struct PRKLazyDenseSolution< F, Method,const P: usize,const S: usize,const E: usize,const D: usize> 
    where Method: PRKMethod<P,S,E>,
    F:  Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
{
    cfg: PRKMethodConfig<Method,P,S,E>,
    ode: F,
    y: SMatrix<f64,D,P>,
    t: f64,
    h: f64,
    f: SMatrix<f64,D,P>,
    k: [SMatrix<f64,D,P>; S],
    e: [SMatrix<f64,D,P>; E]
}

impl<F, Method,const P: usize, const S: usize, const E: usize, const D: usize> PRKLazyDenseSolution<F,Method,P, S, E, D>
    where Method: PRKMethod<P,S,E>,
    F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
{
    pub fn new(cfg: PRKMethodConfig<Method,P,S,E>,ode: F, t0: f64, y0: SMatrix<f64,D,P>, f0: SMatrix<f64,D,P>) -> Self
    {
        let h = PRKMethodController::<Method,P,S,E>::select_initial_timestep(&ode, t0, &y0, &f0, &cfg);
        Self {
            cfg: cfg,
            ode: ode,
            t: t0,
            y: y0,
            f: f0,
            h: h,
            k: array::from_fn(|_i|(SMatrix::<f64,D,P>::zeros())),
            e: array::from_fn(|_i|(SMatrix::<f64,D,P>::zeros()))
        }
    }
}

impl<F, Method, const P: usize,const S: usize, const E: usize, const D: usize> LazyDenseSolution<PRKMethodInterpolant<Method,P,S,E,D>,P,D> for PRKLazyDenseSolution<F,Method,P,S,E,D>
    where Method: PRKMethod<P,S,E>,
    F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
{
    fn next(&mut self) -> PRKMethodInterpolant<Method, P, S, E, D>
    {
        let mut accepted = false;
        let mut new_t = self.t;
        let mut new_y = self.y;
        while !accepted {
            new_t = self.t + self.h;
            new_y = PRKStepper::<Method,P,S,E>::step(&mut self.k, &mut self.e, &self.ode, self.t, &self.y, &self.f, self.h);
            let time_control = PRKMethodController::<Method,P,S,E>::get_next_step(&new_y, &self.e, &self.y, self.h, &self.cfg);
            accepted = time_control.0;
            self.h = time_control.1; 
        }
        let out = PRKMethodInterpolator::<Method,P,S,E>::interpolate_stage(&self.ode, self.t, new_t, &self.y, &new_y, &self.k);
        self.y = new_y;
        self.t = new_t;
        self.f = if Method::FSAL {self.k[S-1]} else {(self.ode)(self.t,&self.y)};
        out
    }
}

pub struct PRKSolver<Method, const P: usize, const S: usize, const E: usize, const D: usize> 
    where Method: PRKMethod<P,S,E>
{
    pub cfg: PRKMethodConfig<Method,P,S,E>,
    k: [SMatrix<f64,D,P>; S],
    e: [SMatrix<f64,D,P>; E],
}

impl<Method,const P: usize, const S: usize, const E: usize, const D: usize> PRKSolver<Method,P,S,E,D>
    where Method: PRKMethod<P,S,E>
{
    
    pub fn new(cfg: PRKMethodConfig<Method,P,S,E>) -> Self {
        Self {
            cfg: cfg,
            k: array::from_fn(|_i|(SMatrix::<f64,D,P>::zeros())),
            e: array::from_fn(|_i|(SMatrix::<f64,D,P>::zeros()))
        }
    }

    // Returns (steps,Vec(t,y))
    fn solve_impl<F,C>(&mut self, ode: &F, y_start: &SMatrix<f64,D,P>, t_start: f64, t_end: f64, stage_consumer: &mut C, verbose: bool) -> Vec<(f64,SMatrix<f64,D,P>)>
        where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>,
        C: FnMut(&[SMatrix<f64,D,P>; S]) -> ()
    {
        let k = &mut self.k;
        let e = &mut self.e;
        let cfg = &self.cfg;

        let mut t = t_start;
        let mut y = *y_start;
        let mut f = ode(t_start,y_start);
        let mut h = PRKMethodController::<Method,P,S,E>::select_initial_timestep(ode, t_start, y_start, &f, cfg);
        let mut points: Vec<(f64,SMatrix<f64,D,P>)> = Vec::new();
        points.push((t_start,*y_start));
        while t < t_end {
            h = (t_end - t).min(h);
            let new_t = t + h;
            if verbose {
                println!("t={}, h={}, evaluating at t={}",t,h,new_t);
            }
            let res = PRKStepper::<Method,P,S,E>::step(k, e, ode, t, &y, &f, h);
            if verbose {
                println!("y(t)={:?}",res)
            }
            let time_control = PRKMethodController::<Method,P,S,E>::get_next_step(&res,e, &y, h, cfg);
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

impl<Method, const P: usize,const S: usize, const E: usize, const D: usize> Default for PRKSolver<Method,P,S,E,D>
    where Method: PRKMethod<P,S,E>
{
    fn default() -> Self {
        Self::new(PRKMethodConfig::<Method,P,S,E>::default())
    }
}

impl<Method, const P: usize, const S: usize, const E: usize, const D: usize> Solver<P,D> for PRKSolver<Method,P,S,E,D>
    where Method: PRKMethod<P,S,E>,
{
    fn solve<F>(&mut self, ode: &F, y_start: &SMatrix<f64,D,P>, t_start: f64, t_end: f64, verbose: bool) -> Vec<(f64,SMatrix<f64,D,P>)> 
            where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
    {
        self.solve_impl(ode, y_start, t_start, t_end, &mut (|_stage: &[SMatrix<f64,D,P>; S]| ()), verbose)
    }
    fn solve_lazy<F>(&mut self, ode: &F, y_start: &SMatrix<f64,D,P>, t_start: f64) -> impl LazySolution<P,D>
            where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P> 
    {
        PRKLazySolution::<_,Method,P,S,E,D>::new(self.cfg, ode, t_start, *y_start, ode(t_start, y_start))
    }
}

impl<Method,const P: usize, const S: usize, const E: usize, const D: usize> DenseSolver<P,D> for PRKSolver<Method,P,S,E,D>
    where Method: PRKMethod<P,S,E>
{
    type InterpolantType = PRKMethodInterpolant<Method,P,S,E,D>;
    fn solve_dense<F>(&mut self, ode: &F, y_start: &SMatrix<f64,D,P>, t_start: f64, t_end: f64, verbose: bool) -> (Vec<(f64,SMatrix<f64,D,P>)>,DenseOutput<Self::InterpolantType,P,D>)
        where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
    {
        let mut stages: Vec<[SMatrix<f64,D,P>; S]> = Vec::new();
        let points = self.solve_impl(ode, y_start, t_start, t_end, &mut (|stage: &[SMatrix<f64,D,P>; S]| stages.push(*stage)), verbose);
        let dense = PRKMethodInterpolator::<Method,P,S,E>::interpolate_dense(ode, &points, &stages);
        (points,dense)
    }
    fn solve_dense_lazy<F>(&mut self, ode: &F, y_start: &SMatrix<f64,D,P>, t_start: f64) -> impl LazyDenseSolution<Self::InterpolantType, P,D>
        where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P> 
    {
        PRKLazyDenseSolution::<_,Method,P,S,E,D>::new(self.cfg, ode, t_start, *y_start, ode(t_start, y_start))
    }
}