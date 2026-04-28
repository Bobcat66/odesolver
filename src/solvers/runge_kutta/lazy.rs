// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SMatrix;

pub struct RKLazySolution<F,T,Method,const P: usize,const S: usize,const E: usize,const D: usize> 
    where Method: PRKMethod<P,S,E>,
    F:  Fn(&SVector<f64,P>,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
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
    F:  Fn(&SVector<f64,P>,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
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
    F:  Fn(&SVector<f64,P>,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
{

    fn next(&mut self) -> (f64,SMatrix<f64,D,P>)
    {
        let mut accepted = false;
        let mut new_t = self.t;
        let mut new_y = self.y;
        while !accepted {
            new_t = self.t + self.h;
            new_y = PRKStepper::<Method,P,S,E>::step_partitioned(&mut self.k, &mut self.e, &self.ode, self.t, &self.y, &self.f, self.h);
            let time_control = PRKMethodController::<Method,P,S,E>::get_next_step(&new_y, &self.e, &self.y, self.h, &self.cfg);
            accepted = time_control.0;
            self.h = time_control.1; 
        }
        self.y = new_y;
        self.t = new_t;
        self.f = if Method::FSAL {self.k[S-1]} else {(self.ode)(&SVector::<f64,P>::repeat(self.t), &self.y)};
        (self.t,self.y)
    }
}

pub struct PRKLazyDenseSolution< F, Method,const P: usize,const S: usize,const E: usize,const D: usize> 
    where Method: PRKMethod<P,S,E>,
    F:  Fn(&SVector<f64,P>,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
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
    F: Fn(&SVector<f64,P>,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
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
    F: Fn(&SVector<f64,P>,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
{
    fn next(&mut self) -> PRKMethodInterpolant<Method, P, S, E, D>
    {
        let mut accepted = false;
        let mut new_t = self.t;
        let mut new_y = self.y;
        while !accepted {
            new_t = self.t + self.h;
            new_y = PRKStepper::<Method,P,S,E>::step_partitioned(&mut self.k, &mut self.e, &self.ode, self.t, &self.y, &self.f, self.h);
            let time_control = PRKMethodController::<Method,P,S,E>::get_next_step(&new_y, &self.e, &self.y, self.h, &self.cfg);
            accepted = time_control.0;
            self.h = time_control.1; 
        }
        let out = PRKMethodInterpolator::<Method,P,S,E>::interpolate_stage(&self.ode, self.t, new_t, &self.y, &new_y, &self.k);
        self.y = new_y;
        self.t = new_t;
        self.f = if Method::FSAL {self.k[S-1]} else {(self.ode)(&SVector::<f64,P>::repeat(self.t), &self.y)};
        out
    }
}
