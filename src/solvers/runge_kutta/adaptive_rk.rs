// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
use std::{f32::consts::E, marker::PhantomData};

use nalgebra::SVector;

use crate::{algebra::{mapping::Mapping, polynomial::Polynomial}, solvers::{common::{norm, select_initial_timestep}, dense::{DenseInterpolant, DenseOutput}, runge_kutta::rk_method::{RKController, RKInterpolator, RKMethod}}};

// Controller
#[derive(Copy, Clone)]
pub struct AdaptiveRKConfig {
    pub atol: f64, // absolute tolerance
    pub rtol: f64, // normalized tolerance
    pub safety: f64, // safety value to reduce overshoot
    pub min_step: f64, // minimum timestep clamp
    pub max_step: f64, // maximum timestep clamp
    pub min_factor: f64, // minimum timestep change factor
    pub max_factor: f64, // maximum timestep change factor
}

impl Default for AdaptiveRKConfig {
    fn default() -> Self 
    {
        Self {
            atol: 1e-6,
            rtol: 1e-3,
            safety: 0.9,
            min_step: 1e-12,
            max_step: f64::MAX,
            min_factor: 0.2,
            max_factor: 10.0
        }
    }
}

pub fn compute_new_h(err_norm: f64, h: f64, safety: f64, max_step: f64, min_step: f64, max_factor: f64, min_factor: f64, error_exponent: f64) -> f64
{
    let mut new_h = (h * safety * err_norm.powf(error_exponent)).clamp(min_step,max_step);
    let factor = new_h/h;

    if factor > max_factor {
        new_h = h * max_factor;
    } else if factor < min_factor {
        new_h = h * min_factor;
    }
    new_h
}

pub struct FirstOrderAdaptiveRKController<Method, const S: usize> 
    where Method: RKMethod<S,1>
{
    _marker: PhantomData<Method>
}

impl<Method, const S: usize> FirstOrderAdaptiveRKController<Method, S>
    where Method: RKMethod<S,1>
{
    const ERROR_EXPONENT: f64 = {-1.0/(Method::ERR_ORDER as f64 + 1.0)};
}

impl<Method, const S: usize> RKController<1> for FirstOrderAdaptiveRKController<Method, S>
    where Method: RKMethod<S,1>
{

    type Config = AdaptiveRKConfig;

    fn get_next_step<const D: usize>(y1: &SVector<f64, D>, e: &[SVector<f64,D>; 1], y0: &SVector<f64,D>, h: f64, cfg: &AdaptiveRKConfig) -> (bool, f64)
    {
        let err = e[0];
        let scale = (y0.abs().sup(&(y1.abs())) * cfg.rtol).add_scalar(cfg.atol);
        let err_norm = norm(&err, &scale);

        (err_norm <= 1.0,compute_new_h(err_norm, h, cfg.safety, cfg.max_step, cfg.min_step, cfg.max_factor, cfg.min_factor, Self::ERROR_EXPONENT))
    }
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SVector<f64,D>, f0: &SVector<f64,D>, cfg: &AdaptiveRKConfig) -> f64
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        select_initial_timestep(ode, y0, t0, f0, cfg.atol, cfg.rtol, Method::ERR_ORDER)
    }
}

// Interpolator. This is based on Shampine's method for deriving polynomial interpolants

pub struct ShampineInterpolant<const S: usize, const P: usize, const D: usize> {
    t0: f64,
    t1: f64,
    h: f64,
    y0: SVector<f64, D>,
    y1: SVector<f64, D>,
    k: [SVector<f64, D>; S],
    b: [Polynomial<f64, P>; S]
}

impl<const S: usize, const P: usize, const D: usize> ShampineInterpolant<S,P,D> {
    fn eval_impl(&self, theta: f64) -> SVector<f64, D>
    {
        let mut y_theta = self.y0.clone();
        for i in 0..S {
            y_theta += self.h * self.b[i].eval(theta) * self.k[i];
        }
        y_theta
    }

    fn get_theta(&self, t: f64) -> f64 {
        (t-self.t0)/self.h
    }

    pub fn new(t0: f64, t1: f64, y0: SVector<f64,D>, y1: SVector<f64,D>, k: [SVector<f64,D>; S], p: [[f64; P]; S]) -> Self {
        Self {
            t0: t0,
            t1: t1,
            h: t1 - t0,
            y0: y0,
            y1: y1,
            k: k,
            b: std::array::from_fn(|i| Polynomial { a: p[i] })
        }
    }
}

impl<const S: usize, const P: usize, const D: usize> DenseInterpolant<D> for ShampineInterpolant<S, P, D> {
    fn eval(&self, t: f64) -> SVector<f64,D>
    {
        self.eval_impl(self.get_theta(t))
    }
    fn low_t(&self) -> f64
    {
        self.t0
    }
    fn high_t(&self) -> f64
    {
        self.t1
    }
    fn y0(&self) -> SVector<f64,D>
    {
        self.y0
    }
    fn y1(&self) -> SVector<f64,D>
    {
        self.y1
    }
}
pub trait ShampineConfig<const P: usize, const S: usize> {
    const P: [[f64; P]; S]; // shampine polynomial weights
}

pub struct ShampineRKInterpolator<Shampine, const P: usize, const S: usize>
    where Shampine: ShampineConfig<P,S>
{
    _marker: PhantomData<Shampine>
}

impl<Shampine, const P: usize, const S: usize> RKInterpolator<S> for ShampineRKInterpolator<Shampine, P, S>
    where Shampine: ShampineConfig<P,S>
{
    fn interpolate_stage<F,const D: usize>(ode: &F, t0: f64, t1: f64, y0: &SVector<f64,D>, y1: &SVector<f64,D>, stage: &[SVector<f64,D>; S]) -> Box<dyn DenseInterpolant<D>> 
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        Box::new(ShampineInterpolant::new(t0, t1, *y0, *y1,*stage, Shampine::P))
    }
}


