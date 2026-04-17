// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
use std::{f32::consts::E, marker::PhantomData};

use nalgebra::SVector;

use crate::solvers::{common::{norm, select_initial_timestep}, dense::{DenseInterpolant, DenseOutput}, runge_kutta::{rk_dense::RKInterpolant, rk_method::{RKController, RKInterpolator, RKMethod}}};

// Controller
pub struct AdaptiveRKConfig {
    pub atol: f64, // absolute tolerance
    pub rtol: f64, // normalized tolerance
    pub safety: f64, // safety value to reduce overshoot
    pub min_clamp: f64, // minimum timestep clamp
    pub max_clamp: f64, // maximum timestep clamp
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
            min_clamp: 1e-12,
            max_clamp: f64::MAX,
            min_factor: 0.2,
            max_factor: 10.0
        }
    }
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

        // recalculate stepsize
        let mut new_h = (h * cfg.safety * err_norm.powf(Self::ERROR_EXPONENT)).clamp(cfg.min_clamp,cfg.max_clamp);
        let factor = new_h/h;

        if factor > cfg.max_factor {
            new_h = h * cfg.max_factor;
        } else if factor < cfg.min_factor {
            new_h = h * cfg.min_factor;
        }

        (err_norm <= 1.0,new_h)
    }
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SVector<f64,D>, f0: &SVector<f64,D>, cfg: &AdaptiveRKConfig) -> f64
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        select_initial_timestep(ode, y0, t0, f0, cfg.atol, cfg.rtol, Method::ERR_ORDER)
    }
}

// Interpolator
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
    fn interpolate_stage<const D: usize>(t0: f64, t1: f64, point: SVector<f64,D>, stage: [SVector<f64,D>; S]) -> Box<dyn DenseInterpolant<D>> {
        Box::new(RKInterpolant::new(t0, t1, point, stage, Shampine::P))
    }
    fn interpolate_dense<const D: usize>(points: &Vec<(f64,SVector<f64,D>)>, stages: &Vec<[SVector<f64,D>; S]>) -> DenseOutput<D> 
    {
        let steps = stages.len();
        let mut segments: Vec<(f64,Box<dyn DenseInterpolant<D>>)> = Vec::new();
        for i in 0..(steps-1) {
            segments.push(
                (
                    points[i].0,
                    Self::interpolate_stage(points[i].0,points[i + 1].0, points[i].1, stages[i])
                )
            );
        }

        DenseOutput::<D>::new(segments)
    }
}


