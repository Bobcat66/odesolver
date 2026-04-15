// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
use std::marker::PhantomData;

use nalgebra::SVector;

use crate::solvers::{common::{norm, select_initial_timestep}, dense::{DenseInterpolant, DenseOutput}, runge_kutta::{rk_controller::RKController, rk_dense::RKInterpolant, rk_interpolator::RKInterpolator, rk_stepper::ButchersTableau}};

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
            atol: 1e-3,
            rtol: 1e-6,
            safety: 0.9,
            min_clamp: 1e-12,
            max_clamp: f64::MAX,
            min_factor: 0.2,
            max_factor: 10.0
        }
    }
}
pub struct FirstOrderAdaptiveRKController<Tableau, const S: usize> 
    where Tableau: ButchersTableau<S,2>
{
    _marker: PhantomData<Tableau>
}

impl<Tableau, const S: usize> FirstOrderAdaptiveRKController<Tableau,S>
    where Tableau: ButchersTableau<S,2>
{
    const ERROR_EXPONENT: f64 = {-1.0/(Tableau::ORDERS[1] as f64 + 1.0)};
}

impl<Tableau, const S: usize> RKController<2> for FirstOrderAdaptiveRKController<Tableau, S>
    where Tableau: ButchersTableau<S,2>
{

    type Config = AdaptiveRKConfig;

    fn get_next_step<const D: usize>(o: &[SVector<f64,D>; 2], y0: &SVector<f64,D>, t1: f64, t0: f64, h: f64, t_end: f64, cfg: &AdaptiveRKConfig) -> (bool, f64)
    {
        let y1 = o[0];
        let err = o[1];
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

        if t1 + new_h > t_end {
            new_h = t_end - t1;
        }

        (err_norm <= 1.0,new_h)
    }
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SVector<f64,D>, f0: &SVector<f64,D>, cfg: &AdaptiveRKConfig) -> f64
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        select_initial_timestep(ode, y0, t0, f0, cfg.atol, cfg.rtol, Tableau::ORDERS[1])
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
    fn interpolate_dense<const D: usize>(steps: usize, points: &Vec<(f64,SVector<f64,D>)>, stages: &Vec<[SVector<f64,D>; S]>) -> DenseOutput<D> 
    {
        let mut segments: Vec<(f64,Box<dyn DenseInterpolant<D>>)> = Vec::new();
        for i in 0..(steps-1) {
            segments.push(
                (
                    points[i].0,
                    Box::new(RKInterpolant::new(points[i].0,points[i + 1].0, points[i].1, stages[i], Shampine::P))
                )
            );
        }
        DenseOutput::<D>::new(segments)
    }
}
