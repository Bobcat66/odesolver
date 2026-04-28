// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::marker::PhantomData;

use nalgebra::{SMatrix,SVector};

use crate::{solvers::{common::{norm, select_initial_timestep,convert_t}, runge_kutta::rk_common::RKController}};

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

pub struct AdaptiveController<const ERR_ORDER: usize, const E: usize> {}

impl<const ERR_ORDER: usize, const E: usize> AdaptiveController<ERR_ORDER, E>
{
    const ERROR_EXPONENT: f64 = {-1.0/(ERR_ORDER as f64 + 1.0)};
}

impl<const ERR_ORDER: usize, const E: usize> RKController<f64,E,1> for AdaptiveController<ERR_ORDER, E>
{

    type Config = AdaptiveRKConfig;

    fn get_next_step<const D: usize>(y1: &SVector<f64,D>, e: &[SVector<f64,D>; E], y0: &SVector<f64,D>, h: f64, cfg: &AdaptiveRKConfig) -> (bool, f64)
    {
        let err = e[0];
        let scale = (y0.abs().sup(&(y1.abs())) * cfg.rtol).add_scalar(cfg.atol);
        let err_norm = norm(&err, &scale);

        (err_norm <= 1.0,compute_new_h(err_norm, h, cfg.safety, cfg.max_step, cfg.min_step, cfg.max_factor, cfg.min_factor, Self::ERROR_EXPONENT))
    }
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SVector<f64,D>, f0: &SVector<f64,D>, cfg: &AdaptiveRKConfig) -> f64
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        select_initial_timestep(ode, &(|t| t), y0, t0, f0, cfg.atol, cfg.rtol, ERR_ORDER)
    }
}

pub struct PartitionedAdaptiveController<const ERR_ORDER: usize, const E: usize, const P: usize> {}

impl<const ERR_ORDER: usize, const E: usize, const P: usize> PartitionedAdaptiveController<ERR_ORDER, E, P>
{
    const ERROR_EXPONENT: f64 = {-1.0/(ERR_ORDER as f64 + 1.0)};
}

impl<const ERR_ORDER: usize, const E: usize, const P: usize> RKController<SVector<f64,P>,E,P> for PartitionedAdaptiveController<ERR_ORDER, E, P>
{

    type Config = AdaptiveRKConfig;

    fn get_next_step<const D: usize>(y1: &SMatrix<f64,D,P>, e: &[SMatrix<f64,D,P>; E], y0: &SMatrix<f64,D,P>, h: f64, cfg: &AdaptiveRKConfig) -> (bool, f64)
    {
        let err = e[0];
        let scale = (y0.abs().sup(&(y1.abs())) * cfg.rtol).add_scalar(cfg.atol);
        let err_norm = norm(&err, &scale);

        (err_norm <= 1.0,compute_new_h(err_norm, h, cfg.safety, cfg.max_step, cfg.min_step, cfg.max_factor, cfg.min_factor, Self::ERROR_EXPONENT))
    }
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SMatrix<f64,D,P>, f0: &SMatrix<f64,D,P>, cfg: &AdaptiveRKConfig) -> f64
        where F: Fn(SVector<f64,P>,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
    {
        select_initial_timestep(ode, &(|t| convert_t(t)), y0, t0, f0, cfg.atol, cfg.rtol, ERR_ORDER)
    }
}









