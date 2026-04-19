// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
use std::{f32::consts::E, marker::PhantomData};

use nalgebra::SVector;

use crate::{algebra::{mapping::Mapping, polynomial::Polynomial}, solvers::{DenseInterpolant, common::{norm, select_initial_timestep}, runge_kutta::rk_method::{RKController, RKInterpolator, RKMethod}}};

// Controller
#[derive(Copy, Clone)]
pub struct FixedRKConfig {
    pub h: f64
}

impl Default for FixedRKConfig {
    fn default() -> Self 
    {
        Self {
            h: 0.01
        }
    }
}

pub struct FixedRKController {}

impl RKController<0> for FixedRKController {
    type Config = FixedRKConfig;
    fn get_next_step<const D: usize>(y1: &SVector<f64, D>, e: &[SVector<f64,D>; 0], y0: &SVector<f64,D>, h: f64, cfg: &FixedRKConfig) -> (bool, f64)
    {
        (true,cfg.h)
    }
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SVector<f64,D>, f0: &SVector<f64,D>, cfg: &FixedRKConfig) -> f64
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        cfg.h
    }
}

// Linear Interpolator.

pub struct LinearInterpolant<const D: usize> {
    t0: f64,
    t1: f64,
    h: f64,
    y0: SVector<f64, D>,
    y1: SVector<f64, D>,
    dy: SVector<f64, D>
}

impl<const D: usize> LinearInterpolant<D> {
    fn eval_impl(&self, theta: f64) -> SVector<f64, D>
    {
        self.y0 + (self.dy * theta)
    }

    fn get_theta(&self, t: f64) -> f64 {
        (t-self.t0)/self.h
    }

    pub fn new(t0: f64, t1: f64, y0: SVector<f64,D>, y1: SVector<f64,D>) -> Self {
        Self {
            t0: t0,
            t1: t1,
            h: t1 - t0,
            y0: y0,
            y1: y1,
            dy: y1 - y0
        }
    }
}

impl<const D: usize> DenseInterpolant<D> for LinearInterpolant<D> {
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
