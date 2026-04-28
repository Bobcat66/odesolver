// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::marker::PhantomData;

use nalgebra::{SMatrix,SVector};

use crate::solvers::{DenseInterpolant, runge_kutta::rkimpl::common::RKController};

// Controller
#[derive(Copy, Clone)]
pub struct FixedConfig {
    pub h: f64
}

impl Default for FixedConfig {
    fn default() -> Self 
    {
        Self {
            h: 0.01
        }
    }
}

pub struct FixedController<T, const P: usize> {
    _marker: PhantomData<T>
}

impl<T,const P: usize> RKController<T,0,P> for FixedController<T,P>{
    type Config = FixedConfig;
    fn get_next_step<const D: usize>(y1: &SMatrix<f64, D, P>, e: &[SMatrix<f64, D, P>; 0], y0: &SMatrix<f64, D, P>, h: f64, cfg: &FixedConfig) -> (bool, f64)
    {
        (true,cfg.h)
    }
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SMatrix<f64, D, P>, f0: &SMatrix<f64, D, P>, cfg: &FixedConfig) -> f64
        where F: Fn(T,&SMatrix<f64, D, P>) -> SMatrix<f64, D, P>
    {
        cfg.h
    }
}

// Linear Interpolator.

pub struct LinearInterpolant<const D: usize, const P: usize> {
    t0: f64,
    t1: f64,
    h: f64,
    y0: SMatrix<f64, D, P>,
    y1: SMatrix<f64, D, P>,
    dy: SMatrix<f64, D, P>
}

impl<const D: usize, const P: usize> LinearInterpolant<D, P> {
    fn eval_impl(&self, theta: f64) -> SMatrix<f64, D, P>
    {
        self.y0 + (self.dy * theta)
    }

    fn get_theta(&self, t: f64) -> f64
    {
        (t-self.t0)/self.h
    }

    pub fn new(t0: f64, t1: f64, y0: SMatrix<f64, D, P>, y1: SMatrix<f64, D, P>) -> Self {
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

impl<const P: usize, const D: usize> DenseInterpolant<D,P> for LinearInterpolant<D,P> {
    fn eval(&self, t: f64) -> SMatrix<f64, D, P>
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
    fn y0(&self) -> SMatrix<f64, D, P>
    {
        self.y0
    }
    fn y1(&self) -> SMatrix<f64, D, P>
    {
        self.y1
    }
}

pub type ExplicitFixedController = FixedController<f64, 1>;
pub type PartitionedFixedController<const P: usize> = FixedController<SVector<f64,P>,P>;
