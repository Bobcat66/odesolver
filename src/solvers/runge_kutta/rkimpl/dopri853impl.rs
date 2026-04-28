// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

// This is an implementation of Hairer's DOPRI8(5,3) method, first implemented in Fortran here: https://www.unige.ch/~hairer/prog/nonstiff/dop853.f

use std::marker::PhantomData;

use nalgebra::SVector;

use crate::solvers::{DenseInterpolant, common::select_initial_timestep, runge_kutta::{adaptive_rk::{AdaptiveRKConfig, compute_new_h}, rk_common::{PRKController, PRKInterpolator, PRKMethod}, rkimpl::{adaptive::AdaptiveRKConfig, common::RKController}}};

// Controller
pub struct DOPRI853Controller<const ERR_ORDER: usize> {}

impl<const ERR_ORDER: usize> DOPRI853Controller<ERR_ORDER>
{
    const ERROR_EXPONENT: f64 = {-1.0/(ERR_ORDER as f64 + 1.0)};
}


impl<const ERR_ORDER: usize> RKController<f64,2, 1> for DOPRI853Controller<ERR_ORDER>
{

    type Config = AdaptiveRKConfig;

    fn get_next_step<const D: usize>(y1: &SVector<f64, D>, e: &[SVector<f64,D>; 2], y0: &SVector<f64,D>, h: f64, cfg: &AdaptiveRKConfig) -> (bool, f64)
    {
        let scale = (y0.abs().sup(&(y1.abs())) * cfg.rtol).add_scalar(cfg.atol);
        let err5 = e[0].component_div(&scale);
        let err3 = e[1].component_div(&scale);
        let err5_norm_2 = err5.norm().powi(2);
        let err3_norm_2 = err3.norm().powi(2);

        if err5_norm_2 == 0.0 && err3_norm_2 == 0.0 {
            // Error norm = 0
            return (true,compute_new_h(0.0, h, cfg.safety, cfg.max_step, cfg.min_step, cfg.max_factor, cfg.min_factor, Self::ERROR_EXPONENT))
        }
        let denom = err5_norm_2 + 0.01 * err3_norm_2;
        let err_norm = h.abs() * err5_norm_2 / (denom * D as f64).sqrt();

        (err_norm <= 1.0,compute_new_h(err_norm, h, cfg.safety, cfg.max_step, cfg.min_step, cfg.max_factor, cfg.min_factor, Self::ERROR_EXPONENT))
    }
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SVector<f64,D>, f0: &SVector<f64,D>, cfg: &AdaptiveRKConfig) -> f64
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        select_initial_timestep(ode, &(|t| t), y0, t0, f0, cfg.atol, cfg.rtol, ERR_ORDER)
    }
}

// Interpolator
pub struct DOPRI853Interpolant<const D: usize> {
    t0: f64,
    t1: f64,
    h: f64,
    y0: SVector<f64,D>,
    y1: SVector<f64,D>,
    r: [SVector<f64,D>; 7]
}

impl<const D: usize> DOPRI853Interpolant<D> {

    fn eval_impl(&self, theta: f64) -> SVector<f64,D> {
        self.y0 + theta * (
            self.r[0] + (1.0 - theta) * (
                self.r[1] + theta * (
                    self.r[2] + (1.0 - theta) * (
                        self.r[3] + theta * (
                            self.r[4] + (1.0 - theta) * (
                                self.r[5] + theta * self.r[6]
                            )
                        )
                    )
                )
            )
        )
    }

    fn get_theta(&self, t: f64) -> f64 {
        (t-self.t0)/self.h
    }

    pub fn new(t0: f64, t1: f64, y0: SVector<f64,D>, y1: SVector<f64, D>, r: [SVector<f64,D>; 7]) -> Self {
        Self {
            t0: t0,
            t1: t1,
            h: t1 - t0,
            y0: y0,
            y1: y1,
            r: r
        }
    }
}

impl<const D: usize> DenseInterpolant<D,1> for DOPRI853Interpolant<D> {
    fn eval(&self, t: f64) -> SVector<f64,D> {
        self.eval_impl(self.get_theta(t))
    }
    fn low_t(&self) -> f64 {
        self.t0
    }
    fn high_t(&self) -> f64 {
        self.t1
    }
    fn y0(&self) -> SVector<f64,D> {
        self.y0
    }
    fn y1(&self) -> SVector<f64,D> {
        self.y1
    }
}

pub struct DOPRI853Interpolator {}

impl PRKInterpolator<1,13> for DOPRI853Interpolator
{
    type InterpolantType<const D: usize> = DOPRI853Interpolant<D>;
    fn interpolate_stage<F, const D: usize>(ode: &F, t0: f64, t1: f64, y0: &SVector<f64,D>, y1: &SVector<f64,D>, stage: &[SVector<f64,D>; 13]) -> Self::InterpolantType<D> 
        where F: Fn(&SVector<f64,1>,&SVector<f64,D>) -> SVector<f64,D>
    {
        let mut k: [SVector<f64, D>; 16] = [SVector::<f64, D>::zeros(); 16];
        let h = t1 - t0;
        k[0..13].copy_from_slice(stage);
        for i in 0..3 {
            let mut dy = SVector::<f64, D>::zeros();
            for j in 0..13 + i {
                dy += k[j] * A_EXTRA[i][j] * h;
            }
            k[13 + i] = ode(&SVector::<f64,1>::repeat(t0 + C_EXTRA[i] * h), &(y0 + dy));
        }
        let mut r = [SVector::<f64,D>::zeros(); 7];
        let delta_y = y1 - y0;
        r[0] = delta_y;
        r[1] = h * k[0] - delta_y;
        r[2] = 2.0 * delta_y - h * (k[12] - k[0]); // valid because of FSAL
        for i in 3..7 {
            for j in 0..16 {
                r[i] += k[j] * h * DENSE[i - 3][j];
            }
        }
        DOPRI853Interpolant::new(t0, t1, *y0, *y1, r)
    }
}
// Method

impl PRKMethod<1,13,2> for DOPRI853 {

    type Controller = DOPRI853RKController<Self>;
    type Interpolator = DOPRI853Interpolator;

    
    const FSAL: bool = true;
    
    const ORDER: usize = 8;
    const ERR_ORDER: usize = 7;
}

