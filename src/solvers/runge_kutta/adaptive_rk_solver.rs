use std::marker::PhantomData;
/* 
use crate::solvers::common::{PointState, SolverState, select_initial_timestep};
use crate::solvers::dense::{DenseInterpolant, DenseOutput};
use crate::solvers::runge_kutta::rk_dense::RKInterpolant;
use crate::solvers::runge_kutta::rk_kernel::RKKernel;
use crate::solvers::solver::{Solver,DenseSolver};
use crate::solvers::runge_kutta::rkimpl::{ark_err_impl};
use nalgebra::SVector;

use crate::solvers::runge_kutta::const_config::*;

pub struct AdaptiveRKKernelConfig {
    pub atol: f64, // absolute tolerance
    pub rtol: f64, // normalized tolerance
    pub safety: f64, // safety value to reduce overshoot
    pub min_clamp: f64, // minimum timestep clamp
    pub max_clamp: f64, // maximum timestep clamp
    pub min_factor: f64, // minimum timestep change factor
    pub max_factor: f64, // maximum timestep change factor
}

impl Default for AdaptiveRKKernelConfig {
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
struct AdaptiveRKKernel<Tableau, const S: usize, const P: usize, const D: usize> 
    where Tableau: ExtendedButchersTableau<S,P>
{
    _marker: PhantomData<Tableau>
}

impl<Tableau, const S: usize, const P: usize, const D: usize>  AdaptiveRKKernel<Tableau, S, P, D>
    where Tableau: ExtendedButchersTableau<S,P>
{
    const ERROR_EXPONENT: f64 = {-1.0/(Tableau::EMBEDDED_ORDER as f64 + 1.0)};
}

impl<Tableau, const S: usize, const P: usize, const D: usize> RKKernel<AdaptiveRKKernelConfig, Tableau, S, D> for AdaptiveRKKernel<Tableau, S, P, D>
    where Tableau: ExtendedButchersTableau<S,P>
{
    fn initial_timestep_impl<F>(ode: &F, initial: &PointState<D>, cfg: &AdaptiveRKKernelConfig) -> f64
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D> 
    {
        select_initial_timestep(ode, &initial.y, initial.t, &initial.f, cfg.atol, cfg.rtol, Tableau::EMBEDDED_ORDER)
    }

    // Returns a tuple containing two values: the step size and whether or not the step should be rejected
    fn time_control_impl<F>(ode: &F, state: &SolverState<D>, y1: &SVector<f64,D>, t_end: f64, k: &mut [SVector<f64, D>; S], cfg: &AdaptiveRKKernelConfig) -> (bool,f64)
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        // compute error
        let scale = (state.point.y.abs().sup(&(y1.abs())) * cfg.rtol).add_scalar(cfg.atol);
        let error = ark_err_impl::<Tableau,S,P,D>(state, k);
        let err_norm = error.component_div(&scale).norm();

        // recalculate stepsize
        let mut new_h = (state.h * cfg.safety * err_norm.powf(Self::ERROR_EXPONENT)).clamp(cfg.min_clamp,cfg.max_clamp);
        let factor = new_h/state.h;

        if factor > cfg.max_factor {
            new_h = state.h * cfg.max_factor;
        } else if factor < cfg.min_factor {
            new_h = state.h * cfg.min_factor;
        }

        if (state.point.t + state.h) + new_h > t_end {
            new_h = t_end - (state.point.t + state.h);
        }

        (err_norm <= 1.0,new_h)
    }

    fn generate_dense_impl<F>(
        ode: &F,
        state: &SolverState<D>, 
        stages: &Vec<[SVector<f64,D>; S]>,
        k: &mut [SVector<f64, D>; S],
        cfg: &AdaptiveRKKernelConfig
    ) -> DenseOutput<D>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D> 
    {
    
        let mut segments: Vec<(f64,Box<dyn DenseInterpolant<D>>)> = Vec::new();
        for i in 0..(state.steps-1) {
            segments.push(
                (
                    state.points[i].0,
                    Box::new(RKInterpolant::new(state.points[i].0,state.points[i + 1].0, state.points[i].1, stages[i], Tableau::P))
                )
            );
        }
        DenseOutput::<D>::new(segments)
    }
}


// S is the number of stages, D is the dims
pub struct AdaptiveRKSolver<Tableau, const S: usize, const P: usize, const D: usize> 
    where Tableau: ExtendedButchersTableau<S,P>
{
    pub cfg: AdaptiveRKKernelConfig,
    k: [SVector<f64,D>; S],
    _marker: PhantomData<Tableau>
}

impl<Tableau, const S: usize, const P: usize, const D: usize> Default for AdaptiveRKSolver<Tableau,S,P,D> 
    where Tableau: ExtendedButchersTableau<S,P> 
{
    fn default() -> Self {
        Self {
            cfg: AdaptiveRKKernelConfig::default(),
            k: [SVector::<f64,D>::zeros(); S],
            _marker: PhantomData
        }
    }
}

impl<Tableau, const S: usize, const P: usize, const D: usize> Solver<D> for AdaptiveRKSolver<Tableau,S,P,D>
    where Tableau: ExtendedButchersTableau<S,P> 
{
    fn solve<F>(&mut self, ode: &F, y0: &SVector<f64,D>, t_start: f64, t_end: f64) -> Vec<(f64,SVector<f64,D>)> 
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        AdaptiveRKKernel::<Tableau,S,P,D>::solve_impl(
            &mut self.k, 
            ode, 
            PointState { 
                t: t_start,
                y: *y0,
                f: ode(t_start, y0) 
            }, 
            t_end, 
            &self.cfg
        )
    }
}

impl<Tableau, const S: usize, const P: usize, const D: usize> DenseSolver<D> for AdaptiveRKSolver<Tableau,S,P,D>
    where Tableau: ExtendedButchersTableau<S,P> 
{
    fn solve_dense<F>(&mut self, ode: &F, y0: &SVector<f64,D>, t_start: f64, t_end: f64) -> (Vec<(f64,SVector<f64,D>)>,DenseOutput<D>)
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        AdaptiveRKKernel::<Tableau,S,P,D>::solve_dense_impl(
            &mut self.k, 
            ode, 
            PointState { 
                t: t_start,
                y: *y0,
                f: ode(t_start, y0) 
            }, 
            t_end, 
            &self.cfg
        )
    }
}
    */