// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
/*
use nalgebra::SVector;

use crate::solvers::{common::{PointState, SolverState}, dense::{DenseInterpolant, DenseOutput}, runge_kutta::{const_config::{ButchersTableau, ExtendedButchersTableau}, rk_dense::RKInterpolant, rkimpl::{rk_stage_impl, rk_weight_impl}}};

// T is the RKKernel config type. The RKKernel is a stateless construct that only implements functions for a solver
pub trait RKKernel<T, Tableau, const S: usize, const D: usize> 
    where Tableau: ButchersTableau<S>
{
    fn initial_timestep_impl<F>(ode: &F, initial: &PointState<D>, cfg: &T) -> f64
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;

    // Returns a tuple containing two values: the step size and whether or not the step should be rejected
    fn time_control_impl<F>(ode: &F, state: &SolverState<D>, y1: &SVector<f64,D>, t_end: f64, k: &mut [SVector<f64, D>; S], cfg: &T) -> (bool,f64)
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;

    fn generate_dense_impl<F>(
        ode: &F,
        state: &SolverState<D>, 
        stages: &Vec<[SVector<f64,D>; S]>,
        k: &mut [SVector<f64, D>; S],
        cfg: &T
    ) -> DenseOutput<D>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;

    fn solve_impl<F>(
        k: &mut [SVector<f64, D>; S],
        ode: &F,
        initial: PointState<D>,
        t_end: f64,
        cfg: &T
    ) -> Vec<(f64,SVector<f64,D>)> 
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        let mut state = SolverState {
            point: initial,
            h: Self::initial_timestep_impl(ode, &initial, cfg),
            points: Vec::new(),
            steps: 0
        };
        state.points.push((state.point.t,state.point.y));
        while state.point.t < t_end {
            rk_stage_impl::<Tableau,S,D,_>(ode, &state, k);
            let y1 = rk_weight_impl::<Tableau,S,D>(&state, k);
            let time_control = Self::time_control_impl(ode, &state, &y1, t_end, k, cfg);
            // Accept or reject step
            if time_control.0 {
                state.point.y = y1;
                state.point.t += state.h;
                state.point.f = if Tableau::FSAL {k[S - 1]} else {ode(state.point.t,&state.point.y)};
                state.points.push((state.point.t,state.point.y));
                state.steps += 1;
            }
            state.h = time_control.1;
        }
        state.points
    }

    fn solve_dense_impl<F>(
        k: &mut [SVector<f64, D>; S],
        ode: &F,
        initial: PointState<D>,
        t_end: f64,
        cfg: &T
    ) -> (Vec<(f64,SVector<f64,D>)>,DenseOutput<D>)
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        let mut state = SolverState {
            point: initial,
            h: Self::initial_timestep_impl(ode, &initial, cfg),
            points: Vec::new(),
            steps: 0
        };
        let mut stages: Vec<[SVector<f64,D>; S]> = Vec::new();
        state.points.push((state.point.t,state.point.y));
        while state.point.t < t_end {
            rk_stage_impl::<Tableau,S,D,_>(ode, &state, k);

            let y1 = rk_weight_impl::<Tableau,S,D>(&state, k);
            let time_control = Self::time_control_impl(ode, &state, &y1, t_end, k, cfg);
            // Accept or reject step
            if time_control.0 {
                state.point.y = y1;
                state.point.t += state.h;
                state.point.f = if Tableau::FSAL {k[S - 1]} else {ode(state.point.t,&state.point.y)};
                state.points.push((state.point.t,state.point.y));
                stages.push((*k).clone());
                state.steps += 1;
            }
            state.h = time_control.1;
        }
        stages.push((*k).clone());
        let dense = Self::generate_dense_impl(ode, &state, &stages, k, cfg);
        (state.points,dense)
    }
}
    */