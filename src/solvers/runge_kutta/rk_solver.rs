// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::marker::PhantomData;

use nalgebra::SVector;

use crate::solvers::{dense::DenseOutput, runge_kutta::{rk_controller::RKController, rk_interpolator::RKInterpolator, rk_stepper::{ButchersTableau, RKStepper}}, solver::{DenseSolver, Solver}};

// T is the RKKernel config type. The RKKernel is a stateless construct that only implements functions for a solver
pub struct RKSolver<Controller, Interpolator, Tableau, const S: usize, const O: usize, const D: usize> 
    where Tableau: ButchersTableau<S,O>,
    Controller: RKController<O>,
    Interpolator: RKInterpolator<S>
{
    pub cfg: Controller::Config,
    k: [SVector<f64,D>; S],
    o: [SVector<f64,D>; O],
    _marker0: PhantomData<Interpolator>,
    _marker1: PhantomData<Tableau>
}

impl<Controller, Interpolator, Tableau, const S: usize, const O: usize, const D: usize> RKSolver<Controller,Interpolator,Tableau,S,O,D>
    where Tableau: ButchersTableau<S,O>,
    Controller: RKController<O>,
    Interpolator: RKInterpolator<S>
{
    pub fn new(cfg: Controller::Config) -> Self {
        Self {
            cfg: cfg,
            k: [SVector::<f64,D>::zeros(); S],
            o: [SVector::<f64,D>::zeros(); O],
            _marker0: PhantomData,
            _marker1: PhantomData
        }
    }

    // Returns (steps,Vec(t,y))
    fn solve_impl<F,C>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64, stage_consumer: &C) -> (usize,Vec<(f64,SVector<f64,D>)>)
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>,
        C: FnMut(&[SVector<f64,D>; S]) -> ()
    {
        let mut k = &mut self.k;
        let mut o = &mut self.o;
        let cfg = &self.cfg;

        let mut t = t_start;
        let mut y = *y_start;
        let mut f = ode(t_start,y_start);
        let mut h = Controller::select_initial_timestep(ode, t_start, y_start, &f, cfg);
        let mut points: Vec<(f64,SVector<f64,D>)> = Vec::new();
        points.push((t_start,*y_start));
        let mut steps: usize = 0;
        while t < t_end {
            let new_t = t + h;
            RKStepper::<Tableau,S,O>::step(k, o, ode, t, &y, &f, h);
            let time_control = Controller::get_next_step(o, &y, new_t, t, h, t_end, cfg);
            if time_control.0 {
                y = o[0];
                t = new_t;
                f = if Tableau::FSAL {k[S - 1]} else {ode(t,&y)};
                points.push((t,y));
                stage_consumer(k);
                steps += 1;
            }
            h = time_control.1;
        }
        stage_consumer(k);
        (steps,points)
    }
}

impl<Controller, Interpolator, Tableau, const S: usize, const O: usize, const D: usize> Default for RKSolver<Controller,Interpolator,Tableau,S,O,D>
    where Tableau: ButchersTableau<S,O>,
    Controller: RKController<O>,
    Interpolator: RKInterpolator<S>
{
    fn default() -> Self {
        Self::new(Controller::Config::default())
    }
}

impl<Controller, Interpolator, Tableau, const S: usize, const O: usize, const D: usize> Solver<D> for RKSolver<Controller,Interpolator,Tableau,S,O,D>
    where Tableau: ButchersTableau<S,O>,
    Controller: RKController<O>,
    Interpolator: RKInterpolator<S>
{
    fn solve<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64) -> Vec<(f64,SVector<f64,D>)> 
            where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        self.solve_impl(ode, y_start, t_start, t_end, &(|stage: &[SVector<f64,D>; S]| ())).1
    }
}

impl<Controller, Interpolator, Tableau, const S: usize, const O: usize, const D: usize> DenseSolver<D> for RKSolver<Controller,Interpolator,Tableau,S,O,D>
    where Tableau: ButchersTableau<S,O>,
    Controller: RKController<O>,
    Interpolator: RKInterpolator<S>
{
    fn solve_dense<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64) -> (Vec<(f64,SVector<f64,D>)>,DenseOutput<D>) 
            where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        let mut stages: Vec<[SVector<f64,D>; S]> = Vec::new();
        let result = self.solve_impl(ode, y_start, t_start, t_end, &(|stage: &[SVector<f64,D>; S]| stages.push(*stage)));
        (result.1,Interpolator::interpolate_dense(result.0, &result.1, &stages))
    }
}