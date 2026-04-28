// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SMatrix;

use crate::solvers::runge_kutta::rkimpl::common::{RKController, RKInterpolator, RKStepper, RKTableau};

pub fn rk_step_impl<
    ODE,Consumer,TimeConverter,
    T,
    const N: usize,
    const S: usize,
    const E: usize,
    const D: usize,
    const P: usize,
    Tableau,
    Controller,
    Stepper
>(
    ode: &ODE,
    time_converter: &TimeConverter,
    t: &mut f64,
    y: &mut SMatrix<f64,D,P>,
    f: &mut SMatrix<f64,D,P>,
    h: &mut f64,
    k: &mut [SMatrix<f64,D,P>; S], 
    e: &mut [SMatrix<f64,D,P>; E],
    cfg: &Controller::Config,
    y_start: &SMatrix<f64,D,P>, 
    t_start: f64, 
    t_end: f64, 
    verbose: bool
) -> (f64, SMatrix<f64,D,P>)
    where ODE: Fn(T,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>,
    TimeConverter: Fn(f64) -> T,
    Tableau: RKTableau<T,N,S,E,P>,
    Controller: RKController<T,E,P>,
    Stepper: RKStepper<Tableau,T,N,S,E,P>
{
    let mut accepted = false;
    let mut new_t = self.t;
    let mut new_y = self.y;
    while !accepted {
        new_t = self.t + self.h;
        new_y = PRKStepper::<Method,P,S,E>::step_partitioned(&mut self.k, &mut self.e, &self.ode, self.t, &self.y, &self.f, self.h);
        let time_control = PRKMethodController::<Method,P,S,E>::get_next_step(&new_y, &self.e, &self.y, self.h, &self.cfg);
        accepted = time_control.0;
        self.h = time_control.1; 
    }
    self.y = new_y;
    self.t = new_t;
    self.f = if Method::FSAL {self.k[S-1]} else {(self.ode)(&SVector::<f64,P>::repeat(self.t), &self.y)};
    (self.t,self.y)
}
// F is the type of the function defining the ODE, T is the type of a function that can convert between f64 and time type
pub fn rk_solve_impl<
    ODE,Consumer,TimeConverter,
    T,
    const N: usize,
    const S: usize,
    const E: usize,
    const D: usize,
    const P: usize,
    Tableau,
    Controller,
    Stepper
>(
    ode: &ODE,
    consumer: &mut Consumer,
    time_converter: &TimeConverter,
    k: &mut [SMatrix<f64,D,P>; S], 
    e: &mut [SMatrix<f64,D,P>; E],
    cfg: &Controller::Config,
    y_start: &SMatrix<f64,D,P>, 
    t_start: f64, 
    t_end: f64, 
    verbose: bool
) -> Vec<(f64,SMatrix<f64,D,P>)>
    where ODE: Fn(T,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>,
    Consumer: FnMut(&[SMatrix<f64,D,P>; S]) -> (),
    TimeConverter: Fn(f64) -> T,
    Tableau: RKTableau<T,N,S,E,P>,
    Controller: RKController<T,E,P>,
    Stepper: RKStepper<Tableau,T,N,S,E,P>
{
    let mut t = t_start;
    let mut y = *y_start;
    let mut f = ode(time_converter(t_start), y_start);
    let mut h = Controller::select_initial_timestep(ode, t_start, y_start, &f, cfg);
    let mut points: Vec<(f64,SMatrix<f64,D,P>)> = Vec::new();
    points.push((t_start,*y_start));
    while t < t_end {
        h = (t_end - t).min(h);
        let new_t = t + h;
        if verbose {
            println!("t={}, h={}, evaluating at t={}",t,h,new_t);
        }
        let res = Stepper::step(k, e, ode, t, &y, &f, h);
        if verbose {
            println!("y(t)={:?}",res)
        }
        let time_control = Controller::get_next_step(&res,e, &y, h, cfg);
        if time_control.0 {
            y = res;
            t = new_t;
            f = if Tableau::FSAL {k[S-1]} else {ode(time_converter(new_t), &y)};
            points.push((t,y));
            consumer(k);
        }
        h = time_control.1;
        if verbose {
            println!("{}",if time_control.0 {"ACCEPTED"} else {"REJECTED"})
        }
    }
    points
}
