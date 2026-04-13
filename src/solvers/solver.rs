// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
use nalgebra::SVector;
use std::vec::Vec;
pub trait Solver<const D: usize> {
    // Returns a tuple in the form (time, state)
    fn solve_single<F>(&mut self, ode: &F, y0: &SVector<f64,D>, t_start: f64, t_end: f64) -> (f64,SVector<f64,D>)
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;
    // Returns a vector of all points sampled during computation, stored in tuples form (time, state)
    fn solve<F>(&mut self, ode: &F, y0: &SVector<f64,D>, t_start: f64, t_end: f64) -> Vec<(f64,SVector<f64,D>)> 
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;
}