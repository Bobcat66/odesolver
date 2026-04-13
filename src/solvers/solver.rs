// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
use nalgebra::SVector;
use std::vec::Vec;
pub trait Solver<const D: usize> {
    // Returns a tuple in the form (state, time)
    fn solve<F>(&self, ode: &F, y0: &SVector<f64,D>, t_start: f64, t_end: f64) -> (SVector<f64,D>,f64)
        where F: Fn(&SVector<f64,D>,f64) -> SVector<f64,D>;
    // Returns a vector of all points sampled during computation, stored in tuples form (state, time)
    fn solve_dense<F>(&self, ode: &F, y0: &SVector<f64,D>, t_start: f64, t_end: f64) -> Vec<(SVector<f64,D>, f64)> 
        where F: Fn(&SVector<f64,D>,f64) -> SVector<f64,D>;
}