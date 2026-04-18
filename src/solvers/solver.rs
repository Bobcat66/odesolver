// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
use nalgebra::SVector;
use std::vec::Vec;
use crate::solvers::dense::{DenseInterpolant, DenseOutput};

pub trait LazySolution<const D: usize>
{
    fn next(&mut self) -> (f64, SVector<f64,D>);
}

pub trait LazyDenseSolution<T, const D: usize>
    where T: DenseInterpolant<D>
{
    fn next(&mut self) -> T;
}
pub trait Solver<const D: usize>
{
    // Returns a vector of all points sampled during computation, stored in tuples form (time, state)
    fn solve<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64) -> Vec<(f64,SVector<f64,D>)> 
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;

    fn solve_lazy<F,C>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64) -> impl LazySolution<D>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;
}

pub trait DenseSolver<const D: usize> : Solver<D>
{
    type InterpolantType: DenseInterpolant<{D}>;

    fn solve_dense<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64) -> (Vec<(f64,SVector<f64,D>)>,DenseOutput<Self::InterpolantType, D>)
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;

    fn solve_dense_lazy<F,C>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64) -> impl LazyDenseSolution<Self::InterpolantType,D>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;
}