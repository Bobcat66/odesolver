// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SMatrix;
use std::vec::Vec;

pub mod runge_kutta;
pub mod common;

// We use matrices to allow for solving partitioned ODEs

// Dense Solutions

pub trait DenseInterpolant<const P: usize, const D: usize>
{
    fn eval(&self, t: f64) -> SMatrix<f64,D,P>;
    fn low_t(&self) -> f64;
    fn high_t(&self) -> f64;
    fn y0(&self) -> SMatrix<f64,D,P>;
    fn y1(&self) -> SMatrix<f64,D,P>;
}

pub struct DenseOutput<I, const P: usize, const D: usize> 
    where I: DenseInterpolant<P,D>
{
    segments: Vec<I>,
}

impl<I, const P: usize, const D: usize> DenseOutput<I, P, D>
    where I: DenseInterpolant<P,D>
{
    pub fn new(segments: Vec<I>) -> Self {
        Self {segments: segments}
    }
    
    fn find_index(&self, t: f64) -> usize {
        self.segments
            .partition_point(|segment| segment.low_t() <= t)
            .saturating_sub(1)
    }

    pub fn eval(&self, t: f64) -> SMatrix<f64,D,P> {
        self.segments[self.find_index(t)].eval(t)
    }
}

pub trait LazySolution<const P: usize, const D: usize>
{
    fn next(&mut self) -> (f64,SMatrix<f64,D,P>);
}

pub trait LazyDenseSolution<I, const P: usize, const D: usize>
    where I: DenseInterpolant<P,D>
{
    fn next(&mut self) -> impl DenseInterpolant<P,D>;
}

// Solvers

pub trait Solver<const P: usize, const D: usize>
{
    fn solve<F>(&mut self, ode: &F, y_start: &SMatrix<f64,D,P>, t_start: f64, t_end: f64, verbose: bool) -> Vec<(f64,SMatrix<f64,D,P>)> 
        where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>;

    fn solve_lazy<F>(&mut self, ode: &F, y_start: &SMatrix<f64,D,P>, t_start: f64) -> impl LazySolution<P,D>
        where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>;
}

pub trait DenseSolver<const P: usize, const D: usize> : Solver<P,D>
{
    type InterpolantType: DenseInterpolant<P,D>;

    fn solve_dense<F>(&mut self, ode: &F, y_start: &SMatrix<f64,D,P>, t_start: f64, t_end: f64, verbose: bool) -> (Vec<(f64,SMatrix<f64,D,P>)>,DenseOutput<Self::InterpolantType,P,D>)
        where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>;

    fn solve_dense_lazy<F>(&mut self, ode: &F, y_start: &SMatrix<f64,D,P>, t_start: f64) -> impl LazyDenseSolution<Self::InterpolantType,P,D>
        where F: Fn(f64,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>;
}
