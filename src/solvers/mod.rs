// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::{SMatrix,SVector};
use std::vec::Vec;

pub mod runge_kutta;
pub mod common;

// We use matrices to allow for solving partitioned ODEs

// Dense Solutions

pub trait DenseInterpolant<const D: usize, const P: usize>
{
    fn eval(&self, t: f64) -> SMatrix<f64,D,P>;
    fn low_t(&self) -> f64;
    fn high_t(&self) -> f64;
    fn y0(&self) -> SMatrix<f64,D,P>;
    fn y1(&self) -> SMatrix<f64,D,P>;
}

pub struct DenseOutput<I, const D: usize, const P: usize> 
    where I: DenseInterpolant<D,P>
{
    segments: Vec<I>,
}

impl<I, const D: usize, const P: usize> DenseOutput<I, D, P>
    where I: DenseInterpolant<D,P>
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

pub trait LazyDenseSolution<I, const D: usize, const P: usize>
    where I: DenseInterpolant<D,P>
{
    fn next(&mut self) -> I;
}

// Solvers

pub trait Solver<T, const P: usize>
{
    type InterpolantType<const D: usize>: DenseInterpolant<D,P>;

    fn solve<F, const D: usize>(&mut self, ode: &F, y_start: &SMatrix<f64,D,P>, t_start: f64, t_end: f64, verbose: bool) -> Vec<(f64,SMatrix<f64,D,P>)> 
        where F: Fn(T,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>;

    fn solve_lazy<F, const D: usize>(&mut self, ode: &F, y_start: &SMatrix<f64,D,P>, t_start: f64) -> impl LazySolution<P,D>
        where F: Fn(T,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>;

    fn solve_dense<F, const D: usize>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64, verbose: bool) -> (Vec<(f64,SVector<f64,D>)>,DenseOutput<Self::InterpolantType<D>,D,P>)
        where F: Fn(T,&SVector<f64,D>) -> SVector<f64,D>;

    fn solve_dense_lazy<F, const D: usize>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64) -> impl LazyDenseSolution<Self::InterpolantType<D>,D,P>
        where F: Fn(T,&SVector<f64,D>) -> SVector<f64,D>;
}