// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SVector;
use std::vec::Vec;

pub mod runge_kutta;
pub mod common;

// Dense Solutions
pub trait DenseInterpolant<const D: usize> {
    fn eval(&self, t: f64) -> SVector<f64,D>;
    fn low_t(&self) -> f64;
    fn high_t(&self) -> f64;
    fn y0(&self) -> SVector<f64,D>;
    fn y1(&self) -> SVector<f64,D>;
}

pub trait DensePartitionedInterpolant<const C: usize, const D: usize>
{
    fn eval(&self, t: f64) -> [SVector<f64,D>; C];
    fn low_t(&self) -> f64;
    fn high_t(&self) -> f64;
    fn y0(&self) -> [SVector<f64,D>; C];
    fn y1(&self) -> [SVector<f64,D>; C];
}

pub struct DenseOutput<T, const D: usize> 
    where T: DenseInterpolant<D>
{
    segments: Vec<T>
}

impl<T, const D: usize> DenseOutput<T, D>
    where T: DenseInterpolant<D>
{
    pub fn new(segments: Vec<T>) -> Self {
        Self {segments: segments}
    }
    
    fn find_index(&self, t: f64) -> usize {
        self.segments
            .partition_point(|segment| segment.low_t() <= t)
            .saturating_sub(1)
    }

    pub fn eval(&self, t: f64) -> SVector<f64,D> {
        self.segments[self.find_index(t)].eval(t)
    }
}

pub struct DensePartitionedOutput<T, const C: usize, const D: usize> 
    where T: DensePartitionedInterpolant<C,D>
{
    segments: Vec<T>
}

impl<T, const C: usize, const D: usize> DensePartitionedOutput<T, C, D>
    where T: DensePartitionedInterpolant<C, D>
{
    pub fn new(segments: Vec<T>) -> Self {
        Self {segments: segments}
    }
    
    fn find_index(&self, t: f64) -> usize {
        self.segments
            .partition_point(|segment| segment.low_t() <= t)
            .saturating_sub(1)
    }

    pub fn eval(&self, t: f64) -> [SVector<f64,D>; C] {
        self.segments[self.find_index(t)].eval(t)
    }
}

// Lazy Solutions
pub trait LazySolution<const D: usize>
{
    fn next(&mut self) -> (f64, SVector<f64,D>);
}

pub trait LazyDenseSolution<T, const D: usize>
    where T: DenseInterpolant<D>
{
    fn next(&mut self) -> T;
}

pub trait LazyPartitionedSolution<const C: usize, const D: usize>
{
    fn next(&mut self) -> (f64,[SVector<f64,D>; C]);
}

pub trait LazyDensePartitionedSolution<T, const C: usize, const D: usize>
    where T: DensePartitionedInterpolant<C,D>
{
    fn next(&mut self) -> T;
}



// Solvers

pub trait Solver<const D: usize>
{
    // Returns a vector of all points sampled during computation, stored in tuples form (time, state)
    fn solve<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64, verbose: bool) -> Vec<(f64,SVector<f64,D>)> 
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;

    fn solve_lazy<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64) -> impl LazySolution<D>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;
}

pub trait DenseSolver<const D: usize> : Solver<D>
{
    type InterpolantType: DenseInterpolant<{D}>;

    fn solve_dense<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64, verbose: bool) -> (Vec<(f64,SVector<f64,D>)>,DenseOutput<Self::InterpolantType, D>)
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;

    fn solve_dense_lazy<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64) -> impl LazyDenseSolution<Self::InterpolantType,D>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>;
}

pub trait PartitionedSolver<const C: usize, const D: usize>
{
    fn solve<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64, t_end: f64, verbose: bool) -> Vec<(f64,[SVector<f64,D>; C])> 
        where F: Fn(f64,&SVector<f64,D>,&SVector<f64,D>) -> (SVector<f64,D>,SVector<f64,D>);

    fn solve_lazy<F>(&mut self, ode: &F, y_start: &SVector<f64,D>, t_start: f64) -> impl LazyPartitionedSolution<C,D>
        where F: Fn(f64,&SVector<f64,D>,&SVector<f64,D>) -> (SVector<f64,D>,SVector<f64,D>);
}