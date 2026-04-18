// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
use nalgebra::SVector;
use std::vec::Vec;
pub trait DenseInterpolant<const D: usize> {
    fn eval(&self, t: f64) -> SVector<f64,D>;
    fn low_t(&self) -> f64;
    fn high_t(&self) -> f64;
    fn y0(&self) -> SVector<f64,D>;
    fn y1(&self) -> SVector<f64,D>;
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