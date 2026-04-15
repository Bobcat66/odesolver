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
}

pub struct DenseOutput<const D: usize> {
    segments: Vec<(f64,Box<dyn DenseInterpolant<D>>)>
}

impl<const D: usize> DenseOutput<D> {
    pub fn new(segments: Vec<(f64,Box<dyn DenseInterpolant<D>>)>) -> Self {
        Self {segments: segments}
    }
    
    fn find_index(&self, t: f64) -> usize {
        self.segments
            .partition_point(|(ti, _)| *ti <= t)
            .saturating_sub(1)
    }

    pub fn eval(&self, t: f64) -> SVector<f64,D> {
        self.segments[self.find_index(t)].1.eval(t)
    }
}