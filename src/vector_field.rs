// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SVector;
pub struct VectorField<const D: usize> {
    // Axes is an array of function pointers that accept a state and a time.
    axes: [fn(&SVector<f64,D>,f64) -> f64; D]
}

impl<const D: usize> VectorField<D> {

    pub fn new(funcs: [fn(&SVector<f64,D>,f64) -> f64; D]) -> Self {
        Self {
            axes: funcs
        }
    }

    pub fn eval(&self, point: &SVector<f64,D>, time: f64) -> SVector<f64,D> {
        let mut vec: SVector<f64,D> = SVector::zeros();

        for i in 0..D {
            vec[i] = (self.axes[i])(&point,time);
        }

        vec
    }
}