// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use crate::vector_field::VectorField;
use nalgebra::SVector;

pub fn iterate_euler<const D: usize>(ode: &VectorField<D>, x_t0: &SVector<f64,D>, t0: f64, delta_t: f64) -> SVector<f64,D> {
    x_t0 + (delta_t * ode.eval(x_t0,t0))
}