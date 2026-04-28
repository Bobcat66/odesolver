// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SMatrix;

pub mod simple;

pub trait ODE<T, const D: usize, const P: usize> {
    fn eval(&self, t: T, y: &SMatrix<f64,D,P>) -> SMatrix<f64,D,P>;
}