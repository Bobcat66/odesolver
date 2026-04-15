// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
use nalgebra::SVector;
pub trait DenseSolution<const D: usize> {
    fn eval(t: f64) -> SVector<f64,D>;
    fn low() -> f64;
    fn high() -> f64;
}