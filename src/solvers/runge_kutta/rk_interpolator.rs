// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SVector;

use crate::solvers::{dense::{DenseInterpolant, DenseOutput}, runge_kutta::rk_dense::RKInterpolant};

pub trait RKInterpolator<const S: usize> {
    fn interpolate_dense<const D: usize>(steps: usize, points: &Vec<(f64,SVector<f64,D>)>, stages: &Vec<[SVector<f64,D>; S]>) -> DenseOutput<D>;
}
