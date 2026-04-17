// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SVector;

use crate::solvers::dense::DenseInterpolant;
use crate::algebra::polynomial::Polynomial;
use crate::algebra::mapping::Mapping;
pub struct RKInterpolant<const S: usize, const P: usize, const D: usize> {
    t0: f64,
    t1: f64,
    h: f64,
    y0: SVector<f64, D>,
    k: [SVector<f64,D>; S],
    b: [Polynomial<f64,P>; S]
}

impl<const S: usize, const P: usize, const D: usize> RKInterpolant<S,P,D> {
    fn eval_impl(&self, theta: f64) -> SVector<f64, D>
    {
        let mut y_theta = self.y0.clone();
        for i in 0..S {
            y_theta += self.h * self.b[i].eval(theta) * self.k[i];
        }
        y_theta
    }

    fn get_theta(&self, t: f64) -> f64 {
        (t-self.t0)/self.h
    }

    pub fn new(t0: f64, t1: f64, y0: SVector<f64,D>, k: [SVector<f64,D>; S], p: [[f64; P]; S]) -> Self {
        Self {
            t0: t0,
            t1: t1,
            h: t1 - t0,
            y0: y0,
            k: k,
            b: std::array::from_fn(|i| Polynomial { a: p[i] })
        }
    }
}

impl<const S: usize, const P: usize, const D: usize> DenseInterpolant<D> for RKInterpolant<S, P, D> {
    fn eval(&self, t: f64) -> SVector<f64,D> {
        self.eval_impl(self.get_theta(t))
    }
    fn low_t(&self) -> f64 {
        self.t0
    }
    fn high_t(&self) -> f64 {
        self.t1
    }
    fn y0(&self) -> SVector<f64,D> {
        self.y0
    }
}