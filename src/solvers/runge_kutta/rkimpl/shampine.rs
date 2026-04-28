// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

// Interpolator. This is based on Shampine's method for deriving polynomial interpolants

use std::marker::PhantomData;

use nalgebra::{SMatrix, SVector};

use crate::{algebra::polynomial::Polynomial, solvers::{DenseInterpolant, runge_kutta::rkimpl::common::RKInterpolator, }};

pub struct ShampineInterpolant<const K: usize, const S: usize, const D: usize, const P: usize>
{
    t0: f64,
    t1: f64,
    h: f64,
    y0: SMatrix<f64,D,P>,
    y1: SMatrix<f64,D,P>,
    k: [SMatrix<f64,D,P>; S],
    b: [Polynomial<f64,K>; S]
}

impl<const K: usize, const S: usize, const D: usize, const P: usize> ShampineInterpolant<K,S,D,P> 
{
    fn eval_impl(&self, theta: f64) -> SMatrix<f64, D, P>
    {
        let mut y_theta = self.y0.clone();
        for i in 0..S {
            y_theta += self.k[i] * self.h * self.b[i].eval(theta);
        }
        y_theta
    }

    fn get_theta(&self, t: f64) -> f64 {
        (t-self.t0.clone())/self.h.clone()
    }

    pub fn new(t0: f64, t1: f64, y0: SMatrix<f64,D,P>, y1: SMatrix<f64,D,P>, k: [SMatrix<f64,D,P>; S], p: [[f64; K]; S]) -> Self
    {
        Self {
            t0: t0,
            t1: t1,
            h: t1 - t0,
            y0: y0,
            y1: y1,
            k: k,
            b: std::array::from_fn(|i| Polynomial::new(p[i]))
        }
    }
}

impl<const K: usize, const S: usize, const D: usize, const P: usize> DenseInterpolant<D,P> for ShampineInterpolant<K,S,D,P>
{
    fn eval(&self, t: f64) -> SMatrix<f64,D,P>
    {
        self.eval_impl(self.get_theta(t))
    }
    fn low_t(&self) -> f64
    {
        self.t0
    }
    fn high_t(&self) -> f64
    {
        self.t1
    }
    fn y0(&self) -> SMatrix<f64,D,P>
    {
        self.y0
    }
    fn y1(&self) -> SMatrix<f64,D,P>
    {
        self.y1
    }
}
pub trait ShampineConfig<const K: usize, const S: usize> {
    const W: [[f64; K]; S]; // shampine polynomial weights
}

pub struct ShampineInterpolator<Shampine, const K: usize, const S: usize>
    where Shampine: ShampineConfig<K,S>
{
    _marker: PhantomData<Shampine>
}

impl<Shampine, const K: usize, const S: usize> RKInterpolator<f64,S,1> for PartitionedShampineInterpolator<Shampine,K,S,1>
    where Shampine: ShampineConfig<K,S>
{
    type InterpolantType<const D: usize> = ShampineInterpolant<K,S,D,1>;
    fn interpolate_stage<F,const D: usize>(ode: &F, t0: f64, t1: f64, y0: &SVector<f64,D>, y1: &SVector<f64,D>, stage: &[SVector<f64,D>; S]) -> Self::InterpolantType<D>
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        ShampineInterpolant::<K,S,D,1>::new(t0, t1, *y0, *y1,*stage, Shampine::W)
    }
}

pub struct PartitionedShampineInterpolator<Shampine, const K: usize, const S: usize, const P: usize>
    where Shampine: ShampineConfig<K,S>
{
    _marker: PhantomData<Shampine>
}

impl<Shampine, const K: usize, const S: usize, const P: usize> RKInterpolator<SVector<f64,P>,S,P> for PartitionedShampineInterpolator<Shampine,K,S,P>
    where Shampine: ShampineConfig<K,S>
{
    type InterpolantType<const D: usize> = ShampineInterpolant<K,S,D,P>;

    fn interpolate_stage<F,const D: usize>(ode: &F, t0: f64, t1: f64, y0: &SMatrix<f64,D,P>, y1: &SMatrix<f64,D,P>, stage: &[SMatrix<f64,D,P>; S]) -> Self::InterpolantType<D>
        where F: Fn(SVector<f64,P>,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
    {
        ShampineInterpolant::<K,S,D,P>::new(t0, t1, *y0, *y1,*stage, Shampine::W)
    }
}