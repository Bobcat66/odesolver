// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::marker::PhantomData;

use nalgebra::{SMatrix, SVector};

use crate::ode::ODE;

pub struct FnODE<F,T,const D: usize,const P: usize> 
    where F: Fn(T,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
{
    f: F,
    _marker: PhantomData<T>
}

impl<F,T,const D: usize, const P: usize> FnODE<F,T,D,P>
    where F: Fn(T,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
{
    pub fn new(f: F) -> Self
    {
        Self {f: f, _marker: PhantomData}
    }
}

impl<F,T,const D: usize, const P: usize> ODE<T,D,P> for FnODE<F,T,D,P>
    where F: Fn(T,&SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
{
    fn eval(&self, t: T, y: &SMatrix<f64,D,P>) -> SMatrix<f64,D,P>
    {
        self.f(t,y)
    }
}

pub type SimpleODE<F,const D: usize> = FnODE<F,f64,D,1>;
pub type PartitionedODE<F,const D: usize, const P: usize> = FnODE<F,SVector<f64,P>,D,P>;
pub type SymplecticODE<F,const D: usize> = PartitionedODE<F,D,2>;