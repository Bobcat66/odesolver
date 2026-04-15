// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SVector;

use crate::solvers::{common::SolverState, dense::{DenseInterpolant, DenseOutput}, runge_kutta::{const_config::{ButchersTableau, ExtendedButchersTableau}, rk_dense::RKInterpolant}, solver::DenseSolver};

pub fn rk_stage_impl<Tableau, const S: usize, const D: usize, F>(
    ode: &F,
    state: &SolverState<D>,
    k: &mut [SVector<f64, D>; S]
) -> ()
    where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>,
    Tableau: ButchersTableau<S>
{
    k[0] = state.point.f;
    for i in 1..S {
        let mut dy = SVector::<f64, D>::zeros();
        for j in 0..i {
            dy += k[j] * Tableau::A[i][j];
        }
        dy *= state.h;
        k[i] = ode(state.point.t + state.h * Tableau::C[i],&(state.point.y + dy));
    }
}

pub fn rk_weight_impl<Tableau, const S: usize, const D: usize>(
    state: &SolverState<D>,
    k: &[SVector<f64,D>;S]
) -> SVector<f64,D>
    where Tableau: ButchersTableau<S>
{
    let mut y1: SVector<f64, D> = SVector::zeros();
    for i in 0..S {
        y1 += k[i] * Tableau::B[i];
    }
    y1 *= state.h;
    y1 += state.point.y;
    y1
}

pub fn ark_err_impl<Tableau, const S: usize, const P: usize, const D: usize>(
    state: &SolverState<D>,
    k: &[SVector<f64,D>;S]
) -> SVector<f64,D>
    where Tableau: ExtendedButchersTableau<S,P>
{
    let mut err: SVector<f64, D> = SVector::zeros();
    for i in 0..S {
        err += k[i] * (Tableau::B[i] - Tableau::B_LOW[i]);
    }
    err *= state.h;
    err
}