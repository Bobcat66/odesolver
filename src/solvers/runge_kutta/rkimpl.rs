// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::SVector;

use crate::solvers::{dense::{DenseInterpolant, DenseOutput}, runge_kutta::{butcher::{ButchersTableau, ExtendedButchersTableau}, rk_dense::RKInterpolant}};

pub fn rk_stage_impl<Tableau, const S: usize, const D: usize, F>(
    ode: &F,
    t0: f64,
    y0: &SVector<f64, D>,
    f_t0: &SVector<f64, D>,
    h: f64,
    k: &mut [SVector<f64, D>; S]
) -> ()
    where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>,
    Tableau: ButchersTableau<S>
{
    k[0] = *f_t0;
    for i in 1..S {
        let mut dy = SVector::<f64, D>::zeros();
        for j in 0..i {
            dy += k[j] * Tableau::A[i][j];
        }
        dy *= h;
        k[i] = ode(t0 + h * Tableau::C[i],&(y0 + dy));
    }
}

// Adaptive RK weights
pub fn ark_weight_impl<Tableau, const S: usize, const P: usize, const D: usize>(
    y0: &SVector<f64, D>,
    h: f64,
    k: &[SVector<f64,D>;S]
) -> (SVector<f64,D>,SVector<f64,D>)
    where Tableau: ExtendedButchersTableau<S,P>
{
    let mut y1: SVector<f64, D> = SVector::zeros();
    let mut err: SVector<f64, D> = SVector::zeros();
    for i in 0..S {
        y1 += k[i] * Tableau::B[i];
        err += k[i] * (Tableau::B[i] - Tableau::B_LOW[i]);
    }
    y1 *= h;
    y1 += y0;
    err *= h;
    (y1,err)
}

// Adaptive RK weights
pub fn ark_step_impl<Tableau, const S: usize, const P: usize, const D: usize, F>(
    ode: &F,
    t0: f64,
    y0: &SVector<f64, D>,
    f_t0: &SVector<f64, D>,
    h: f64,
    k: &mut [SVector<f64, D>; S]
) -> (SVector<f64, D>,SVector<f64, D>) 
    where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>,
    Tableau: ExtendedButchersTableau<S,P>
{
    rk_stage_impl::<Tableau,S,D,_>(ode, t0, y0, f_t0, h, k);
    ark_weight_impl::<Tableau,S,P,D>(y0, h, k)
}

// Normalizes vec using RMS, scaled to y
pub fn scale_norm<const D: usize>(vec: &SVector<f64,D>, y: &SVector<f64,D>, atol: f64, rtol: f64) -> f64
{
    let mut vec_norm = 0.0;
    for i in 0..D {
        let scale = atol + rtol * y[i].abs();
        vec_norm += (vec[i]/scale).powi(2);
    }
    (vec_norm / D as f64).sqrt()
}

// Returns initial guess for h
pub fn guess_timestep<F, const D: usize>(ode: &F, y0: &SVector<f64,D>, t0: f64, atol: f64, rtol: f64) -> f64
    where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
{
    let d0 = scale_norm(y0, y0, atol, rtol);
    let ydot_0 = ode(t0,y0);
    let d1 = scale_norm(&ydot_0,y0, atol, rtol);

    // Create initial guess
    let h0: f64 =  if d0 < 1e-5 || d1 < 1e-5 {1e-6} else {0.01 * d0 / d1};

    // Take euler step to estimate curvature
    let y1 = y0 + h0 * ydot_0;
    let ydot_1 = ode(t0+h0,&y1);
    let d2 = scale_norm(&(ydot_1 - ydot_0), y0, atol, rtol);
    
    (h0 * 100.0).min((0.01/d1.max(d2)).powf(0.2))
}

// Adaptive RK Solve
pub fn ark_solve_impl<Tableau, const S: usize, const P: usize, const D: usize, F>(
    ode: &F,
    y0: &SVector<f64,D>,
    t_start: f64,
    t_end: f64,
    atol: f64,
    rtol: f64,
    min_clamp: f64,
    max_clamp: f64,
    safety: f64,
    k: &mut [SVector<f64, D>; S]
) -> Vec<(f64,SVector<f64,D>)> 
    where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>,
    Tableau: ExtendedButchersTableau<S,P>
{
    let mut t = t_start;
    let mut y = *y0;
    let mut h = guess_timestep(ode, y0, t_start, atol, rtol);
    let mut points: Vec<(f64,SVector<f64,D>)> = Vec::new();
    points.push((t,y));
    let mut f = ode(t_start, y0);
    while t < t_end {
        let res = ark_step_impl::<Tableau,S,P,D,_>(ode, t, &y, &f, h, k);

        // compute error
        let err_norm = scale_norm(&res.1,&y, atol, rtol);

        // recalculate stepsize
        let new_h = (h * safety * err_norm.powf(-0.2)).clamp(min_clamp,max_clamp);

        // Accept or reject step
        if err_norm <= 1.0 {
            y = res.0;
            t += h;
            f = if Tableau::FSAL {k[S - 1]} else {ode(t,&y)};
            points.push((t,y));
        }
        h = new_h;
    }
    points
}

// Adaptive RK dense solve
pub fn ark_solve_dense_impl<Tableau, const S: usize, const P: usize, const D: usize, F>(
    ode: &F,
    y0: &SVector<f64,D>,
    t_start: f64,
    t_end: f64,
    atol: f64,
    rtol: f64,
    min_clamp: f64,
    max_clamp: f64,
    safety: f64,
    k: &mut [SVector<f64, D>; S]
) -> (Vec<(f64,SVector<f64,D>)>,DenseOutput<D>)
    where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>,
    Tableau: ExtendedButchersTableau<S,P>
{
    let mut t = t_start;
    let mut y = *y0;
    let mut h = guess_timestep(ode, y0, t_start, atol, rtol);
    let mut points: Vec<(f64,SVector<f64,D>)> = Vec::new();
    let mut stages: Vec<[SVector<f64,D>; S]> = Vec::new();
    let mut steps: usize = 0;
    points.push((t,y));
    let mut f = ode(t_start, y0);
    while t < t_end {
        let res = ark_step_impl::<Tableau,S,P,D,_>(ode, t, &y, &f, h, k);

        // compute error
        let err_norm = scale_norm(&res.1,&y, atol, rtol);

        // recalculate stepsize
        let new_h = (h * safety * err_norm.powf(-0.2)).clamp(min_clamp,max_clamp);

        // Accept or reject step
        if err_norm <= 1.0 {
            y = res.0;
            t += h;
            f = if Tableau::FSAL {k[S - 1]} else {ode(t,&y)};
            points.push((t,y));
            stages.push((*k).clone());
            steps += 1;
        }
        h = new_h;
    }
    stages.push((*k).clone());
    let mut segments: Vec<(f64,Box<dyn DenseInterpolant<D>>)> = Vec::new();
    for i in 0..(steps-1) {
        segments.push(
            (
                points[i].0,
                Box::new(RKInterpolant::new(points[i].0,points[i + 1].0, points[i].1, stages[i], Tableau::P))
            )
        );
    }

    (points,DenseOutput::new(segments))
}
