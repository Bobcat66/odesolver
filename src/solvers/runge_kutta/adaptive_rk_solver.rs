use std::marker::PhantomData;

use crate::solvers::dense::DenseOutput;
use crate::solvers::solver::Solver;
use crate::solvers::runge_kutta::rkimpl::ark_solve_impl;
use nalgebra::SVector;

use crate::solvers::runge_kutta::butcher::*;

// S is the number of stages, D is the dims
pub struct AdaptiveRKSolver<Tableau, const S: usize, const P: usize, const D: usize> 
    where Tableau: ExtendedButchersTableau<S,P>
{
    pub atol: f64, // absolute tolerance
    pub rtol: f64, // normalized tolerance
    pub safety: f64, // safety value to reduce overshoot
    pub min_clamp: f64, // minimum timestep clamp
    pub max_clamp: f64, // maximum timestep clamp
    k: [SVector<f64,D>; S],
    _marker: PhantomData<Tableau>
}

impl<Tableau, const S: usize, const P: usize, const D: usize> Default for AdaptiveRKSolver<Tableau,S,P,D> 
    where Tableau: ExtendedButchersTableau<S,P> 
{
    fn default() -> Self {
        Self {
            atol: 1e-3,
            rtol: 1e-6,
            safety: 0.9,
            min_clamp: 1e-12,
            max_clamp: f64::MAX,
            k: [SVector::<f64,D>::zeros(); S],
            _marker: PhantomData
        }
    }
}

impl<Tableau, const S: usize, const P: usize, const D: usize> Solver<D> for AdaptiveRKSolver<Tableau,S,P,D>
    where Tableau: ExtendedButchersTableau<S,P> 
{
    fn solve<F>(&mut self, ode: &F, y0: &SVector<f64,D>, t_start: f64, t_end: f64) -> Vec<(f64,SVector<f64,D>)> 
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        ark_solve_impl::<Tableau,S,P,D,_>(
            ode,
            y0,
            t_start,
            t_end,
            self.atol,
            self.rtol,
            self.min_clamp,
            self.max_clamp, 
            self.safety,
            &mut self.k
        )
    }

    fn solve_dense<F>(&mut self, ode: &F, y0: &SVector<f64,D>, t_start: f64, t_end: f64) -> (Vec<(f64,SVector<f64,D>)>,DenseOutput<D>)
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        
    }
}