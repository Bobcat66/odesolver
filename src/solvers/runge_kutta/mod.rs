use crate::solvers::runge_kutta::{dopri3::DOPRI3, dopri5::DOPRI5, dopri853::DOPRI853,prk_solver::PRKSolver};

// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
pub mod dopri5;
pub mod rk_stepper;
pub mod adaptive_prk;
pub mod rk_solver;
pub mod dopri853;
pub mod rk_method;
pub mod dopri3;
pub mod fixed_prk;
pub mod prk_method;
pub mod prk_stepper;
pub mod prk_solver;

pub type DOPRI3Solver<const D: usize> = PRKSolver<DOPRI3, 1,4, 1, D>;
pub type DOPRI5Solver<const D: usize> = PRKSolver<DOPRI5, 1,7, 1, D>;
pub type DOPRI853Solver<const D: usize> = PRKSolver<DOPRI853 ,1,13, 2, D>;