use crate::solvers::runge_kutta::{dopri3::DOPRI3, dopri5::DOPRI5, dopri853::DOPRI853, rk_solver::RKSolver};

// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
pub mod dopri5;
pub mod rk_stepper;
pub mod adaptive_rk;
pub mod rk_solver;
pub mod dopri853;
pub mod rk_method;
pub mod dopri3;
pub mod fixed_rk;

pub type DOPRI3Solver<const D: usize> = RKSolver<DOPRI3, 4, 1, D>;
pub type DOPRI5Solver<const D: usize> = RKSolver<DOPRI5, 7, 1, D>;
pub type DOPRI853Solver<const D: usize> = RKSolver<DOPRI853 ,13, 2, D>;