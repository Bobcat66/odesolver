// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use odesolver::solvers::solver::DenseSolver;
use odesolver::solvers::runge_kutta::dopri5::DOPRI5Solver;
use nalgebra::SVector;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const SIGMA: f64 = 10.0;
    const RHO: f64 = 28.0;
    const BETA: f64 = 8.0/3.0;
    // point[0] = x, point[1] = y, point[2] = z
    let lorenz_system = |_time: f64, point: &SVector<f64,3>| {
        SVector::<f64,3>::new(
            SIGMA * (point[1] - point[0]),
            point[0] * (RHO - point[2]) - point[1],
            point[0] * point[1] - BETA * point[2]
        )
    };

    let mut solver = DOPRI5Solver::<3>::default();
    solver.cfg.rtol = 1e-8;
    solver.cfg.atol = 1e-10;
    let point = SVector::<f64,3>::new(1.0,1.0,1.0);
    let res = solver.solve_dense(&lorenz_system,&point,0.0,100.0);
    for (time, point) in &res.0 {
        println!("t: {}, point: {:?}", time, point);
    }

    let file = File::create("lorenz.csv")?;
    let mut w = BufWriter::new(file);

    writeln!(w, "t,x,y,z")?;

    for (t, y) in &res.0 {
        writeln!(w, "{},{},{},{}", t, y[0], y[1], y[2])?;
    }

    println!("{}",res.1.eval(10.0));
    Ok(())
}