// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use odesolver::solvers::solver::Solver;
use odesolver::solvers::runge_kutta::dopri5::DOPRI5Solver;

use nalgebra::SVector;

use std::fs::File;
use std::io::{BufWriter, Write};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    

    // A simple example of the ODE solver that models a pendulum
    const MU: f64 = 0.2; // This term represents energy loss to friction, heat, air resistance, etc.
    const G: f64 = 9.81; // gravity, in m/s/s
    const L: f64 = 1.0; // Length of the pendulum, in meters

    // First element is theta
    // Second element is omega
    let pendulum_ode = |_time: f64, point: &SVector<f64, 2>| {
        SVector::<f64,2>::new(
            point[1],
            (-MU * point[1]) - ((G/L) * point[0].sin())
        )
    };

    let mut solver = DOPRI5Solver::<2>::default();

    let point = SVector::<f64,2>::new(0.0,7.0);
    let points = solver.solve(&pendulum_ode, &point, 0.0, 30.0);
    for (time, point) in &points {
        println!("t: {}, point: {:?}", time, point);
    }
    // This is a state-space representation of the pendulum, which decomposes the second-order ODE describing the dynamics of the system into a pair of first-order differential equations
    // One axis is the position of the pendulum (measured as an angle), the other axis is the velocity of the pendulum.
    // The dynamics of the system are represented as a vector field over the state space, which maps each point of the state space to a vector describing its derivative

    
    let file = File::create("pendulum.csv")?;
    let mut w = BufWriter::new(file);

    writeln!(w, "t,x,y")?;

    for (t, y) in points {
        writeln!(w, "{},{},{}", t, y[0], y[1])?;
    }

    

    Ok(())
}
