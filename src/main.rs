// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

mod solvers;
use std::f64::consts::PI;
use solvers::solver::Solver;
use solvers::dormand_prince::DormandPrinceSolver;

use nalgebra::SVector;

fn main() {
    // A simple example of the ODE solver that models a pendulum
    const MU: f64 = 0.5; // This term represents energy loss to friction, heat, air resistance, etc.
    const G: f64 = 9.81; // gravity, in m/s/s
    const L: f64 = 1.0; // Length of the pendulum, in meters

    // First element is theta
    // Second element is omega
    let pendulum_ode = |point: &SVector<f64, 2>, _time: f64| {
        SVector::<f64,2>::new(
            point[1],
            (-MU * point[1]) - ((G/L) * point[0].sin())
        )
    };

    let solver = DormandPrinceSolver::<2>::default();

    let point = SVector::<f64,2>::new(PI/2.0,1.0);
    let point_dot = pendulum_ode(&point,0.0);
    println!("{:?}", point_dot);
    let point_t10 = solver.solve(&pendulum_ode, &point, 0.0, 10.0);
    println!("{:?}", point);
    println!("{:?}", point_t10);
    let points = solver.solve_dense(&pendulum_ode, &point, 0.0, 10.0);
    for (point, time) in points {
        println!("t: {}, point: {:?}", time, point);
    }
    // This is a state-space representation of the pendulum, which decomposes the second-order ODE describing the dynamics of the system into a pair of first-order differential equations
    // One axis is the position of the pendulum (measured as an angle), the other axis is the velocity of the pendulum.
    // The dynamics of the system are represented as a vector field over the state space, which maps each point of the state space to a vector describing its derivative

    
    println!("Hello, world!");
}
