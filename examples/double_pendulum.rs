// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use odesolver::solvers::solver::{DenseSolver, Solver};
use odesolver::solvers::runge_kutta::dopri5::DOPRI5Solver;

use nalgebra::SVector;

use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

const G: f64 = 9.81; // gravity, in m/s/s
const L1: f64 = 1.0; // Length of the first pendulum, in meters
const L2: f64 = 1.0; // Length of the second pendulum, in meters
const M1: f64 = 1.0; // Mass of the first pendulum bob, in kilograms
const M2: f64 = 1.0; // Mass of the second pendulum bob, in kilograms

const FPS: usize = 30;
const SECONDS: usize = 15;
const TDELTA: f64 = 1.0 / FPS as f64;

// Converts a state into a pair of cartesian coordinates, in the form (bob1,bob2). The origin is the origin of the pendulum, in cartesian coordinates
fn to_cartesian(state: &SVector<f64,4>, origin: &SVector<f64,2>) -> (SVector<f64,2>,SVector<f64,2>) {
    // The trig functions are reversed here because theta is measured from the vertical axis
    let mut bob1 = SVector::<f64, 2>::new(state[0].sin() * L1, -(state[0].cos()) * L1);
    let mut bob2 = bob1 + SVector::<f64, 2>::new(state[1].sin() * L2, -(state[1].cos()) * L2);
    bob1 += origin;
    bob2 += origin;
    (bob1,bob2)
}

fn cartesian_speed(state: &SVector<f64,4>) -> f64 {
    let theta_1 = state[0];
    let theta_2 = state[1];
    let omega_1 = state[2];
    let omega_2 = state[3];

    // bob1 velocity
    let v1 = SVector::<f64, 2>::new(
        L1 * theta_1.cos() * omega_1,
        L1 * theta_1.sin() * omega_1
    );

    // bob2 velocity = v1 + relative contribution
    let v2 = v1 + SVector::<f64, 2>::new(
        L2 * theta_2.cos() * omega_2,
        L2 * theta_2.sin() * omega_2
    );

    v2.norm()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Degrees are in radians, velocity is in rad/s, acceleration is in rad/s/s
    // First element is theta_1
    // Second element is theta_2
    // Third element is omega_1
    // Fourth element is omega_2
    let double_pendulum_ode = |_time: f64, point: &SVector<f64, 4>| {
        let theta_1 = point[0];
        let theta_2 = point[1];
        let omega_1 = point[2];
        let omega_2 = point[3];

        let delta = theta_1 - theta_2;

        let omega_1_sqr = omega_1.powi(2);
        let omega_2_sqr = omega_2.powi(2);

        let delta_sin = delta.sin();
        let delta_cos = delta.cos();

        let cos_2delta = 2.0 * delta_cos * delta_cos - 1.0;
        let q = 2.0 * M1 + M2 - M2 * cos_2delta;

        let omega_1_dot = 
            (-G * (2.0 * M1 + M2) * theta_1.sin() 
            - M2 * G * (theta_1 - 2.0 * theta_2).sin()
            - 2.0 * M2 * delta_sin * (omega_2_sqr * L2 + omega_1_sqr * L1 * delta_cos))
            / (L1 * q);
        let omega_2_dot =
            (2.0 * delta_sin * 
                (omega_1_sqr * L1 * (M1 + M2) + 
                G * (M1 + M2) * theta_1.cos() + 
                omega_2_sqr * L2 * M2 * delta_cos))
        / (L2 * q);
        SVector::<f64,4>::new(
            omega_1,
            omega_2,
            omega_1_dot,
            omega_2_dot
        )
    };

    let mut solver = DOPRI5Solver::<4>::default();

    solver.cfg.rtol = 1e-8;
    solver.cfg.atol = 1e-10;

    let point = SVector::<f64,4>::new(PI/2.0,PI/2.0,0.0,0.0);
    let result = solver.solve_dense(&double_pendulum_ode, &point, 0.0, SECONDS as f64);

    let interpolator = result.1;
    let mut interp_points = Vec::<(f64,SVector<f64,4>)>::new();
    let mut t = 0.0;
    while t <= SECONDS as f64 {
        interp_points.push((t,interpolator.eval(t)));
        t += TDELTA;
    }
    // This is a state-space representation of the pendulum, which decomposes the second-order ODE describing the dynamics of the system into a pair of first-order differential equations
    // One axis is the position of the pendulum (measured as an angle), the other axis is the velocity of the pendulum.
    // The dynamics of the system are represented as a vector field over the state space, which maps each point of the state space to a vector describing its derivative

    
    let file = File::create("double_pendulum.csv")?;
    let mut w = BufWriter::new(file);

    writeln!(w, "t,x1,y1,x2,y2,speed")?;

    let origin = SVector::<f64,2>::new(0.0,0.0);

    for (t, y) in interp_points {
        let cartesian = to_cartesian(&y, &origin);
        writeln!(w, "{},{},{},{},{},{}", t, cartesian.0[0], cartesian.0[1], cartesian.1[0], cartesian.1[1],cartesian_speed(&y))?;
    }

    

    Ok(())
}
