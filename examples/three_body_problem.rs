// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use std::{f64::consts::PI, fs::File};
use std::io::{BufWriter, Write};

use nalgebra::SVector;
use odesolver::solvers::{runge_kutta::{dopri5::DOPRI5Solver, dopri853::{DOPRI853, DOPRI853Solver}}, solver::DenseSolver};

const G: f64 = 4.0 * PI * PI; // units are in solar masses, years, and astronomical units
const M1: f64 = 1.0;
const M2: f64 = 1.0;
const M3: f64 = 1.0;

const FPY: usize = 30;
const YEARS: usize = 50;
const TDELTA: f64 = 1.0 / FPY as f64;


fn main() -> Result<(), Box<dyn std::error::Error>> {

    // lwk need to add support for types other than f64, an 18-dimensional system is ridiculous
    let three_body_problem = |_time: f64, point: &SVector<f64, 18>| {
        let r1 = SVector::<f64,3>::new(point[0],point[1], point[2]);
        let r2 = SVector::<f64,3>::new(point[3],point[4], point[5]);
        let r3 = SVector::<f64,3>::new(point[6],point[7], point[8]);
        let r1_dot = SVector::<f64,3>::new(point[9],point[10], point[11]);
        let r2_dot = SVector::<f64,3>::new(point[12],point[13], point[14]);
        let r3_dot = SVector::<f64,3>::new(point[15],point[16], point[17]);

        // We clamp the denominators to prevent singularities, otherwise we divide by zero and the solver breaks. Pretend there's a magic repulsor field that prevents the objects from colliding.
        let r1_r2 = r1 - r2;
        let r1_r3 = r1 - r3;
        let r1_dot_dot = -G * M2 * r1_r2/(r1_r2.norm().max(1e-8).powi(3)) - G * M3 * r1_r3/(r1_r3.norm().max(1e-8).powi(3));

        let r2_r3 = r2 - r3;
        let r2_r1 = r2 - r1;
        let r2_dot_dot = -G * M3 * r2_r3/(r2_r3.norm().max(1e-8).powi(3)) - G * M1 * r2_r1/(r2_r1.norm().max(1e-8).powi(3));

        let r3_r1 = r3 - r1;
        let r3_r2 = r3 - r2;
        let r3_dot_dot = -G * M1 * r3_r1/(r3_r1.norm().max(1e-8).powi(3)) - G * M2 * r3_r2/(r3_r2.norm().max(1e-8).powi(3));

        SVector::<f64,18>::from_row_slice(&[
            r1_dot[0],
            r1_dot[1],
            r1_dot[2],

            r2_dot[0],
            r2_dot[1],
            r2_dot[2],

            r3_dot[0],
            r3_dot[1],
            r3_dot[2],

            r1_dot_dot[0],
            r1_dot_dot[1],
            r1_dot_dot[2],

            r2_dot_dot[0],
            r2_dot_dot[1],
            r2_dot_dot[2],

            r3_dot_dot[0],
            r3_dot_dot[1],
            r3_dot_dot[2]
        ])
    };

    let mut solver = DOPRI853Solver::<18>::default();

    let y0 = SVector::<f64,18>::from_row_slice(&[
        0.577350269,  0.0,         0.0,
        -0.288675135,  0.5,         0.0,
        -0.288675135, -0.5,         0.0,

        0.0,         6.283185307,  0.0,
        -5.441398093,-3.141592654,  0.0,
        5.441398093,-3.141592654,  0.0,
    ]);
    println!("{:?}",three_body_problem(0.0,&y0));

    let result = solver.solve_dense(&three_body_problem, &y0, 0.0, YEARS as f64, true);

    let file = File::create("three_body_problem.csv")?;
    let mut w = BufWriter::new(file);

    writeln!(w, "t,x1,y1,z1,x2,y2,z2,x3,y3,z3,x1_dot,y1_dot,z1_dot,x2_dot,y2_dot,z2_dot,x3_dot,y3_dot,z3_dot")?;

    let interpolator = result.1;

    let mut interp_points = Vec::<(f64,SVector<f64,18>)>::new();
    let mut t = 0.0;
    while t <= YEARS as f64 {
        interp_points.push((t,interpolator.eval(t)));
        t += TDELTA;
    }
    for (t, y) in interp_points {
        writeln!(w, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}", t, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17])?;
    }


    Ok(())
}
