mod vector_field;
use core::time;
use std::f64::consts::PI;

use nalgebra::SVector;

fn main() {
    // A simple example of the ODE solver that models a pendulum
    const MU: f64 = 0.0; // This 
    const G: f64 = 9.81; // gravity, in m/s/s
    const L: f64 = 1.0; // Length of the pendulum, in meters

    // First element is theta
    // Second element is theta dot
    let pendulum_state: vector_field::VectorField<2> = vector_field::VectorField::new(
        [
            |point: &SVector<f64, 2>, time: f64| point[1],
            |point: &SVector<f64, 2>, time: f64| (-MU * point[1]) - ((G/L) * point[0].sin())
        ]
    );

    let point = SVector::<f64,2>::new(PI/2.0,0.0);
    let point_dot = pendulum_state.eval(point,0.0);
    println!("{:?}", point_dot);
    // This is a state-space representation of the pendulum, which decomposes the second-order ODE describing the dynamics of the system into a pair of first-order differential equations
    // One axis is the position of the pendulum (measured as an angle), the other axis is the velocity of the pendulum.
    // The dynamics of the system are represented as a vector field over the state space, which maps each point of the state space to a vector describing its derivative

    
    println!("Hello, world!");
}
