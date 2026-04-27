// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use crate::solvers::runge_kutta::{adaptive_prk::{FirstOrderAdaptiveRKController, ShampineConfig, ShampineRKInterpolator}, prk_method::PRKMethod};

pub struct DOPRI3 {}

impl PRKMethod<1,4,1> for DOPRI3
{
    type Controller = FirstOrderAdaptiveRKController<Self,1,4>;
    type Interpolator = ShampineRKInterpolator<Self,1, 4, 4>;

    const C: [f64; 4] = [0.0, 1.0/2.0, 3.0/4.0, 1.0];
    const A: [[[f64; 4]; 4]; 1] = [[
        [    0.0,     0.0,     0.0, 0.0],
        [1.0/2.0,     0.0,     0.0, 0.0],
        [    0.0, 3.0/4.0,     0.0, 0.0],
        [2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0]
    ]];
    const B: [[f64; 4]; 1] = [[2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0]];
    const E_B: [[[f64; 4]; 1]; 1] = [[[5.0/72.0, -1.0/12.0, -1.0/9.0, 1.0/8.0]]];
    const FSAL: bool = true;
    const ORDER: usize = 3;
    const ERR_ORDER: usize = 2;
}

impl ShampineConfig<4,4> for DOPRI3 {
    const W: [[f64; 4]; 4] = [
        [0.0, 1.0, -4.0/3.0,  5.0/9.0],
        [0.0, 0.0,      1.0, -2.0/3.0],
        [0.0, 0.0,  4.0/3.0, -8.0/9.0],
        [0.0, 0.0,     -1.0,      1.0]
    ];
}