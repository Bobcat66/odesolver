// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
mod rkimpl;
pub mod utils;
pub mod dopri5;

pub struct ButchersTableau<const S: usize,const E: usize> {
     a: [[f64; S]; S], // stage coefficients
    pub b: [[f64; S]; E], // solution coefficients
    pub c: [f64; S] // timestamps
}