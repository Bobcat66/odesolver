// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

// S is number of stages
pub trait ButchersTableau<const S: usize> {
    const C: [f64; S]; // time coefficients
    const A: [[f64; S]; S]; // stage coefficients
    const B: [f64; S]; // weights
    const FSAL: bool;
    const ORDER: usize;
}
// S is number of stages, P is the degree of the shampine polynomials. This tableau is extended to support embedded solutions and dense output
pub trait ExtendedButchersTableau<const S: usize, const P: usize> : ButchersTableau<S> {
    const B_LOW: [f64; S]; // weights for low-order
    const P: [[f64; P]; S]; // shampine polynomial weights
    const EMBEDDED_ORDER: usize;
}