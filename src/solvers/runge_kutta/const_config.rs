// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

// S is number of stages, O is the number of solutions. B[0] is always the main solution, while B[1..n] are embedded solutions
pub trait ButchersTableau<const S: usize, const O: usize> {
    const C: [f64; S]; // time coefficients
    const A: [[f64; S]; S]; // stage coefficients
    const B: [[f64; S]; O]; // weights
    const ORDERS: [usize; O]; // order of solutions
    const FSAL: bool;
}
// S is number of stages, P is the number of coefficients of the shampine polynomials (which is 1 greater than its order, as the polynomial solver requires zeroth order coeffs). This tableau is extended to support embedded solutions and dense output
pub trait ShampineConfig<const P: usize, const S: usize> {
    const P: [[f64; P]; S]; // shampine polynomial weights
}