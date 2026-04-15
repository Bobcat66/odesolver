// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

// S is number of stages, P is the degree of the shampine polynomials.
pub trait ButchersTableau<const S: usize> {
    fn c() -> &'static [f64; S]; // time coefficients
    fn a() -> &'static [[f64; S]; S]; // stage coefficients
    fn b() -> &'static [f64; S]; // weights
}
// S is number of stages, P is the degree of the shampine polynomials. This tableau is extended to support embedded solutions and dense output
pub trait ExtendedButchersTableau<const S: usize, const P: usize> : ButchersTableau<S> {
    fn b_low() -> &'static [f64; S]; // weights for low-order
    fn p() -> &'static [[f64; P]; S]; // shampine polynomial weights
}