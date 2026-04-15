// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use odesolver::algebra::polynomial::Polynomial;
use odesolver::algebra::mapping::Mapping;
// Polynomial test
fn main() {
    let polynomial: Polynomial<f64,3> = Polynomial {
        a: [1.0, 2.0, 3.0]
    };
    println!("{}",polynomial.eval(3.5));
}