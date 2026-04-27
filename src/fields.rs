// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use nalgebra::ComplexField;

pub type Real<T> = <T as ComplexField>::RealField;

pub fn coerce_f64<T>(x: f64) -> T
    where T: ComplexField
{
    return T::from_f64(x).unwrap();
}