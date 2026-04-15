// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

use crate::algebra::mapping::Mapping;

// D is the number of coefficients of the polynomial.
pub struct Polynomial<T, const D: usize>
{
    pub a: [T; D],
}

impl<T, const D: usize> Mapping<T,T> for Polynomial<T,D>
    where T: Clone
    + std::ops::Add<T, Output=T>
    + std::ops::Mul<T, Output=T>
{
    fn eval(&self,x: T) -> T
    {
        let mut y = self.a[D - 1].clone();

        for i in (0..D - 1).rev() {
            y = self.a[i].clone() + y * x.clone();
        }
        y
    }
}
