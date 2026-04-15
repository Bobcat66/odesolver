// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

pub trait Mapping<X,Y> {
    fn eval(&self, x: X) -> Y;
}