// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project

// This file contains the actual implementations of RK solving algorithms, all business logic lives here. Everything else is just scaffolding

pub mod solve;
pub mod explicit;
pub mod partitioned;
pub mod adaptive;
pub mod shampine;
pub mod common;
pub mod fixed;
pub mod dopri853impl;
pub mod dopri853constants;