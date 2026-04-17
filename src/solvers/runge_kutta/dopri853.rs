// Copyright (c) Jesse Kane
// You may use, distribute, and modify this software under the terms of
// the license found in the root directory of this project
/* 
use std::marker::PhantomData;

use nalgebra::SVector;

use crate::solvers::{common::{norm, select_initial_timestep}, dense::DenseInterpolant, runge_kutta::{adaptive_rk::{AdaptiveRKConfig, FirstOrderAdaptiveRKController, ShampineConfig, ShampineRKInterpolator, compute_new_h}, rk_method::{RKController, RKMethod}, rk_solver::RKSolver}};

pub struct DOPRI853 {}

// Controller
pub struct DOPRI853RKController<Method> 
    where Method: RKMethod<13,2>
{
    _marker: PhantomData<Method>
}

impl<Method> DOPRI853RKController<Method>
    where Method: RKMethod<13,2>
{
    const ERROR_EXPONENT: f64 = {-1.0/(Method::ERR_ORDER as f64 + 1.0)};
}


impl<Method> RKController<2> for DOPRI853RKController<Method>
    where Method: RKMethod<13,2>
{

    type Config = AdaptiveRKConfig;

    fn get_next_step<const D: usize>(y1: &SVector<f64, D>, e: &[SVector<f64,D>; 2], y0: &SVector<f64,D>, h: f64, cfg: &AdaptiveRKConfig) -> (bool, f64)
    {
        let scale = (y0.abs().sup(&(y1.abs())) * cfg.rtol).add_scalar(cfg.atol);
        let err5 = e[0].component_div(&scale);
        let err3 = e[1].component_div(&scale);
        let err5_norm_2 = err5.norm().powi(2);
        let err3_norm_2 = err3.norm().powi(2);

        if err5_norm_2 == 0.0 && err3_norm_2 == 0.0 {
            // Error norm = 0
            return (true,compute_new_h(0.0, h, cfg.safety, cfg.max_step, cfg.min_step, cfg.max_factor, cfg.min_factor, Self::ERROR_EXPONENT))
        }
        let denom = err5_norm_2 + 0.01 * err3_norm_2;
        let err_norm = h.abs() * err5_norm_2 / (denom * D as f64).sqrt();

        (err_norm <= 1.0,compute_new_h(err_norm, h, cfg.safety, cfg.max_step, cfg.min_step, cfg.max_factor, cfg.min_factor, Self::ERROR_EXPONENT))
    }
    fn select_initial_timestep<F, const D: usize>(ode: &F, t0: f64, y0: &SVector<f64,D>, f0: &SVector<f64,D>, cfg: &AdaptiveRKConfig) -> f64
        where F: Fn(f64,&SVector<f64,D>) -> SVector<f64,D>
    {
        select_initial_timestep(ode, y0, t0, f0, cfg.atol, cfg.rtol, Method::ERR_ORDER)
    }
}

// Interpolator
pub struct DOPRI853Interpolant<const D: usize> {
    t0: f64,
    t1: f64,
    h: f64,
    y0: SVector<f64, D>,
    r: [SVector<f64,D>; 8]
}

impl<const D: usize> DOPRI853Interpolant<D> {

    fn eval_impl(&self, theta: f64) -> SVector<f64,D> {
        self.r[0] + theta * (
            self.r[1] + (1.0 - theta) * (
                self.r[2] + theta * (
                    self.r[3] + (1.0 - theta) * (
                        self.r[4] + theta * (
                            self.r[5] + (1.0 - theta) * (
                                self.r[6] + theta * self.r[7]
                            )
                        )
                    )
                )
            )
        )
    }

    fn get_theta(&self, t: f64) -> f64 {
        (t-self.t0)/self.h
    }

    pub fn new(t0: f64, t1: f64, y0: SVector<f64,D>, r: [SVector<f64,D>; 8]) -> Self {
        Self {
            t0: t0,
            t1: t1,
            h: t1 - t0,
            y0: y0,
            r: r
        }
    }
}

impl<const D: usize> DenseInterpolant<D> for DOPRI853Interpolant<D> {
    fn eval(&self, t: f64) -> SVector<f64,D> {
        self.eval_impl(self.get_theta(t))
    }
    fn low_t(&self) -> f64 {
        self.t0
    }
    fn high_t(&self) -> f64 {
        self.t1
    }
    fn y0(&self) -> SVector<f64,D> {
        self.y0
    }
}

const A_EXTRA: [[f64; 17]; 4] = []

// Method

impl RKMethod<13,2> for DOPRI853 {

    type Controller = DOPRI853RKController<Self>;
    type Interpolator = ShampineRKInterpolator<Self, 8, 13>;

    const C: [f64; 13] = [0.0, 0.526001519587677318785587544488e-01, 0.789002279381515978178381316732e-01, 0.118350341907227396726757197510, 0.281649658092772603273242802490, 0.333333333333333333333333333333, 0.25, 0.307692307692307692307692307692, 0.651282051282051282051282051282, 0.6, 0.857142857142857142857142857142, 1.0, 1.0];
    const A: [[f64; 13]; 13] = [
        [                  0.0,                 0.0,                   0.0,                   0.0,                   0.0,                    0.0,                  0.0,                   0.0,                    0.0,                    0.0,                   0.0,                  0.0, 0.0],
        [ 5.260015195876773e-2,                 0.0,                   0.0,                   0.0,                   0.0,                    0.0,                  0.0,                   0.0,                    0.0,                    0.0,                   0.0,                  0.0, 0.0],
        [1.9725056984537899e-2, 5.91751709536137e-2,                   0.0,                   0.0,                   0.0,                    0.0,                  0.0,                   0.0,                    0.0,                    0.0,                   0.0,                  0.0, 0.0],
        [ 2.958758547680685e-2,                 0.0,  8.876275643042055e-2,                   0.0,                   0.0,                    0.0,                  0.0,                   0.0,                    0.0,                    0.0,                   0.0,                  0.0, 0.0],
        [ 2.413651341592667e-1,                 0.0, -8.845494793282861e-1,   9.24834003261792e-1,                   0.0,                    0.0,                  0.0,                   0.0,                    0.0,                    0.0,                   0.0,                  0.0, 0.0],
        [ 3.703703703703704e-2,                 0.0,                   0.0, 1.7082860872947387e-1, 1.2546768756682243e-1,                    0.0,                  0.0,                   0.0,                    0.0,                    0.0,                   0.0,                  0.0, 0.0],
        [         3.7109375e-2,                 0.0,                   0.0, 1.7025221101954404e-1,  6.021653898045596e-2,          -1.7578125e-2,                  0.0,                   0.0,                    0.0,                    0.0,                   0.0,                  0.0, 0.0],
        [3.7092000118504793e-2,                 0.0,                   0.0, 1.7038392571223999e-1, 1.0726203044637328e-1, -1.5319437748624402e-2, 8.273789163814023e-3,                   0.0,                    0.0,                    0.0,                   0.0,                  0.0, 0.0],
        [ 6.241109587160757e-1,                 0.0,                   0.0,   -3.3608926294469413,  -8.68219346841726e-1,   2.7592099699446707e1, 2.0154067550477893e1,  -4.348988418106996e1,                    0.0,                    0.0,                   0.0,                  0.0, 0.0],
        [ 4.776625364382644e-1,                 0.0,                   0.0,   -2.4881146199716676,  -5.90290826836843e-1,   2.1230051448181194e1, 1.5279233632882424e1, -3.3288210968984863e1, -2.0331201708508626e-2,                    0.0,                   0.0,                  0.0, 0.0],
        [-9.371424300859873e-1,                 0.0,                   0.0,     5.186372428844064,    1.0914373489967296,     -8.149787010746926, -1.852006565999696e1,  2.2739487099350504e1,     2.4936055526796524,    -3.0467644718982195,                   0.0,                  0.0, 0.0],
        [    2.273310147516538,                 0.0,                   0.0,  -1.053449546673725e1,   -2.0008720582248625,  -1.7958931863118799e1,  2.794888452941996e1,   -2.8589982771350237,      -8.87285693353063,   1.2360567175794303e1,  6.433927460157635e-1,                  0.0, 0.0],
        [ 5.429373411656876e-2,                 0.0,                   0.0,                   0.0,                   0.0,      4.450312892752409,   1.8915178993145004,    -5.801203960010585,   3.111643669578199e-1, -1.5216094966251608e-1, 2.0136540080403035e-1, 4.471061572777259e-2, 0.0]
    ];
    
    const B: [f64; 13] = [5.429373411656876e-2, 0.0, 0.0, 0.0, 0.0, 4.450312892752409, 1.8915178993145004, -5.801203960010585, 3.111643669578199e-1, -1.5216094966251608e-1, 2.0136540080403035e-1, 4.471061572777259e-2, 0.0];
    const FSAL: bool = true;
    
    const ORDER: usize = 8;
    const ERR_ORDER: usize = 7;
}

impl ShampineConfig<5,7> for DOPRI853 {
    const P: [[f64; 5]; 7] = [
        [0.0, 1.0,   -8048581381.0/2820520608.0,     8663915743.0/2820520608.0,  -12715105075.0/11282082432.0],
        [0.0, 0.0,                          0.0,                           0.0,                           0.0],
        [0.0, 0.0, 131558114200.0/32700410799.0,  -68118460800.0/10900136933.0,   87487479700.0/32700410799.0],
        [0.0, 0.0,    -1754552775.0/470086768.0,    14199869525.0/1410260304.0,   -10690763975.0/1880347072.0],
        [0.0, 0.0, 127303824393.0/49829197408.0, -318862633887.0/49829197408.0, 701980252875.0/199316789632.0],
        [0.0, 0.0,     -282668133.0/205662961.0,      2019193451.0/616988883.0,     -1453857185.0/822651844.0],
        [0.0, 0.0,        40617522.0/29380423.0,       -110615467.0/29380423.0,         69997945.0/29380423.0]
    ];
}

pub type DOPRI853Solver<const D: usize> = RKSolver<DOPRI853 ,13, 3, D>;

*/