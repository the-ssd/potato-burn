use std::f32::consts::FRAC_PI_2;

use burn::prelude::*;
pub mod attention;
pub mod bliniar;
pub mod expand_reduce;
pub mod expander;
pub mod gate;
pub mod reduction;
pub mod transformer_layer;

fn hamming_distance<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>) -> Tensor<B, D> {
    not(xor(a, b)).mean_dim(D - 1)
}

// Keeps value within -1 and 1. While 1 is 1 and -1 is -1
pub fn soft_clamp<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    (input * FRAC_PI_2).sin()
    //input.tanh()
}

pub fn softmax1<B: Backend, const D: usize>(input: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    burn::tensor::activation::quiet_softmax(input, dim)
}
pub fn softmax<B: Backend, const D: usize>(input: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    burn::tensor::activation::softmax(input, dim)
}
pub fn sigmoid<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    burn::tensor::activation::sigmoid(input)
}

////////////////////////////////////////////////////////
////                Logic functions                 ////
////////////////////////////////////////////////////////

//  Inputs must be -1 to 1
// Don't do correction
pub fn and<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>) -> Tensor<B, D> {
    post_process(a * b)
}

pub fn or<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>) -> Tensor<B, D> {
    let ab = a.clone() * b.clone();
    post_process(a + b - ab)
}

pub fn xor<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>) -> Tensor<B, D> {
    post_process(-a * b)
}

pub fn not<B: Backend, const D: usize>(a: Tensor<B, D>) -> Tensor<B, D> {
    -a
}

////////////////////////////////////////////////////////
////                Post processing                 ////
////////////////////////////////////////////////////////

pub fn post_process<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    //smoothstep_parameterized(input)
    input
}

pub fn smoothstep<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let x = input.clamp(0.0, 1.0);

    3.0 * x.clone().powi_scalar(2) - 2.0 * x.powi_scalar(3)
}

pub fn smoothstep_parameterized<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let x = (input * 0.5 + 0.5).clamp(0.0, 1.0);
    let d = 0.5;

    (d * x.clone() + 3.0 * (1.0 - d) * x.clone().powi_scalar(2)
        - 2.0 * (d - 1.0) * x.powi_scalar(3))
        * 2.0
        - 1.0
}
