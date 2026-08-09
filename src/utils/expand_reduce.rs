use burn::prelude::*;
use rand::{Rng, seq::IteratorRandom};

use crate::utils::{expander::Expander, gate::LogicGate, reduction::Reducer};

#[derive(Module, Debug)]
pub struct ExpandReduce<B: Backend> {
    expand: Expander<B>,
    reduce: Reducer<B>,
}

impl<B: Backend> ExpandReduce<B> {
    pub fn init(
        device: &B::Device,
        inputs: usize,
        expansion_multiplier: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let expand = Expander::init(device, inputs, inputs * expansion_multiplier, rng);
        let reduce = Reducer::init(device, inputs * expansion_multiplier, inputs);
        ExpandReduce { expand, reduce }
    }

    // Tensors are 2D because of batching
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.expand.forward(x);
        let x = self.reduce.forward(x);
        x
    }
}
