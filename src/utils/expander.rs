use burn::prelude::*;
use rand::{Rng, seq::IteratorRandom};

use crate::utils::{entropy::Entropy, gate::LogicGate};

#[derive(Module, Debug)]
pub struct Expander<B: Backend> {
    gates: Vec<LogicGate<B>>,
    offsets: Vec<usize>,
}

impl<B: Backend> Expander<B> {
    pub fn init(device: &B::Device, inputs: usize, outputs: usize, rng: &mut impl Rng) -> Self {
        let copies = outputs / inputs;
        let mut gates = vec![];
        let mut offsets = (1..inputs).sample(rng, copies);

        for i in 0..copies {
            gates.push(LogicGate::init(device, inputs));
        }

        Expander { gates, offsets }
    }

    // Tensors are 2D because of batching
    pub fn forward(&self, input: Tensor<B, 2>, entropy: &mut Entropy<B>) -> Tensor<B, 2> {
        let mut outputs = vec![];
        for i in 0..self.offsets.len() {
            let offset = self.offsets[i];
            let gate = &self.gates[i];

            let a = input.clone();
            let b = input.clone().roll_dim(offset, 1);
            outputs.push(gate.forward(a, b, entropy));
        }

        Tensor::cat(outputs, 1)
    }
}
