use burn::prelude::*;

use crate::utils::{entropy::Entropy, gate::LogicGate};

#[derive(Module, Debug)]
pub struct Reducer<B: Backend> {
    gates: Vec<LogicGate<B>>,
    inputs: usize,
}

impl<B: Backend> Reducer<B> {
    pub fn init(device: &B::Device, inputs: usize, outputs: usize) -> Self {
        assert!(inputs % 2 == 0);
        let diff = inputs / outputs;
        let diff = (diff as f32).log2();
        assert!(diff.fract() == 0.0, "{diff}");
        let num_reducers = diff as usize;

        let mut gates = vec![];

        let mut size = inputs;
        for i in 0..num_reducers {
            gates.push(LogicGate::init(device, size / 2));
            size /= 2;
        }

        Reducer { inputs, gates }
    }

    // Tensors are 2D because of batching
    pub fn forward(&self, mut input: Tensor<B, 2>, entropy: &mut Entropy<B>) -> Tensor<B, 2> {
        for gate in &self.gates {
            assert!(input.shape()[1] % 2 == 0);
            let size = input.shape()[1] / 2;
            let mut vec = input.split(size, 1);
            let a = vec.remove(0);
            let b = vec.remove(0);
            input = gate.forward(a, b, entropy);
        }
        input
    }
}
