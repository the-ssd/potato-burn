use crate::utils::{
    attention::BinaryMultiHeadAttention,
    bliniar::{BLinear, BLinearConfig},
    expand_reduce::ExpandReduce,
    gate::LogicGate,
};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct TransformerLayer<B: Backend> {
    attention: BinaryMultiHeadAttention<B>,
    residual1: LogicGate<B>,
    liniar1: BLinear<B>,
    liniar2: BLinear<B>,
    er_layer: ExpandReduce<B>,
    residual2: LogicGate<B>,
}

impl<B: Backend> TransformerLayer<B> {
    pub fn init(device: &B::Device, max_seq_len: usize, n_heads: usize, d_model: usize) -> Self {
        TransformerLayer {
            attention: BinaryMultiHeadAttention::init(
                device,
                max_seq_len,
                n_heads,
                d_model,
                d_model / n_heads,
            ),
            residual1: LogicGate::init(device, d_model),
            liniar1: BLinearConfig::new(d_model, d_model * 2).init(device),
            liniar2: BLinearConfig::new(d_model * 2, d_model).init(device),
            er_layer: ExpandReduce::init(device, d_model, d_model, &mut rand::rng()),
            residual2: LogicGate::init(device, d_model),
        }
    }

    // Tensors are 2D because of batching
    pub fn forward(&self, x: Tensor<B, 3>, mask_pad: &Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let x_attention = self
            .attention
            .forward(x.clone(), x.clone(), x.clone(), mask_pad);
        let x = self.residual1.forward(x, x_attention);
        let ffl = self.liniar1.forward(x.clone());
        let ffl = self.liniar2.forward(ffl);
        //let ffl = self.er_layer.forward(x.clone());
        let x = self.residual2.forward(x, ffl);
        x
    }
}
