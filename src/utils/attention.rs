use burn::{module::Param, prelude::*};

use crate::utils::{expander::Expander, gate::LogicGate, reduction::Reducer, softmax1};

#[derive(Module, Debug)]
pub struct BinaryAttention<B: Backend> {
    // TODO: Position embedding
    bias: Param<Tensor<B, 1>>,
    mask: Tensor<B, 2, Bool>,
    alibi_dist: Tensor<B, 2>,
    slope: Param<Tensor<B, 1>>,
    temperature: Param<Tensor<B, 1>>,
}

impl<B: Backend> BinaryAttention<B> {
    pub fn init(device: &B::Device, max_seq_len: usize) -> Self {
        let mask = Tensor::tril_mask([max_seq_len, max_seq_len], 0, device);

        let pos = Tensor::arange(0..max_seq_len as i64, device).float();
        let q_pos = pos.clone().unsqueeze_dim(1);
        let k_pos = pos.unsqueeze_dim(0);
        let dist = q_pos - k_pos;
        let alibi_dist = dist.mask_fill(mask.clone(), 0);

        BinaryAttention {
            bias: todo!(),
            mask,
            alibi_dist,
            temperature: todo!(),
            slope: todo!(),
        }
    }

    // Tensor dims are [batch_size, tokens, embedding]
    pub fn forward(&self, q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
        let q: Tensor<B, 4> = q.unsqueeze_dim(2);
        let k: Tensor<B, 4> = k.unsqueeze_dim(1);
        // [batch, query_token, key_token, embedding * embedding]
        let xnor = q * k;
        // sum is like popcnt - n/2
        let sum: Tensor<B, 3> = xnor.sum_dim(3).squeeze_dim(3); // Sum across embeddings

        let scores = sum * self.temperature.val().unsqueeze() + self.bias.val().unsqueeze();

        let scores = scores.mask_fill(self.mask.clone().unsqueeze(), -f32::INFINITY);
        let scores_normilized = softmax1(scores, 2);

        let score_expanded: Tensor<B, 4> = scores_normilized.unsqueeze_dim(3);
        let output = (score_expanded * v.unsqueeze_dim(1))
            .sum_dim(2)
            .squeeze_dim(2);

        output
    }
}
