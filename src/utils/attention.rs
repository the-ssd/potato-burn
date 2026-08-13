use burn::{module::Param, prelude::*};

use crate::utils::{
    bliniar::{BLinear, BLinearConfig},
    expander::Expander,
    gate::LogicGate,
    reduction::Reducer,
    softmax1,
};

#[derive(Module, Debug)]
pub struct BinaryMultiHeadAttention<B: Backend> {
    // TODO: Position embedding
    n_heads: usize,
    /// Size of the key and query vectors.
    d_k: usize,
    attention_mask: Tensor<B, 2, Bool>,
    alibi_dist: Tensor<B, 2>,
    alibi_slope: Param<Tensor<B, 1>>,
    temperature1: Param<Tensor<B, 1>>,
    bias1: Param<Tensor<B, 1>>,
    temperature2: Param<Tensor<B, 1>>,
    bias2: Param<Tensor<B, 1>>,

    q: BLinear<B>,
    k: BLinear<B>,
    v: BLinear<B>,
    output_projection: BLinear<B>,
}

impl<B: Backend> BinaryMultiHeadAttention<B> {
    pub fn init(
        device: &B::Device,
        max_seq_len: usize,
        n_heads: usize,
        d_model: usize,
        d_k: usize,
    ) -> Self {
        // TODO: Check if tril is correct
        let attention_mask = Tensor::tril_mask([max_seq_len, max_seq_len], 0, device);
        //let attention_mask = Tensor::triu_mask([max_seq_len, max_seq_len], 0, device);

        let pos = Tensor::arange(0..max_seq_len as i64, device).float();
        let q_pos = pos.clone().unsqueeze_dim(1);
        let k_pos = pos.unsqueeze_dim(0);
        let dist = q_pos - k_pos;
        let dist = dist.clamp_max(64);
        let alibi_dist = dist.mask_fill(attention_mask.clone(), 0);

        BinaryMultiHeadAttention {
            attention_mask,
            alibi_dist,
            temperature1: Param::from_tensor(
                Tensor::<B, 1>::from_data([1.0], device).expand([n_heads]),
            ),
            bias1: Param::from_tensor(Tensor::<B, 1>::from_data([0.0], device).expand([n_heads])),
            temperature2: Param::from_tensor(
                Tensor::<B, 1>::from_data([1.0], device).expand([n_heads]),
            ),
            bias2: Param::from_tensor(Tensor::<B, 1>::from_data([0.0], device).expand([n_heads])),
            alibi_slope: Param::from_tensor(
                Tensor::<B, 1>::from_data([0.1], device).expand([n_heads]),
            ),
            n_heads,
            d_k,
            q: BLinearConfig::new(d_model, d_model).init(device),
            k: BLinearConfig::new(d_model, d_model).init(device),
            v: BLinearConfig::new(d_model, d_model).init(device),
            output_projection: BLinearConfig::new(d_model, d_model).init(device),
        }
    }

    // Tensor dims are [batch_size, tokens, embedding]
    pub fn forward(
        &self,
        q: Tensor<B, 3>,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        mask_pad: &Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_length_1, d_model] = q.dims();
        let [batch_size, seq_length] = mask_pad.dims();

        let q = self.attention_linear(q, &self.q);
        let k = self.attention_linear(k, &self.k);
        let v = self.attention_linear(v, &self.v);

        let alibi_dist = self
            .alibi_dist
            .clone()
            .slice([0..seq_length, 0..seq_length]);
        let alibi = alibi_dist.unsqueeze() * self.alibi_slope.val().unsqueeze_dims(&[0, -1, -1]);
        // TODO: change div?, add ALiBi
        let attn_scores = q.matmul(k.transpose()).div_scalar((self.d_k as f32).sqrt());
        let attn_scores = (attn_scores * self.temperature1.val().unsqueeze_dims(&[0, -1, -1])
            + self.bias1.val().unsqueeze_dims(&[0, -1, -1])
            + alibi)
            .tanh();

        let mask_attn = nn::attention::generate_autoregressive_mask::<B>(
            batch_size,
            seq_length,
            &self.devices()[0],
        );
        let weights = attn_scores
            .mask_fill(mask_attn.unsqueeze_dim(1), -f32::INFINITY)
            .mask_fill(
                mask_pad.clone().reshape([batch_size, 1, 1, seq_length]),
                -f32::INFINITY,
            );
        let weights = softmax1(weights, 3);

        // NOTE: No transposition
        let context = (weights.matmul(v) * self.temperature1.val().unsqueeze_dims(&[0, -1, -1])
            + self.bias1.val().unsqueeze_dims(&[0, -1, -1]))
        .tanh();

        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_length_1, d_model]);

        let context = self.output_projection.forward(context);

        context
        /*let q: Tensor<B, 4> = q.unsqueeze_dim(2);
        let k: Tensor<B, 4> = k.unsqueeze_dim(1);
        // [batch, query_token, key_token, embedding * embedding]
        let xnor = q * k;
        // sum is like popcnt - n/2
        let sum: Tensor<B, 3> = xnor.sum_dim(3).squeeze_dim(3); // Sum across embeddings

        let scores = sum * self.temperature.val().unsqueeze() + self.bias.val().unsqueeze();
        let scores = scores.tanh();

        let scores = scores.mask_fill(self.mask.clone().unsqueeze(), -f32::INFINITY);
        let scores_normilized = softmax1(scores, 2);

        let score_expanded: Tensor<B, 4> = scores_normilized.unsqueeze_dim(3);
        let output = (score_expanded * v.unsqueeze_dim(1))
            .sum_dim(2)
            .squeeze_dim(2);

        output*/
    }

    pub fn attention_linear(&self, x: Tensor<B, 3>, linear: &BLinear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _d_model] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }
}
