use crate::{
    data::{FalconTokenizer, Tokenizer, TrainingTextGenerationBatch},
    utils::{entropy::Entropy, transformer_layer::TransformerLayer, *},
};
use burn::{
    module::Param,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        attention::generate_autoregressive_mask,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

#[derive(Config, Debug)]
pub struct TextGenerationModelConfig {
    #[config(default = 4)]
    num_layers: usize,
    #[config(default = 256)]
    max_seq_length: usize,
    #[config(default = 256)]
    embedding_dimensions: usize,
    #[config(default = 4)]
    n_heads: usize,
    #[config(default = 256)]
    pub d_model: usize,
}

#[derive(Module, Debug)]
pub struct TextGenerationModel<B: Backend> {
    transformer_layers: Vec<TransformerLayer<B>>,
    //bias: Param<Tensor<B, 1>>,
    embedding_token: Embedding<B>,
    temperature: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
    //embedding_pos: Embedding<B>,
    //output: Linear<B>,
    embedding_dimensions: usize,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}
/*
======================== Learner Summary ========================
Model:
"TextGenerationModel" {
  transformer: TransformerEncoder {d_model: 256, d_ff: 512, n_heads: 2, n_layers: 2, dropout: 0.1, norm_first: true, quiet_softmax: false, params: 1054208}
  embedding_token: Embedding {n_embedding: 32768, d_model: 256, params: 8388608}
  embedding_dimensions: 256
  vocab_size: 32768
  pad_token: 0
  max_seq_length: 256
  params: 9442816
}
Total Epochs: 1


| Split | Metric        | Min.     | Epoch    | Max.     | Epoch    |
|-------|---------------|----------|----------|----------|----------|
| Train | Accuracy      | 21.432   | 1        | 21.432   | 1        |
| Train | Learning Rate | 2.500e-3 | 1        | 2.500e-3 | 1        |
| Train | Loss          | 5.538    | 1        | 5.538    | 1        |
| Train | Perplexity    | 964278.726| 1        | 964278.726| 1        |
| Valid | Accuracy      | 25.428   | 1        | 25.428   | 1        |
| Valid | Loss          | 4.822    | 1        | 4.822    | 1        |
| Valid | Perplexity    | 139.599  | 1        | 139.599  | 1        |


  params: 9444874
}
Total Epochs: 1


| Split | Metric        | Min.     | Epoch    | Max.     | Epoch    |
|-------|---------------|----------|----------|----------|----------|
| Valid | Accuracy      | 35.337   | 1        | 35.337   | 1        |
| Valid | Loss          | 2.694    | 1        | 2.694    | 1        |
| Valid | Perplexity    | 42.219   | 1        | 42.219   | 1        |



*/
impl TextGenerationModelConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        tokenizer: &FalconTokenizer,
    ) -> TextGenerationModel<B> {
        //let output = LinearConfig::new(self.transformer.d_model, self.vocab_size).init(device);
        //let transformer = self.transformer.init(device);
        let embedding_token =
            EmbeddingConfig::new(tokenizer.vocab_size(), self.embedding_dimensions)
                .with_initializer(nn::Initializer::Uniform {
                    min: -0.1,
                    max: 0.1,
                })
                .init(device);
        let mut transformer_layers = vec![];
        for i in 0..self.num_layers {
            transformer_layers.push(TransformerLayer::init(
                device,
                self.max_seq_length,
                self.n_heads,
                self.d_model,
            ));
        }

        TextGenerationModel {
            transformer_layers,
            embedding_token,
            temperature: Param::from_data([1.0], device),
            bias: Param::from_data([0.0], device),
            //output,
            embedding_dimensions: self.embedding_dimensions,
            vocab_size: tokenizer.vocab_size(),
            pad_token: tokenizer.pad_token(),
            max_seq_length: self.max_seq_length,
        }
    }
}

impl<B: Backend> TextGenerationModel<B> {
    pub fn forward_training(
        &self,
        item: TrainingTextGenerationBatch<B>,
        sign: bool,
    ) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = item.tokens_inputs.dims();
        let device = &self.devices()[0];
        let mut entropy = Entropy::new(sign, device);

        let inputs = item.tokens_inputs.to_device(device);
        let targets = item.targets.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        /*let index_positions = Tensor::arange(0..seq_length as i64, device)
        .reshape([1, seq_length])
        .repeat_dim(0, batch_size);*/

        //let embedding_positions = self.embedding_pos.forward(index_positions);
        let mut embedding_tokens = self.embedding_token.forward(inputs).tanh();
        entropy.add_entropy(&mut embedding_tokens);
        //let embedding = (embedding_positions + embedding_tokens) / 2;
        //let embedding = xor(sigmoid(embedding_tokens), sigmoid(embedding_positions));
        let embedding = embedding_tokens;

        //let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        /*let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask_pad)
                .mask_attn(mask_attn),
        );*/
        let mut output = embedding;
        for transformer in &self.transformer_layers {
            output = transformer.forward(output, &mask_pad, &mut entropy);
        }
        let encoded = output;

        let embeddings = self.embedding_token.weight.val();
        let embeddings = embeddings.transpose().tanh();
        let output = encoded.matmul(embeddings.unsqueeze()) * self.temperature.val().unsqueeze();
        let output_flatten = output.reshape([batch_size * seq_length, self.vocab_size]);

        /*let output = self.output.forward(encoded);
        let output_flatten = output.reshape([batch_size * seq_length, self.vocab_size]);*/
        let targets_flatten = targets.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token]))
            .init(&output_flatten.device());
        let loss = loss.forward(output_flatten.clone(), targets_flatten.clone());
        let entropy = entropy.normalized();
        //println!("Entropy: {}", entropy.clone().into_scalar());
        //let loss = loss + entropy * 0.1;

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_flatten,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for TextGenerationModel<B> {
    type Input = TrainingTextGenerationBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: TrainingTextGenerationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_training(item, false);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> InferenceStep for TextGenerationModel<B> {
    type Input = TrainingTextGenerationBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: TrainingTextGenerationBatch<B>) -> ClassificationOutput<B> {
        self.forward_training(item, false)
    }
}
