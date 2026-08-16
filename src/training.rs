use crate::{
    cli_renderer::CliMetricsRenderer,
    data::{FalconTokenizer, TextGenerationBatcher, TextGenerationItem, Tokenizer},
    model::TextGenerationModelConfig,
};
use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{Dataset, transform::SamplerDataset},
    },
    lr_scheduler::{self, noam::NoamLrSchedulerConfig},
    nn::transformer::TransformerEncoderConfig,
    optim::{AdamConfig, decay::WeightDecayConfig},
    prelude::*,
    record::{CompactRecorder, DefaultRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        Learner, SupervisedTraining,
        metric::{AccuracyMetric, LearningRateMetric, LossMetric, PerplexityMetric},
    },
};
use log::info;
use std::sync::Arc;

#[derive(Config, Debug)]
pub struct ExperimentConfig {
    pub model: TextGenerationModelConfig,
    optimizer: AdamConfig,
    #[config(default = 256)]
    pub max_seq_length: usize,
    #[config(default = 3)]
    pub batch_size: usize,
    #[config(default = 1)]
    num_epochs: usize,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        ExperimentConfig::new(
            TextGenerationModelConfig::new(),
            burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
        )
    }
}

pub fn train<B: AutodiffBackend, D: Dataset<TextGenerationItem> + 'static>(
    device: B::Device,
    dataset_train: D,
    dataset_test: D,
    config: ExperimentConfig,
    artifact_dir: &str,
) {
    let tokenizer = Arc::new(FalconTokenizer::default());
    let batcher = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_length);

    let mut model = config.model.init::<B>(&device, &tokenizer);
    if std::env::args().any(|x| x == "--continue") {
        model = model
            .load_file(
                format!("{artifact_dir}/model"),
                &DefaultRecorder::new(),
                &device,
            )
            .unwrap();
    }

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .num_workers(8)
        .build(SamplerDataset::new(dataset_train, 100_000)); // 20_000

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(6)
        .build(SamplerDataset::new(dataset_test, 10_000));

    let accum = 3; // Effective batch size = 3 * 3 = 9
    let optim = config.optimizer.init();
    /*let lr_scheduler = NoamLrSchedulerConfig::new(0.05 / accum as f64)
    .with_warmup_steps(1000)
    .with_model_size(config.model.d_model)
    .init()
    .unwrap();*/

    /*let lr_scheduler = lr_scheduler::linear::LinearLrSchedulerConfig::new(0.001, 0.02, 2000)
    .init()
    .unwrap();*/
    let lr_scheduler = 0.03 / accum as f64;

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        //.metric_train(CudaMetric::new())
        //.metric_valid(CudaMetric::new())
        .metric_train_numeric(PerplexityMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_valid_numeric(PerplexityMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_train_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_valid_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_train_numeric(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .grads_accumulation(accum)
        .num_epochs(config.num_epochs)
        //.renderer(CliMetricsRenderer::new())
        .summary();

    let result = training.launch(Learner::new(model, optim, lr_scheduler));

    config.save(format!("{artifact_dir}/config.json")).unwrap();

    DefaultRecorder::new()
        .record(
            result.model.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}
