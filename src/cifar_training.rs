use std::time::Instant;

use crate::{
    cifar_model::CifarModel,
    data::{CIFAR10Loader, ClassificationBatch, ClassificationBatcher},
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::ImageFolderDataset},
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, SgdConfig},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, LossMetric, PerplexityMetric},
    },
};

const NUM_CLASSES: u8 = 10;
const ARTIFACT_DIR: &str = "/tmp/custom-image-dataset";

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub optimizer: AdamConfig,
    #[config(default = 2)] // Burn leaks VRAM each epoch
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 6)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 0.02)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: B::Device) {
    create_artifact_dir(ARTIFACT_DIR);

    config
        .save(format!("{ARTIFACT_DIR}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    // Dataloaders
    let batcher_train = ClassificationBatcher::<B>::new(device.clone());
    let batcher_valid = ClassificationBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::cifar10_train());

    // NOTE: we use the CIFAR-10 test set as validation for demonstration purposes
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::cifar10_test());

    // Learner config
    let training = SupervisedTraining::new(ARTIFACT_DIR, dataloader_train, dataloader_test)
        .metric_train_numeric(PerplexityMetric::new())
        .metric_valid_numeric(PerplexityMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        //.renderer(crate::cli_renderer::CliMetricsRenderer::new())
        //.renderer(burn::train::renderer::CliMetricsRenderer::new())
        .summary();

    let model = CifarModel::new(NUM_CLASSES.into(), &device);

    // Training
    let now = Instant::now();
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));
    let elapsed = now.elapsed().as_secs();
    println!("Training completed in {}m{}s", (elapsed / 60), elapsed % 60);

    result
        .model
        .save_file(format!("{ARTIFACT_DIR}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
