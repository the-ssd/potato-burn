#![recursion_limit = "256"]
#![allow(unused)]
//#![feature(f16)]

use crate::{data::DbPediaDataset, training::ExperimentConfig};
use burn::optim::{AdamConfig, decay::WeightDecayConfig};

//mod cifar;
mod cli_renderer;
mod data;
mod inference;
mod model;
mod utils;

mod training;

#[cfg(not(feature = "f16"))]
type Elem = f32;
#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;

#[cfg(not(feature = "cuda"))]
type Backend = burn::backend::Autodiff<burn::backend::Wgpu<Elem>>;
#[cfg(feature = "cuda")]
type Backend = burn::backend::Autodiff<burn::backend::LibTorch<Elem>>;

fn main() {
    if std::env::args().any(|x| x == "--infer") {
        return inference::gpu_infer::infer();
    }

    #[cfg(not(feature = "cuda"))]
    let device = Default::default();
    #[cfg(feature = "cuda")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);

    // Force Vulkan
    #[cfg(not(feature = "cuda"))]
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );

    /*let config = TrainingConfig::new(AdamConfig::new());

    cifar::cifar_training::train::<Backend>(config, device);*/

    let mut config = ExperimentConfig::default();

    if let Some(batches) = std::env::args().find(|x| x.starts_with("--batches=")) {
        let batches: usize = batches.strip_prefix("--batches=").unwrap().parse().unwrap();
        config.batch_size = batches;
    }
    if let Some(dim) = std::env::args().find(|x| x.starts_with("--dim=")) {
        let dim: usize = dim.strip_prefix("--dim=").unwrap().parse().unwrap();
        config.model.d_model = dim;
        config.model.embedding_dimensions = dim;
    }
    if let Some(layers) = std::env::args().find(|x| x.starts_with("--layers=")) {
        let layers: usize = layers.strip_prefix("--layers=").unwrap().parse().unwrap();
        config.model.num_layers = layers;
    }

    crate::training::train::<Backend, DbPediaDataset>(
        device,
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "target/artifacts",
    );
}
