#![recursion_limit = "256"]
#![allow(unused)]
//#![feature(f16)]

use crate::{cifar_training::TrainingConfig, data::DbPediaDataset, training::ExperimentConfig};
use burn::optim::{AdamConfig, decay::WeightDecayConfig};

mod cifar_model;
mod cifar_training;
mod cli_renderer;
mod data;
mod model;
mod utils;

mod training;

type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;

fn main() {
    let device = Default::default();
    // Force Vulkan
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );

    let config = TrainingConfig::new(AdamConfig::new());

    cifar_training::train::<Backend>(config, device);

    /*let config = ExperimentConfig::default();


    crate::training::train::<Backend, DbPediaDataset>(
        device,
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "target/artifacts",
    );*/
}
