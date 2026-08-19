use std::sync::Arc;

use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use serde::de;

use crate::{
    data::{
        FalconTokenizer, TextGenerationBatch, TextGenerationBatcher, TextGenerationItem, Tokenizer,
        TrainingTextGenerationBatch,
    },
    model::{TextGenerationModel, TextGenerationModelConfig},
    training::ExperimentConfig,
};

type B = burn::backend::Wgpu;

pub fn infer() {
    let config = ExperimentConfig::load("target/artifacts/config.json")
        .unwrap()
        .with_batch_size(1);
    let device: burn::backend::wgpu::WgpuDevice = burn::backend::wgpu::WgpuDevice::DefaultDevice;
    let record = CompactRecorder::new()
        .load("target/artifacts/model".into(), &device)
        .unwrap();

    let tokenizer = Arc::new(FalconTokenizer::default());
    let batcher = TextGenerationBatcher::new(tokenizer.clone(), config.model.max_seq_length);
    let model = config
        .model
        .init::<B>(&device, &tokenizer)
        .load_record(record);

    let mut text = std::env::args().nth(2).unwrap_or(String::new());

    loop {
        let batch: TrainingTextGenerationBatch<B> =
            batcher.batch(vec![TextGenerationItem::new(text.clone())], &device);

        //println!("{}", &batch.targets);
        //let output = model.forward_training(batch, false);
        let output = model.forward_training(batch);

        let predicted: Vec<i32> = output
            .output
            .argmax(1)
            .flatten::<1>(0, 1)
            .into_data()
            .convert::<i32>()
            .into_vec()
            .unwrap();

        if *predicted.last().unwrap() == tokenizer.end_token() as i32 {
            break;
        }

        text.push_str(&tokenizer.decode(&[*predicted.last().unwrap() as usize]));
        println!("Text: {text}");
    }
}
