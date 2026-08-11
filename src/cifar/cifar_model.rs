use burn::{
    module::Param,
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLossConfig,
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use log::debug;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{
    cifar::ClassificationBatch,
    utils::{expand_reduce::ExpandReduce, gate::LogicGate, *},
};

#[derive(Module, Debug)]
pub struct CifarModel<B: Backend> {
    gates: Vec<LogicGate<B>>,

    temperature: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
}

/*

TODO:
Fully connected LogicLayer into DNF: n^2/2 + n*n^2/2
Reduction (X)OR "distant" bits: 2N -> N

ANF connected into DNF: n+n^2 + n^2
DNF vs CNF?

*/
/*
| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | Accuracy   | 33.398   | 1        | 33.398   | 1        |
| Train | Loss       | 1.867    | 1        | 1.867    | 1        |
| Train | Perplexity | 6.971    | 1        | 6.971    | 1        |
| Valid | Accuracy   | 39.510   | 1        | 39.510   | 1        |
| Valid | Loss       | 1.720    | 1        | 1.720    | 1        |
| Valid | Perplexity | 5.575    | 1        | 5.575    | 1        |

  params: 294923
}
Total Epochs: 1


| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | Accuracy   | 30.768   | 1        | 30.768   | 1        |
| Train | Loss       | 1.939    | 1        | 1.939    | 1        |
| Train | Perplexity | 7.536    | 1        | 7.536    | 1        |
| Valid | Accuracy   | 37.710   | 1        | 37.710   | 1        |
| Valid | Loss       | 1.784    | 1        | 1.784    | 1        |
| Valid | Perplexity | 6.094    | 1        | 6.094    | 1        |

  params: 196619
}
Total Epochs: 1


| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | Accuracy   | 33.358   | 1        | 33.358   | 1        |
| Train | Loss       | 1.874    | 1        | 1.874    | 1        |
| Train | Perplexity | 7.150    | 1        | 7.150    | 1        |
| Valid | Accuracy   | 39.800   | 1        | 39.800   | 1        |
| Valid | Loss       | 1.702    | 1        | 1.702    | 1        |
| Valid | Perplexity | 5.664    | 1        | 5.664    | 1        |

*/

impl<B: Backend> CifarModel<B> {
    pub fn new(num_classes: usize, device: &Device<B>) -> Self {
        let mut prng = &mut SmallRng::seed_from_u64(42);
        Self {
            temperature: Param::from_tensor(Tensor::<B, 1>::from_data([1.0], device)),
            bias: Param::from_tensor(Tensor::<B, 1>::from_data([0.0; 10], device)),
            gates: vec![
                /*ExpandReduce::init(device, 32 * 32 * 3 * 2, 4, prng),
                ExpandReduce::init(device, 32 * 32 * 3 * 2, 4, prng),
                ExpandReduce::init(device, 32 * 32 * 3 * 2, 4, prng),
                ExpandReduce::init(device, 32 * 32 * 3 * 2, 4, prng),*/
                LogicGate::init(device, 32 * 32 * 3 * 2),
                LogicGate::init(device, 32 * 32 * 3 * 2),
                LogicGate::init(device, 32 * 32 * 3 * 2),
                LogicGate::init(device, 32 * 32 * 3 * 2),
                LogicGate::init(device, 32 * 32 * 3 * 2),
                LogicGate::init(device, 32 * 32 * 3 * 2),
                LogicGate::init(device, 32 * 32 * 3 * 2),
                LogicGate::init(device, 32 * 32 * 3 * 2),
                /*LogicGate::init(device, 32 * 32 * 3),
                LogicGate::init(device, 32 * 32 * 3),
                LogicGate::init(device, 32 * 32 * 3),
                LogicGate::init(device, 32 * 32 * 3),
                LogicGate::init(device, 32 * 32 * 3),
                LogicGate::init(device, 32 * 32 * 3),
                LogicGate::init(device, 32 * 32 * 3),
                LogicGate::init(device, 32 * 32 * 3),*/
            ],
        }
    }

    // Tensor is 4d because batch + 2d image + 3 color channels
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut prng = &mut SmallRng::seed_from_u64(42);
        // Put all pixel data into 1 dimension
        let mut gate_input = input.flatten(1, 3);

        for gate in self.gates.iter() {
            let shifted_x = gate_input
                .clone()
                .roll_dim(prng.random_range(1..gate_input.shape()[1]), 1);

            gate_input = gate.forward(gate_input, shifted_x);
            assert!(
                gate_input
                    .clone()
                    .equal_elem(f32::NAN)
                    .any()
                    .into_data()
                    .convert::<bool>()
                    .as_slice::<bool>()
                    .unwrap()[0]
                    == false,
                "{gate:?}"
            );
            //gate_input = gate.forward(gate_input);
        }
        let output = gate_input;

        let [batch_size, num_elements] = output.shape().dims();
        let bucket_size = num_elements / 10;

        let mut sums = vec![];
        for i in 0..10 {
            let start = i * bucket_size;
            let end = ((i + 1) * bucket_size).min(num_elements);
            assert!(end - start > 0, "{start}, {end}");

            sums.push(output.clone().slice([0..batch_size, start..end]).sum_dim(1));
        }
        let output = Tensor::cat(sums, 1) / bucket_size as f32 * self.temperature.val().unsqueeze();

        //let temperature = self.temperature.val().unsqueeze();
        //let temperature = self.temperature.val().expand(output.shape());
        return output + self.bias.val().unsqueeze(); // * temperature;
    }
}

impl<B: Backend> CifarModel<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for CifarModel<B> {
    type Input = ClassificationBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: ClassificationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for CifarModel<B> {
    type Input = ClassificationBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: ClassificationBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
