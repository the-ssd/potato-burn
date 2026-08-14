use burn::{Tensor, tensor::backend::Backend};

#[derive(Debug)]
pub struct Entropy<B: Backend> {
    pub entropy: Option<Tensor<B, 1>>,
    pub entropy_sources: usize,
}

impl<B: Backend> Entropy<B> {
    pub fn new(device: &B::Device) -> Self {
        Entropy {
            entropy: Some(Tensor::from_data([0.0], device)),
            entropy_sources: 0,
        }
    }

    pub fn add_entropy<const D: usize>(&mut self, data: Tensor<B, D>) {
        // 0 at -1 and 1
        // 1 at 0
        let entropy: Tensor<B, D> = 1.0 - data.powi_scalar(2);

        let entropy = entropy.sum();
        let sources = entropy.shape().flatten().len();

        self.entropy = Some(self.entropy.take().unwrap().add(entropy));

        self.entropy_sources += sources;
    }

    pub fn normalized(&self) -> Tensor<B, 1> {
        self.entropy.clone().unwrap() / self.entropy_sources as f32
    }
}
