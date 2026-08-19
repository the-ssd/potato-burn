use burn::{
    Tensor,
    tensor::{Distribution, backend::Backend},
};

#[derive(Debug)]
pub struct Entropy<B: Backend> {
    pub entropy: Option<Tensor<B, 1>>,
    pub entropy_sources: usize,
    sign: bool,
}

impl<B: Backend> Entropy<B> {
    pub fn new(sign: bool, device: &B::Device) -> Self {
        Entropy {
            entropy: Some(Tensor::from_data([0.0], device)),
            entropy_sources: 0,
            sign,
        }
    }

    pub fn add_entropy<const D: usize>(&mut self, data: &mut Tensor<B, D>) {
        // 0 at -1 and 1
        // 1 at 0
        //println!("{data}");

        let entropy: Tensor<B, D> = 1.0 - data.clone().powi_scalar(2);
        if self.sign {
            /*let random = data
            .random_like(Distribution::Default)
            .lower_elem(self.sign_threshold);*/
            let sign = data.clone().sign();
            // STE
            let sign = data.clone() - data.clone().detach() + sign.detach();

            *data = sign;
            //*data = data.clone().mask_where(random, sign);

            //*data = data.clone().sign();
        }
        //let entropy: Tensor<B, D> = 1.0 - data.abs();
        //println!("{entropy}");

        /*assert!(
            entropy
                .clone()
                .greater_equal_elem(1.01)
                .any()
                .into_data()
                .convert::<bool>()
                .as_slice::<bool>()
                .unwrap()[0]
                == false
        );*/

        let sources = entropy.shape().num_elements();
        let entropy = entropy.sum();

        self.entropy = Some(self.entropy.take().unwrap().add(entropy));

        self.entropy_sources += sources;
    }

    pub fn normalized(&self) -> Tensor<B, 1> {
        self.entropy.clone().unwrap() / self.entropy_sources as f32
    }
}
