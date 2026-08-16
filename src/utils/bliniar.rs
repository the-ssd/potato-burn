use burn::{
    module::Param,
    nn::{Initializer, Linear},
    prelude::*,
};

use crate::utils::entropy::Entropy;

/// Configuration to create a [`BLinear`] layer using the [init function](BLinearConfig::init).
#[derive(Config, Debug)]
pub struct BLinearConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the output features.
    pub d_output: usize,

    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::KaimingUniform{gain:1.0/f64::sqrt(3.0), fan_out_only:false}")]
    //#[config(default = "Initializer::Zeros")]
    pub initializer: Initializer,
}

/// Applies a linear transformation to the input tensor.
///
/// Should be created with [BLinearConfig]
///
/// `O = tanh(IW + b)`
#[derive(Module, Debug)]
pub struct BLinear<B: Backend> {
    /// Matrix of shape `[d_output, d_input]`
    /// NOTE: Transposed
    pub weight: Param<Tensor<B, 2>>,
    /// Vector of size `d_output`
    pub bias: Param<Tensor<B, 1>>,
    pub temperature: Param<Tensor<B, 1>>,
}

impl BLinearConfig {
    /// Initialize a new [`Linear`] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BLinear<B> {
        let shape = [self.d_input, self.d_output];
        // NOTE: Transposed
        //let shape = [self.d_output, self.d_input];

        let weight =
            self.initializer
                .init_with(shape, Some(self.d_input), Some(self.d_output), device);

        /*let bias = self.initializer.init_with(
            [self.d_output],
            Some(self.d_input),
            Some(self.d_output),
            device,
        );*/
        let bias = Initializer::Zeros.init_with(
            [self.d_output],
            Some(self.d_input),
            Some(self.d_output),
            device,
        );

        BLinear {
            weight,
            bias,
            temperature: Param::from_data([1.0], device),
        }
    }
}

impl<B: Backend> BLinear<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Arguments
    ///
    /// - `input` - The input tensor of shape `[..., d_input]`.
    ///
    /// # Shapes
    ///
    /// - input: `[..., d_input]`
    /// - output: `[..., d_output]`
    ///
    /// # Returns
    ///
    /// The transformed tensor of shape `[..., d_output]`.
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<B, D>,
        entropy: &mut Entropy<B>,
    ) -> Tensor<B, D> {
        // .transpose()
        /*B::linear(
            input.into_primitive().tensor(),
            self.weight.val().into_primitive().tensor(),
            Some(self.bias.val().into_primitive().tensor()),
        );*/
        let mut weight = self.weight.val().tanh();
        entropy.add_entropy(&mut weight);

        burn::tensor::module::linear(
            input,
            weight.mul(self.temperature.val().unsqueeze()),
            Some(self.bias.val()),
        )
        .tanh()
        //(input.matmul(self.weight.val().unsqueeze()) + self.bias.val().unsqueeze()).tanh()
    }
}
