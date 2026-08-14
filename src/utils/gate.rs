use crate::utils::{entropy::Entropy, *};
use burn::{module::Param, nn::Initializer, prelude::*};

#[derive(Module, Debug)]
pub struct LogicGate<B: Backend> {
    params: [Param<Tensor<B, 1>>; 4],
}

impl<B: Backend> LogicGate<B> {
    pub fn init(device: &B::Device, inputs: usize) -> LogicGate<B> {
        let offset = 0.1;
        let ones = Initializer::Uniform {
            min: 1.0 - offset,
            max: 1.0,
        };

        let zeros = Initializer::Uniform {
            min: -1.0,
            max: -1.0 + offset,
        };
        /*let ones = Initializer::Uniform {
            min: -1.0,
            max: 1.0,
        };
        let zeros = Initializer::Uniform {
            min: -1.0,
            max: 1.0,
        };*/

        LogicGate {
            params: [
                ones.init(Shape::new([inputs]), device),
                ones.init(Shape::new([inputs]), device),
                zeros.init(Shape::new([inputs]), device),
                zeros.init(Shape::new([inputs]), device),
            ],
        }
    }

    // Tensors are 2D because of batching
    pub fn forward<const D: usize>(
        &self,
        a: Tensor<B, D>,
        b: Tensor<B, D>,
        entropy: &mut Entropy<B>,
    ) -> Tensor<B, D> {
        //a.repeat_dim(1, times);
        //let a = a.clamp(-1.0, 1.0);
        //let b = b.clamp(-1.0, 1.0);
        let [w00, w01, w10, w11] = &self.params;
        entropy.add_entropy(w00.val().tanh());
        entropy.add_entropy(w01.val().tanh());
        entropy.add_entropy(w10.val().tanh());
        entropy.add_entropy(w11.val().tanh());

        //assert_eq!(a.dims()[1], w00.dims()[0]);

        /*assert!(
            w00.val()
                .clone()
                .equal_elem(f32::NAN)
                .any()
                .into_data()
                .convert::<bool>()
                .as_slice::<bool>()
                .unwrap()[0]
                == false,
        );
        assert!(
            a.clone()
                .equal_elem(f32::NAN)
                .any()
                .into_data()
                .convert::<bool>()
                .as_slice::<bool>()
                .unwrap()[0]
                == false,
        );*/
        let correction = 1.0 / 16.0
            * (a.clone().powi_scalar(2) + b.clone().powi_scalar(2) - 2)
            * (1.0
                - soft_clamp(w00.val().unsqueeze())
                    * soft_clamp(w01.val().unsqueeze())
                    * soft_clamp(w10.val().unsqueeze())
                    * soft_clamp(w11.val().unsqueeze()))
            * (soft_clamp(w00.val().unsqueeze())
                + soft_clamp(w01.val().unsqueeze())
                + soft_clamp(w10.val().unsqueeze())
                + soft_clamp(w11.val().unsqueeze()));

        let a0: Tensor<B, D> = 1.0 - a.clone();
        let a1: Tensor<B, D> = 1.0 + a;

        let b0: Tensor<B, D> = 1.0 - b.clone();
        let b1: Tensor<B, D> = 1.0 + b;

        let lookup_table = (a0.clone() * b0.clone() * soft_clamp(w00.val().unsqueeze())
            + a0 * b1.clone() * soft_clamp(w01.val().unsqueeze())
            + a1.clone() * b0 * soft_clamp(w10.val().unsqueeze())
            + a1 * b1 * soft_clamp(w11.val().unsqueeze()))
            / 4.0;

        post_process(correction + lookup_table)
    }
}
