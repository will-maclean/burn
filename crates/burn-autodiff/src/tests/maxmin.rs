#[burn_tensor_testgen::testgen(ad_maxmin)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn should_diff_max_dim() {
        let device = Default::default();
        let tensor_1 =
            TestAutodiffTensor::from_floats([[1.0, 7.0], [-2.0, -3.0]], &device).require_grad();
        let tensor_2 =
            TestAutodiffTensor::from_floats([[4.0, -7.0], [2.0, 3.0]], &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_1.clone().mul(tensor_3.max_dim(1).unsqueeze());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[50.0, 34.0], [40.0, -10.0]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[8.0, 10.0], [56.0, 15.0]]), 5);
    }

    #[test]
    fn should_diff_min_dim() {
        let device = Default::default();
        let tensor_1 =
            TestAutodiffTensor::from_floats([[1.0, 7.0], [-2.0, -3.0]], &device).require_grad();
        let tensor_2 =
            TestAutodiffTensor::from_floats([[4.0, -7.0], [2.0, 3.0]], &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_1.clone().mul(tensor_3.min_dim(1).unsqueeze());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[-42.0, 38.0], [-34.0, -24.0]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[10.0, 8.0], [15.0, 56.0]]), 5);
    }

    #[test]
    fn test_max_dim_complex() {
        let device = Default::default();
        let a: Vec<f32> = vec![0.0, 0.0];
        let b = [0, 0];
        let b: Tensor<TestAutodiffBackend, 2, Int> =
            Tensor::from_data(Data::from(b.as_slice()), &device).reshape([2, 1]);
        let a = Tensor::from_data(Data::from(a.as_slice()), &device)
            .reshape([2, 1])
            .require_grad();

        let loss = a.gather(1, b);
        let loss = loss.clone().max_dim(0) + loss; //No panic if this line is commented out
        let loss = loss.sum();
        let g = loss.backward();
    }
}
