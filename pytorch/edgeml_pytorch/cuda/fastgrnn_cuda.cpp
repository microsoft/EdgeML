#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> fastgrnn_cuda_forward(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor bias_z,
    torch::Tensor bias_h_prime,
    torch::Tensor old_h,
    torch::Tensor zeta,
    torch::Tensor nu);

std::vector<torch::Tensor> fastgrnn_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor input,
    torch::Tensor old_h,
    torch::Tensor z_t,
    torch::Tensor h_prime_t,
    torch::Tensor pre_comp,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor bias_z,
    torch::Tensor bias_h_prime,
    torch::Tensor zeta,
    torch::Tensor nu);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fastgrnn_forward(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor bias_z,
    torch::Tensor bias_h_prime,
    torch::Tensor old_h,
    torch::Tensor zeta,
    torch::Tensor nu) {
  CHECK_INPUT(input);
  CHECK_INPUT(w);
  CHECK_INPUT(u);
  CHECK_INPUT(bias_z);
  CHECK_INPUT(bias_h_prime);
  CHECK_INPUT(old_h);
  CHECK_INPUT(zeta);
  CHECK_INPUT(nu);

  return fastgrnn_cuda_forward(input, w, u, bias_z, bias_h_prime, old_h, zeta, nu);
}

std::vector<torch::Tensor> fastgrnn_backward(
    torch::Tensor grad_h,
    torch::Tensor input,
    torch::Tensor old_h,
    torch::Tensor z_t,
    torch::Tensor h_prime_t,
    torch::Tensor pre_comp,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor bias_z,
    torch::Tensor bias_h_prime,
    torch::Tensor zeta,
    torch::Tensor nu) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(input);
  CHECK_INPUT(old_h);
  CHECK_INPUT(z_t);
  CHECK_INPUT(h_prime_t);
  CHECK_INPUT(pre_comp);
  CHECK_INPUT(w);
  CHECK_INPUT(u);
  CHECK_INPUT(bias_z);
  CHECK_INPUT(bias_h_prime);
  CHECK_INPUT(zeta);
  CHECK_INPUT(nu);

  return fastgrnn_cuda_backward(
    grad_h,
    input,
    old_h,
    z_t,
    h_prime_t,
    pre_comp,
    w,
    u,
    bias_z,
    bias_h_prime,
    zeta,
    nu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fastgrnn_forward, "FastGRNN forward (CUDA)");
  m.def("backward", &fastgrnn_backward, "FastGRNN backward (CUDA)");
}
