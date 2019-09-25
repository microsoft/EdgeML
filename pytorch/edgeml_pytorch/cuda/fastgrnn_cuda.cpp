#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> fastgrnn_cuda_forward(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor bias_gate,
    torch::Tensor bias_update,
    torch::Tensor zeta,
    torch::Tensor nu,
    torch::Tensor old_h);

std::vector<torch::Tensor> fastgrnn_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor input,
    torch::Tensor old_h,
    torch::Tensor zeta,
    torch::Tensor nu,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor z,
    torch::Tensor h_prime);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fastgrnn_forward(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor bias_gate,
    torch::Tensor bias_update,
    torch::Tensor zeta,
    torch::Tensor nu,
    torch::Tensor old_h) {
  CHECK_INPUT(input);
  CHECK_INPUT(w);
  CHECK_INPUT(u);
  CHECK_INPUT(bias_gate);
  CHECK_INPUT(bias_update);
  CHECK_INPUT(zeta);
  CHECK_INPUT(nu);
  CHECK_INPUT(old_h);

  return fastgrnn_cuda_forward(input, w, u, bias_gate, bias_update, zeta, nu, old_h);
}

std::vector<torch::Tensor> fastgrnn_backward(
    torch::Tensor grad_h,
    torch::Tensor input,
    torch::Tensor old_h,
    torch::Tensor zeta,
    torch::Tensor nu,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor z,
    torch::Tensor h_prime) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(input);
  CHECK_INPUT(old_h);
  CHECK_INPUT(zeta);
  CHECK_INPUT(nu);
  CHECK_INPUT(z);
  CHECK_INPUT(h_prime);
  CHECK_INPUT(w);
  CHECK_INPUT(u);

  return fastgrnn_cuda_backward(grad_h, input, old_h, zeta, nu, w, u, z, h_prime);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fastgrnn_forward, "FastGRNN forward (CUDA)");
  m.def("backward", &fastgrnn_backward, "FastGRNN backward (CUDA)");
}
