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
  torch::Tensor old_h,
  int z_non_linearity,
  torch::Tensor w1,
  torch::Tensor w2,
  torch::Tensor u1,
  torch::Tensor u2);

std::vector<torch::Tensor> fastgrnn_cuda_backward(
  torch::Tensor grad_h,
  torch::Tensor input,
  torch::Tensor old_h,
  torch::Tensor zeta,
  torch::Tensor nu,
  torch::Tensor w,
  torch::Tensor u,
  int z_non_linearity,
  torch::Tensor z,
  torch::Tensor h_prime,
  torch::Tensor w1,
  torch::Tensor w2,
  torch::Tensor u1,
  torch::Tensor u2);

std::vector<torch::Tensor> fastgrnn_unroll_cuda_forward(
  torch::Tensor input,
  torch::Tensor w,
  torch::Tensor u,
  torch::Tensor bias_z,
  torch::Tensor bias_h_prime,
  torch::Tensor zeta,
  torch::Tensor nu,
  torch::Tensor initial_h,
  int z_non_linearity,
  torch::Tensor w1,
  torch::Tensor w2,
  torch::Tensor u1,
  torch::Tensor u2);

std::vector<torch::Tensor> fastgrnn_unroll_cuda_backward(
  torch::Tensor grad_h,
  torch::Tensor input,
  torch::Tensor hidden_states,
  torch::Tensor zeta,
  torch::Tensor nu,
  torch::Tensor w,
  torch::Tensor u,
  torch::Tensor z,
  torch::Tensor h_prime,
  torch::Tensor initial_h,
  int z_non_linearity,
  torch::Tensor w1,
  torch::Tensor w2,
  torch::Tensor u1,
  torch::Tensor u2);

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
  torch::Tensor old_h,
  int z_non_linearity,
  torch::Tensor w1,
  torch::Tensor w2,
  torch::Tensor u1,
  torch::Tensor u2) {
  CHECK_INPUT(input);
  if(w1.size(0) == 0) {
    CHECK_INPUT(w);
  } else {
    CHECK_INPUT(w1);
    CHECK_INPUT(w2);
  }
  if (u1.size(0) == 0) {
    CHECK_INPUT(u);
  } else {
    CHECK_INPUT(u1);
    CHECK_INPUT(u2);
  }
  CHECK_INPUT(bias_gate);
  CHECK_INPUT(bias_update);
  CHECK_INPUT(zeta);
  CHECK_INPUT(nu);
  CHECK_INPUT(old_h);

  return fastgrnn_cuda_forward(input, w, u, bias_gate, bias_update, zeta, nu, old_h, z_non_linearity, w1, w2, u1, u2);
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
  torch::Tensor h_prime,
  torch::Tensor w1,
  torch::Tensor w2,
  torch::Tensor u1,
  torch::Tensor u2,
  int z_non_linearity) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(input);
  CHECK_INPUT(old_h);
  CHECK_INPUT(zeta);
  CHECK_INPUT(nu);
  CHECK_INPUT(z);
  CHECK_INPUT(h_prime);
  if(w1.size(0) == 0) {
    CHECK_INPUT(w);
  } else {
    CHECK_INPUT(w1);
    CHECK_INPUT(w2);
  }
  if (u1.size(0) == 0) {
    CHECK_INPUT(u);
  } else {
    CHECK_INPUT(u1);
    CHECK_INPUT(u2);
  }

  return fastgrnn_cuda_backward(grad_h, input, old_h, zeta, nu, w, u, z_non_linearity, z, h_prime, w1, w2, u1, u2);
}

std::vector<torch::Tensor> fastgrnn_unroll_forward(
  torch::Tensor input,
  torch::Tensor w,
  torch::Tensor u,
  torch::Tensor bias_z,
  torch::Tensor bias_h_prime,
  torch::Tensor zeta,
  torch::Tensor nu,
  torch::Tensor initial_h,
  int z_non_linearity,
  torch::Tensor w1,
  torch::Tensor w2,
  torch::Tensor u1,
  torch::Tensor u2) {
  CHECK_INPUT(input);
  if(w1.size(0) == 0) {
    CHECK_INPUT(w);
  } else {
    CHECK_INPUT(w1);
    CHECK_INPUT(w2);
  }
  if (u1.size(0) == 0) {
    CHECK_INPUT(u);
  } else {
    CHECK_INPUT(u1);
    CHECK_INPUT(u2);
  }
  CHECK_INPUT(bias_z);
  CHECK_INPUT(bias_h_prime);
  CHECK_INPUT(initial_h);
  CHECK_INPUT(zeta);
  CHECK_INPUT(nu);
  return fastgrnn_unroll_cuda_forward(input, w, u, bias_z, bias_h_prime, zeta, nu, initial_h, z_non_linearity, w1, w2, u1, u2);
}

std::vector<torch::Tensor> fastgrnn_unroll_backward(
  torch::Tensor grad_h,
  torch::Tensor input,
  torch::Tensor hidden_states,
  torch::Tensor zeta,
  torch::Tensor nu,
  torch::Tensor w,
  torch::Tensor u,
  torch::Tensor z,
  torch::Tensor h_prime,
  torch::Tensor initial_h,
  torch::Tensor w1,
  torch::Tensor w2,
  torch::Tensor u1,
  torch::Tensor u2,
  int z_non_linearity) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(input);
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(z);
  CHECK_INPUT(h_prime);
  if(w1.size(0) == 0) {
    CHECK_INPUT(w);
  } else {
    CHECK_INPUT(w1);
    CHECK_INPUT(w2);
  }
  if (u1.size(0) == 0) {
    CHECK_INPUT(u);
  } else {
    CHECK_INPUT(u1);
    CHECK_INPUT(u2);
  }
  CHECK_INPUT(zeta);
  CHECK_INPUT(nu);
  CHECK_INPUT(initial_h);

  return fastgrnn_unroll_cuda_backward(
    grad_h,
    input,
    hidden_states,
    zeta,
    nu,
    w,
    u,
    z,
    h_prime,
    initial_h,
    z_non_linearity,
    w1, w2, u1, u2);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fastgrnn_forward, "FastGRNN forward (CUDA)");
  m.def("backward", &fastgrnn_backward, "FastGRNN backward (CUDA)");
  m.def("forward_unroll", &fastgrnn_unroll_forward, "FastGRNN Unrolled forward (CUDA)");
  m.def("backward_unroll", &fastgrnn_unroll_backward, "FastGRNN Unrolled backward (CUDA)");
}
