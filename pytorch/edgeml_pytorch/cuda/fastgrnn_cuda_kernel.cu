#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__global__ void fastgrnn_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pre_comp,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> old_h,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_h,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z_t,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> h_prime_t,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> bias_z,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> bias_h_prime,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> zeta,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> nu) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < pre_comp.size(1)){
    z_t[n][c] = sigmoid(pre_comp[n][c] + bias_z[n][c]);
    h_prime_t[n][c] = tanh(pre_comp[n][c] + bias_h_prime[n][c]);
    
    new_h[n][c] = (sigmoid(zeta[0][0]) * (1 - z_t[n][c]) + sigmoid(nu[0][0])) * h_prime_t[n][c] + z_t[n][c] * old_h[n][c];
  }
}

template <typename scalar_t>
__global__ void fastgrnn_cuda_backward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_zeta,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_nu,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_precomp,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_bias_z,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_bias_h_prime_t,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_old_h,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_h,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> old_h,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z_t,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> h_prime_t,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pre_comp,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> bias_z,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> bias_h_prime,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> zeta,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> nu) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < d_precomp.size(1)){
    auto temp_grad = grad_h[n][c] * h_prime_t[n][c];
    d_zeta[0][0] = temp_grad * (1 - z_t[n][c]) * d_sigmoid(zeta[0][0]);
    d_nu[0][0] = temp_grad * d_sigmoid(nu[0][0]);
    d_bias_z[n][c] = grad_h[n][c] * (sigmoid(zeta[0][0]) * -1 * h_prime_t[n][c] + old_h[n][c]) * d_sigmoid(pre_comp[n][c] + bias_z[n][c]);;
    d_bias_h_prime_t[n][c] = grad_h[n][c] * (sigmoid(zeta[0][0]) * (1 - z_t[n][c]) + sigmoid(nu[0][0])) * d_tanh(pre_comp[n][c] + bias_h_prime[n][c]);
    d_old_h[n][c] = grad_h[n][c] * z_t[n][c];
    d_precomp[n][c] = d_bias_z[n][c] + d_bias_h_prime_t[n][c];
  }
}
} // namespace

std::vector<torch::Tensor> fastgrnn_cuda_forward(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor bias_z,
    torch::Tensor bias_h_prime,
    torch::Tensor old_h,
    torch::Tensor zeta,
    torch::Tensor nu) {
  auto w_comp = torch::mm(input, w);
  auto u_comp = torch::mm(old_h, u);
  auto pre_comp = torch::add(u_comp, w_comp);

  const auto batch_size = old_h.size(0);
  const auto state_size = old_h.size(1);

  auto new_h = torch::zeros_like(old_h);
  auto z_t = torch::zeros_like(old_h);
  auto h_prime_t = torch::zeros_like(old_h);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(pre_comp.type(), "fastgrnn_forward_cuda", ([&] {
    fastgrnn_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        pre_comp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        old_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        new_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        z_t.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        h_prime_t.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        bias_z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        bias_h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  }));

  return {new_h, z_t, h_prime_t, pre_comp};
}

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
    torch::Tensor nu) {
  auto d_precomp = torch::zeros_like(pre_comp);
  auto d_old_h = torch::zeros_like(old_h);
  auto d_zeta = torch::zeros_like(zeta);
  auto d_nu = torch::zeros_like(nu);
  auto d_bias_z = torch::zeros_like(bias_z);
  auto d_bias_h_prime = torch::zeros_like(bias_h_prime);

  const auto batch_size = old_h.size(0);
  const auto state_size = old_h.size(1);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(pre_comp.type(), "fastgrnn_forward_cuda", ([&] {
    fastgrnn_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_precomp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_bias_z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_bias_h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_old_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        grad_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        old_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        z_t.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        h_prime_t.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        pre_comp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        bias_z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        bias_h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  }));

  d_old_h = torch::add(d_old_h, torch::mm(torch::add(d_bias_h_prime, d_bias_z), u.transpose(0, 1)));
  auto d_input = torch::mm(d_precomp, w.transpose(0, 1));
  auto d_w = torch::mm(input.transpose(0, 1), d_precomp);  
  auto d_u = torch::mm(old_h.transpose(0, 1), d_precomp);

  return {d_old_h, d_input, d_w, d_u, d_bias_z, d_bias_h_prime, d_nu, d_zeta};
}
