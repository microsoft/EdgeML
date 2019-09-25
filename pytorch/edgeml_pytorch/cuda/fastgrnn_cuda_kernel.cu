#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__forceinline__ torch::Tensor d_sigmoid(torch::Tensor z) {
  return (1 - z) * z;
}

__forceinline__ torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.pow(2);
}


namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t sig_z) {
  return (1.0 - sig_z) * sig_z;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t tan_z) {
  return 1 - (tan_z * tan_z);
}

template <typename scalar_t>
__global__ void fastgrnn_cuda_forward_kernel(
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_h,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> h_prime,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pre_comp,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> bias_z,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> bias_h_prime,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> nu,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> zeta,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> old_h) {
  const int n = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < old_h.size(1)){
    z[n][c] = sigmoid(pre_comp[n][c] + bias_z[0][c]);
    h_prime[n][c] = tanh(pre_comp[n][c] + bias_h_prime[0][c]);
    new_h[n][c] = (zeta[0][0] * (1.0 - z[n][c]) + nu[0][0]) * h_prime[n][c] + old_h[n][c] * z[n][c];
  }
}


template <typename scalar_t>
__global__ void fastgrnn_cuda_backward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_precomp,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_old_h,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_bias_z,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_bias_h_prime,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_nu,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_zeta,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_h,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> h_prime,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> zeta,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> nu,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_zeta_sigmoid,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_nu_sigmoid,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> old_h) {
  const int n = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < old_h.size(1)){ 
    d_old_h[n][c] = z[n][c] * grad_h[n][c];
    d_bias_h_prime[n][c] = (zeta[0][0] * (1.0 - z[n][c]) + nu[0][0]) * d_tanh(h_prime[n][c]) * grad_h[n][c];
    d_bias_z[n][c] = (old_h[n][c] - zeta[0][0] * h_prime[n][c]) * d_sigmoid(z[n][c]) * grad_h[n][c];
    d_precomp[n][c] = d_bias_z[n][c] + d_bias_h_prime[n][c];
    d_zeta[n][c] = (1.0 - z[n][c]) * h_prime[n][c]*grad_h[n][c] * d_zeta_sigmoid[0][0];
    d_nu[n][c] = h_prime[n][c] * grad_h[n][c] * d_nu_sigmoid[0][0];
  }
}
} // namespace

std::vector<torch::Tensor> fastgrnn_cuda_forward(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor bias_z,
    torch::Tensor bias_h_prime,
    torch::Tensor zeta,
    torch::Tensor nu,
    torch::Tensor old_h) {
  
  auto pre_comp = torch::addmm(torch::mm(input, w.transpose(0, 1)), old_h, u.transpose(0, 1));
  nu = torch::sigmoid(nu);
  zeta = torch::sigmoid(zeta);
  const auto batch_size = old_h.size(0);
  const auto state_size = old_h.size(1);
  auto new_h = torch::zeros_like(old_h);
  auto z = torch::zeros_like(old_h);
  auto h_prime = torch::zeros_like(old_h);
  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);
  AT_DISPATCH_FLOATING_TYPES(pre_comp.type(), "fastgrnn_forward_cuda", ([&] {
    fastgrnn_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        new_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        pre_comp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        bias_z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        bias_h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        old_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  }));
  return {new_h, z, h_prime};
}

std::vector<torch::Tensor> fastgrnn_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor input,
    torch::Tensor old_h,
    torch::Tensor zeta,
    torch::Tensor nu,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor z,
    torch::Tensor h_prime) {
    auto d_precomp = torch::zeros_like(old_h);
    auto d_bias_z = torch::zeros_like(old_h);
    auto d_bias_h_prime = torch::zeros_like(old_h);
    auto d_nu = torch::zeros_like(old_h);
    auto d_zeta = torch::zeros_like(old_h);
    auto d_old_h = torch::zeros_like(old_h);
    zeta = torch::sigmoid(zeta);
    nu = torch::sigmoid(nu);
    auto d_nu_sigmoid = d_sigmoid(nu);
    auto d_zeta_sigmoid = d_sigmoid(zeta);
    const auto batch_size = old_h.size(0);
    const auto state_size = old_h.size(1);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(old_h.type(), "fastgrnn_backward_cuda", ([&] {
    fastgrnn_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_precomp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_old_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_bias_z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_bias_h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        grad_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_zeta_sigmoid.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_nu_sigmoid.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        old_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  }));

  d_old_h = torch::addmm(d_old_h, d_precomp, u);
  auto d_input = torch::mm(d_precomp, w);  
  auto d_w = torch::mm(d_precomp.transpose(0, 1), input);
  auto d_u = torch::mm(d_precomp.transpose(0, 1), old_h);
  d_bias_z = d_bias_z.sum(0, true);
  d_bias_h_prime = d_bias_h_prime.sum(0, true);
  d_zeta = (d_zeta.sum(0, true)).sum(1, true);
  d_nu = (d_nu.sum(0, true)).sum(1, true);
    
  return {d_input, d_w, d_u, d_bias_z, d_bias_h_prime, d_zeta, d_nu, d_old_h};
}
