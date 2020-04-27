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
__device__ __forceinline__ scalar_t relu(scalar_t z) {
  return z > 0 ? z : 0;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t sig_z) {
  return (1.0 - sig_z) * sig_z;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_relu(scalar_t relu_z) {
  return (relu_z == 0)? 0: 1;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t tan_z) {
  return 1.0 - (tan_z * tan_z);
}

template <typename scalar_t, scalar_t (*non_linearity) (scalar_t)>
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
      z[n][c] = non_linearity(pre_comp[n][c] + bias_z[0][c]);
      h_prime[n][c] = tanh(pre_comp[n][c] + bias_h_prime[0][c]);
      new_h[n][c] = (zeta[0][0] * (1.0 - z[n][c]) + nu[0][0]) * h_prime[n][c] + old_h[n][c] * z[n][c];
    }
}


template <typename scalar_t, scalar_t (*d_non_linearity) (scalar_t)>
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
      d_bias_z[n][c] = (old_h[n][c] - zeta[0][0] * h_prime[n][c]) * d_non_linearity(z[n][c]) * grad_h[n][c];
      d_precomp[n][c] = d_bias_z[n][c] + d_bias_h_prime[n][c];
      d_zeta[n][c] = (1.0 - z[n][c]) * h_prime[n][c] * grad_h[n][c] * d_zeta_sigmoid[0][0];
      d_nu[n][c] = h_prime[n][c] * grad_h[n][c] * d_nu_sigmoid[0][0];
    }
}

template <typename scalar_t, scalar_t (*d_non_linearity) (scalar_t)>
__global__ void fastgrnn_unroll_cuda_backward_kernel(
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
      scalar_t temp_bias_h_prime = (zeta[0][0] * (1.0 - z[n][c]) + nu[0][0]) * d_tanh(h_prime[n][c]) * grad_h[n][c];
      scalar_t temp_bias_z = (old_h[n][c] - zeta[0][0] * h_prime[n][c]) * d_non_linearity(z[n][c]) * grad_h[n][c];
      d_bias_h_prime[n][c] += temp_bias_h_prime;
      d_bias_z[n][c] += temp_bias_z;
      d_precomp[n][c] = temp_bias_z + temp_bias_h_prime;
      d_zeta[n][c] += (1.0 - z[n][c]) * h_prime[n][c] * grad_h[n][c] * d_zeta_sigmoid[0][0];
      d_nu[n][c] += h_prime[n][c] * grad_h[n][c] * d_nu_sigmoid[0][0];
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
    torch::Tensor old_h,
    int z_non_linearity,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor u1,
    torch::Tensor u2) {

  bool w_low_rank = w1.size(0) != 0;
  bool u_low_rank = u1.size(0) != 0;
  if (w_low_rank){
    w = torch::mm(w2, w1);
  }
  if (u_low_rank){
    u = torch::mm(u2, u1);
  }

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
  if (z_non_linearity == 0) {
    AT_DISPATCH_FLOATING_TYPES(pre_comp.type(), "fastgrnn_forward_cuda", ([&] {
      fastgrnn_cuda_forward_kernel<scalar_t, sigmoid><<<blocks, threads>>>(
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
  } else if (z_non_linearity == 1) {
    AT_DISPATCH_FLOATING_TYPES(pre_comp.type(), "fastgrnn_forward_cuda", ([&] {
      fastgrnn_cuda_forward_kernel<scalar_t, relu><<<blocks, threads>>>(
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
  } else if (z_non_linearity == 2) {
    AT_DISPATCH_FLOATING_TYPES(pre_comp.type(), "fastgrnn_forward_cuda", ([&] {
      fastgrnn_cuda_forward_kernel<scalar_t, tanh><<<blocks, threads>>>(
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
  }
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
  int z_non_linearity,
  torch::Tensor z,
  torch::Tensor h_prime,
  torch::Tensor w1,
  torch::Tensor w2,
  torch::Tensor u1,
  torch::Tensor u2) {
    auto d_precomp = torch::zeros_like(old_h);
    auto d_bias_z = torch::zeros_like(old_h);
    auto d_bias_h_prime = torch::zeros_like(old_h);
    auto d_nu = torch::zeros_like(old_h);
    auto d_zeta = torch::zeros_like(old_h);
    auto d_old_h = torch::zeros_like(old_h);
    auto d_w1 = torch::empty(0);
    auto d_w2 = torch::empty(0);
    auto d_u1 = torch::empty(0);
    auto d_u2 = torch::empty(0);

    bool w_low_rank = w1.size(0) != 0;
    bool u_low_rank = u1.size(0) != 0;
    if(w_low_rank) {
      w = torch::mm(w2, w1);
    }
    if (u_low_rank) {
      u = torch::mm(u2, u1);
    }
    zeta = torch::sigmoid(zeta);
    nu = torch::sigmoid(nu);
    auto d_nu_sigmoid = d_sigmoid(nu);
    auto d_zeta_sigmoid = d_sigmoid(zeta);
    const auto batch_size = old_h.size(0);
    const auto state_size = old_h.size(1);
    
    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);
    if (z_non_linearity == 0) {
      AT_DISPATCH_FLOATING_TYPES(old_h.type(), "fastgrnn_backward_cuda", ([&] {
      fastgrnn_cuda_backward_kernel<scalar_t, d_sigmoid><<<blocks, threads>>>(
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
    } else if (z_non_linearity == 1) {
      AT_DISPATCH_FLOATING_TYPES(old_h.type(), "fastgrnn_backward_cuda", ([&] {
      fastgrnn_cuda_backward_kernel<scalar_t, d_relu><<<blocks, threads>>>(
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
    } else if (z_non_linearity == 2) {
      AT_DISPATCH_FLOATING_TYPES(old_h.type(), "fastgrnn_backward_cuda", ([&] {
      fastgrnn_cuda_backward_kernel<scalar_t, d_tanh><<<blocks, threads>>>(
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
    }

    d_old_h = torch::addmm(d_old_h, d_precomp, u);
    auto d_input = torch::mm(d_precomp, w);  
    auto d_w = torch::mm(d_precomp.transpose(0, 1), input);
    auto d_u = torch::mm(d_precomp.transpose(0, 1), old_h);
    d_bias_z = d_bias_z.sum(0, true);
    d_bias_h_prime = d_bias_h_prime.sum(0, true);
    d_zeta = (d_zeta.sum(0, true)).sum(1, true);
    d_nu = (d_nu.sum(0, true)).sum(1, true);
    if (w_low_rank) {
      d_w1 = torch::mm(w2.transpose(0, 1), d_w);
      d_w2 = torch::mm(d_w, w1.transpose(0, 1));
      d_w = torch::empty(0);
    }
    if(u_low_rank) {
      d_u1 = torch::mm(u2.transpose(0, 1), d_u);
      d_u2 = torch::mm(d_u, u1.transpose(0, 1));
      d_u = torch::empty(0);
    }
    return {d_input, d_bias_z, d_bias_h_prime, d_zeta, d_nu, d_old_h, d_w, d_u, d_w1, d_w2, d_u1, d_u2};
}

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
  torch::Tensor u2) {
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device().type());
    const auto timesteps = input.size(0);
    const auto batch_size = initial_h.size(0);
    const auto state_size = initial_h.size(1);

    auto hidden_states = torch::zeros({timesteps, batch_size, state_size}, options);
    auto z_s = torch::zeros_like(hidden_states);
    auto h_prime_s = torch::zeros_like(hidden_states);

    auto prev_h = initial_h;
    auto new_h = torch::zeros_like(prev_h);
    auto z = torch::zeros_like(prev_h);
    auto h_prime = torch::zeros_like(prev_h);
    auto pre_comp = torch::zeros_like(prev_h);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);
    bool w_low_rank = w1.size(0) != 0;
    bool u_low_rank = u1.size(0) != 0;
    if (w_low_rank){
      w = torch::mm(w1.transpose(0, 1), w2.transpose(0, 1));
    }  else {
      w = w.transpose(0, 1);
    }
    if (u_low_rank){
      u = torch::mm(u1.transpose(0, 1), u2.transpose(0, 1));
    } else {
      u = u.transpose(0, 1);
    }

    zeta = torch::sigmoid(zeta);
    nu = torch::sigmoid(nu);

    for (int t=0; t < timesteps; t++) {
      pre_comp = torch::addmm(torch::mm(input[t], w), prev_h, u);
      
      if (z_non_linearity == 0) 
        AT_DISPATCH_FLOATING_TYPES(pre_comp.type(), "fastgrnn_forward_cuda", ([&] {
          fastgrnn_cuda_forward_kernel<scalar_t, sigmoid><<<blocks, threads>>>(
            new_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pre_comp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            bias_z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            bias_h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            prev_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
        }));
      else if(z_non_linearity == 1)
        AT_DISPATCH_FLOATING_TYPES(pre_comp.type(), "fastgrnn_forward_cuda", ([&] {
          fastgrnn_cuda_forward_kernel<scalar_t, relu><<<blocks, threads>>>(
            new_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pre_comp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            bias_z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            bias_h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            prev_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
        }));
      else if (z_non_linearity == 2)
        AT_DISPATCH_FLOATING_TYPES(pre_comp.type(), "fastgrnn_forward_cuda", ([&] {
          fastgrnn_cuda_forward_kernel<scalar_t, tanh><<<blocks, threads>>>(
            new_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pre_comp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            bias_z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            bias_h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            prev_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
        }));
      hidden_states[t].copy_(new_h);
      z_s[t] = z;
      h_prime_s[t] = h_prime;
      prev_h = new_h;
    }
    return {hidden_states, z_s, h_prime_s};
}

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
  torch::Tensor u2) {
  
  auto d_input = torch::zeros_like(input);
  auto d_zeta = torch::zeros_like(initial_h);
  auto d_nu = torch::zeros_like(initial_h);
  auto d_bias_z = torch::zeros_like(initial_h);
  auto d_bias_h_prime = torch::zeros_like(initial_h);
  auto d_w1 = torch::empty(0);
  auto d_w2 = torch::empty(0);
  auto d_u1 = torch::empty(0);
  auto d_u2 = torch::empty(0);

  bool w_low_rank = w1.size(0) != 0;
  bool u_low_rank = u1.size(0) != 0;
  if(w_low_rank) {
    w = torch::mm(w2, w1);
  }
  if (u_low_rank) {
    u = torch::mm(u2, u1);
  }
  auto d_w = torch::zeros_like(w);
  auto d_u = torch::zeros_like(u);
  
  zeta = torch::sigmoid(zeta);
  nu = torch::sigmoid(nu);
  auto d_nu_sigmoid = d_sigmoid(nu);
  auto d_zeta_sigmoid = d_sigmoid(zeta);
  

  auto grad_curr_h = torch::zeros_like(initial_h);
  auto d_precomp = torch::zeros_like(initial_h);
  auto d_old_h = torch::zeros_like(initial_h);
  auto prev_h_ = hidden_states[0];
  auto z_t_ = torch::zeros_like(initial_h);
  auto h_prime_t_ = torch::zeros_like(initial_h);

  const auto batch_size = hidden_states.size(1);
  const auto state_size = hidden_states.size(2);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);
  for (auto t = hidden_states.size(0) - 1; t>=0; t--) {
    grad_curr_h = torch::add(grad_h[t], d_old_h);
    z_t_ = z[t];
    h_prime_t_ = h_prime[t];
    
    if (t == 0)
       prev_h_ = initial_h;
    else
       prev_h_ = hidden_states[t-1];

    if (z_non_linearity == 0)
      AT_DISPATCH_FLOATING_TYPES(z_t_.type(), "fastgrnn_forward_cuda", ([&] {
        fastgrnn_unroll_cuda_backward_kernel<scalar_t, d_sigmoid><<<blocks, threads>>>(
          d_precomp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_old_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_bias_z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_bias_h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          grad_curr_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          z_t_.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          h_prime_t_.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_zeta_sigmoid.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_nu_sigmoid.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          prev_h_.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
      }));
    else if (z_non_linearity == 1)
      AT_DISPATCH_FLOATING_TYPES(z_t_.type(), "fastgrnn_forward_cuda", ([&] {
        fastgrnn_unroll_cuda_backward_kernel<scalar_t, d_relu><<<blocks, threads>>>(
          d_precomp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_old_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_bias_z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_bias_h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          grad_curr_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          z_t_.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          h_prime_t_.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_zeta_sigmoid.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_nu_sigmoid.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          prev_h_.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
      }));
    else if(z_non_linearity == 2)
      AT_DISPATCH_FLOATING_TYPES(z_t_.type(), "fastgrnn_forward_cuda", ([&] {
        fastgrnn_unroll_cuda_backward_kernel<scalar_t, d_sigmoid><<<blocks, threads>>>(
          d_precomp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_old_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_bias_z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_bias_h_prime.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          grad_curr_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          z_t_.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          h_prime_t_.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          zeta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          nu.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_zeta_sigmoid.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_nu_sigmoid.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          prev_h_.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
      }));
    d_old_h = torch::addmm(d_old_h, d_precomp, u);
    d_input[t] = torch::mm(d_precomp, w);
    d_w = torch::addmm(d_w, d_precomp.transpose(0, 1), input[t]);
    d_u = torch::addmm(d_u, d_precomp.transpose(0, 1), prev_h_);
  }
  d_bias_z = d_bias_z.sum(0, true);
  d_bias_h_prime = d_bias_h_prime.sum(0, true);
  d_zeta = (d_zeta.sum(0, true)).sum(1, true);
  d_nu = (d_nu.sum(0, true)).sum(1, true);
  if (w_low_rank) {
    d_w1 = torch::mm(w2.transpose(0, 1), d_w);
    d_w2 = torch::mm(d_w, w1.transpose(0, 1));
    d_w = torch::empty(0);
  }
  if(u_low_rank) {
    d_u1 = torch::mm(u2.transpose(0, 1), d_u);
    d_u2 = torch::mm(d_u, u1.transpose(0, 1));
    d_u = torch::empty(0);
  }
  return {d_input, d_bias_z, d_bias_h_prime, d_zeta, d_nu, d_old_h, d_w, d_u, d_w1, d_w2, d_u1, d_u2};
}