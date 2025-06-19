#pragma once
#include <cuda_runtime.h>
#include <torch/torch.h>

using torch::Tensor;

namespace mc
{
    // defination
    std::vector<Tensor> marching_cubes(
        const Tensor&,
        const float,
        const std::vector<float>,
        const std::vector<float>);
    std::vector<Tensor> marching_cubes_wrapper(
        const Tensor&,
        const float,
        const float*,
        const float*);

    void save_mesh_as_ply(const std::string, Tensor, Tensor, Tensor);
} // namespace mc(marching cube)

// Utils
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x)      \
    CHECK_CUDA(x);          \
    CHECK_CONTIGUOUS(x)

#define CHECK_CPU_INPUT(x)  \
    CHECK_CPU(x);           \
    CHECK_CONTIGUOUS(x)

