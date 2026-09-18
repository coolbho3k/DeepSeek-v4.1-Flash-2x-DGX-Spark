// SPDX-License-Identifier: AGPL-3.0-only
#include <torch/extension.h>
#include <vector>

void ds41_mul1_forward(const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const at::Tensor&);
std::vector<int64_t> ds41_mul1_resources();

void ds41_staged1_forward(const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,const at::Tensor&,int64_t,int64_t);
std::vector<int64_t> ds41_staged1_resources();
void ds41_small_staged_forward(const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,const at::Tensor&,int64_t,int64_t);
std::vector<int64_t> ds41_small_staged_resources();
void ds41_grouped_staged_forward(const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,const at::Tensor&,int64_t,int64_t);
std::vector<int64_t> ds41_grouped_staged_resources();
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_staged_grouped_small",&ds41_grouped_staged_forward);
    m.def("staged_grouped_small_resources",&ds41_grouped_staged_resources);
    m.def("forward_staged_small",&ds41_small_staged_forward);
    m.def("staged_small_resources",&ds41_small_staged_resources);
    m.def("forward_staged1",&ds41_staged1_forward);
    m.def("staged1_resources",&ds41_staged1_resources);
    m.def("forward", &ds41_mul1_forward);
    m.def("resources", &ds41_mul1_resources);
    m.def("max_rows", []() { return 2048; });
    m.def("graph_capture_supported", []() { return true; });
    m.def("contract_version", []() { return 2; });
}
