#include <torch/extension.h>
#include <vector>

void ds41_mul1_forward(const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const at::Tensor&);
std::vector<int64_t> ds41_mul1_resources();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ds41_mul1_forward);
    m.def("resources", &ds41_mul1_resources);
    m.def("contract_version", []() { return 1; });
}
