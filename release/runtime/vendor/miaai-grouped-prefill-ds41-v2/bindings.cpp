// SPDX-License-Identifier: AGPL-3.0-only
#include "include/quant/exl3_fat_moe.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather", &exl3_fat_moe_gather);
    m.def("gateup", &exl3_fat_moe_gateup);
    m.def("down", &exl3_fat_moe_down);
    m.def("tile_rows_gateup", &exl3_fat_moe_tile_rows_gateup);
    m.def("tile_rows_down", &exl3_fat_moe_tile_rows_down);
    m.def("abi", &exl3_fat_moe_abi);
}
