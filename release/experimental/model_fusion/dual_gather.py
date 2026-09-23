# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded fused gate/up input gather; native FP16 boundaries are unchanged."""
import ctypes as C
import functools
import hashlib
from pathlib import Path

BINARY_SHA256 = 'e8194b01e87e068d4349b1ee7821d6bad1b295ba39281cadaeb42bdbe7b5c67f'


@functools.cache
def load():
    path = Path(__file__).resolve().parents[1] / 'libds41_dual_gather.so'
    if hashlib.sha256(path.read_bytes()).hexdigest() != BINARY_SHA256:
        raise RuntimeError('Changed dual-gather binary')
    library = C.CDLL(str(path))
    library.ds41_dual_gather_abi.restype = C.c_int
    if library.ds41_dual_gather_abi() != 1:
        raise RuntimeError('Changed dual-gather ABI')
    function = library.ds41_dual_gather
    function.restype = C.c_int
    function.argtypes = [C.POINTER(C.c_void_p), C.c_int, C.c_int, C.c_void_p]
    return library, function


def forward(inputs, tokens, experts, gate_scales, up_scales, gate, up,
            active_rows, rows_bound, stream):
    # The grouped dispatcher owns and validates these buffers and the stream.
    # Bound from CPU shape metadata; the actual row count remains device-only.
    if (not 1 <= rows_bound <= len(tokens) or rows_bound > 12288
            or inputs.shape[1] != 5120 or gate.shape != up.shape
            or gate.shape[1] != 5120 or len(gate) < rows_bound):
        raise ValueError('Invalid grouped gather bounds')
    pointers = (C.c_void_p * 8)(*[tensor.data_ptr() for tensor in
        (inputs, tokens, experts, gate_scales, up_scales, gate, up, active_rows)])
    status = load()[1](pointers, rows_bound, 5120, C.c_void_p(stream.cuda_stream))
    if status:
        raise RuntimeError('Dual-gather launch failed: ' + str(status))
