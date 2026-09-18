# Validated native artifact

`prepare_profile.py` and the adapter pin `cooperative_moe.so` to

`a09a589cbdcecb5372991c7b091d732236d58bc5f5aea14ab91e38e426f08d78`.

A clean nvcc rebuild in the recipe image is **not** bit-identical: GNU build-id
and CUDA `-lineinfo` filename metadata change across runs. The operator path
therefore installs this validated binary rather than compiling it.

Place `cooperative_moe.so` in this directory (Git ignores `*.so` at the repo
root; this path is the exception) or download the matching GitHub Release asset
and check it with `sha256sum -c SHA256SUMS`. Do not repin a different hash
without repeating the GPU integration gate.
