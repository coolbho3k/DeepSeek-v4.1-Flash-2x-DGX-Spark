#!/bin/sh
# Build libfastcomm.so inside the serving image (nvcc 13.0, sm_121); verbs headers from the host.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?output dir}
mkdir -p "$out"
docker run --rm --pull=never --network=none --user=$(id -u):$(id -g) \
  -v "$here":/src:ro -v /usr/include/infiniband:/hostinc/infiniband:ro -v "$(realpath "$out")":/out \
  --entrypoint /bin/sh sha256:a5ef1cecb16259d16e49578c334b05f60dc58873eeac94cc4a29c5c246d0bcbf -c \
  'nvcc -O3 -std=c++17 -arch=sm_121 -shared -Xcompiler -fPIC -I/hostinc /src/${SRC:-fastcomm.cu} -o /out/libfastcomm.so -libverbs -lpthread && sha256sum /out/libfastcomm.so'
