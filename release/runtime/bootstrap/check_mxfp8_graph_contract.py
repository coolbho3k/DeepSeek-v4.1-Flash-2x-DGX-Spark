"""Stdlib-only input-guard tests; no FlashInfer/Torch/Docker/GPU execution."""
import copy
import sys
from types import SimpleNamespace
import generate_mxfp8_graph_cpu as graph


def main():
    good = dict(env=dict(graph.EXPECTED_ENV), available=48*graph.GIB,
                limits={'memory.max': str(4*graph.GIB), 'memory.swap.max': '0',
                        'cpu.max': '200000 100000'}, devices=[], isolated=True)
    graph.validate_environment(**good)
    cases = []
    for key in graph.EXPECTED_ENV:
        bad = copy.deepcopy(good); del bad['env'][key]; cases.append(bad)
        bad = copy.deepcopy(good); bad['env'][key] = 'unexpected'; cases.append(bad)
    for key, values in {
        'available': [48*graph.GIB-1, True, float(48*graph.GIB)],
        'devices': [['/dev/nvidia0'], ['/dev/dri/renderD128']],
        'isolated': [False],
    }.items():
        for value in values:
            bad = copy.deepcopy(good); bad[key] = value; cases.append(bad)
    for key, values in {
        'memory.max': ['max', str(4*graph.GIB+1), '0'],
        'memory.swap.max': ['max', '1'],
        'cpu.max': ['max 100000', '200001 100000', '0 100000', '100000 0'],
    }.items():
        for value in values:
            bad = copy.deepcopy(good); bad['limits'][key] = value; cases.append(bad)
    for bad in cases:
        try:
            graph.validate_environment(**bad)
        except ValueError:
            continue
        raise AssertionError('Unsafe graph-generation environment accepted')
    modules = {'flashinfer.'+p[:-3].replace('/', '.'):
               SimpleNamespace(__file__=str(graph.PACKAGE/p))
               for p in graph.SOURCE_SHA256 if p.endswith('.py')}
    graph.validate_imported_sources(modules)
    for name in modules:
        for replacement in (None, SimpleNamespace(__file__='/unexpected/module.py')):
            bad = dict(modules); bad[name] = replacement
            try:
                graph.validate_imported_sources(bad)
            except ValueError:
                continue
            raise AssertionError('Missing or shadowed generator module accepted')
    assert 'torch' not in sys.modules and 'flashinfer' not in sys.modules
    print(f'PASS: {len(cases)} unsafe environments and {2*len(modules)} missing/shadowed '
          'module bindings refused; no package imports; '
          'native graph generation and compilation NOT tested')


if __name__ == '__main__':
    main()
