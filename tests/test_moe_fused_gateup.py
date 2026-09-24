from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[1]
CODE=runpy.run_path(str(ROOT/'release/experimental/moe_critical_path/fused_gateup.py'))
PARENT=ROOT/'artifacts/cooperative-six-session-build-v1'
WRAPPER=ROOT/'release/runtime/sources/cooperative24.cu'
VENDOR=ROOT/'release/runtime/vendor/miaai-cooperative-moe-agpl/extensions/cooperative_moe/native/cooperative_moe_kernel.cuh'


def parents():
    # Reproduce the retained C6 source from the public, pinned C2 foundation.
    kernel=VENDOR.read_text().replace('p.slots_max = 48; p.rows_max = 8;',
        'p.slots_max = 144; p.rows_max = 24;').replace('p.ctr_a_len = 432; p.ctr_b_len = 320;',
        'p.ctr_a_len = 1296; p.ctr_b_len = 960;')
    kernel='// Local DS41 C6 adaptation: 24 physical rows / 144 routed slots, ABI 2.\n'+kernel
    return WRAPPER.read_bytes(),kernel.encode()


class FusedGateUp(unittest.TestCase):
    def test_scope_and_parent(self):
        wrapper,kernel=parents();w,k=CODE['transform'](wrapper,kernel)
        self.assertIn(b'rows>1',w)
        self.assertIn(b'bits==3 && geometry==1',w)
        self.assertIn(b'goal50_coop_experiment() { return 101; }',w)
        self.assertEqual(k.count(b'bool LOCAL_C = false'),1)
        # Everything from the original down-kernel declaration is unchanged.
        anchor=b'template <int bits, int cb, bool WIDE>\n__global__ __launch_bounds__(THREADS)\nvoid exl3_moe_coop_b_kernel'
        self.assertEqual(kernel[kernel.index(anchor):],k[k.index(anchor):])

    def test_changed_parent_rejected(self):
        wrapper,kernel=parents()
        with self.assertRaises(ValueError):CODE['transform'](wrapper+b'\n',kernel)

    def test_parallel_variant_keeps_logical_geometry(self):
        wrapper,kernel=parents();w,k=CODE['transform'](wrapper,kernel,True)
        self.assertIn(b'goal50_coop_experiment() { return 102; }',w)
        self.assertIn(b'dim3(fuse_a?1024:MOE_COOP_THREADS)',w)
        self.assertIn(b'LOCAL_C ? threadIdx.x % THREADS : threadIdx.x',k)
        self.assertIn(b'constexpr int WK = 16;',k)
        self.assertIn(b'constexpr int FOLD = 4;',k)
        self.assertIn(b'__launch_bounds__(2 * THREADS)',CODE['fragment'](True))

    def test_local_storage_and_unchanged_fold(self):
        wrapper,kernel=parents();_,candidate=CODE['transform'](wrapper,kernel)
        self.assertIn(b'constexpr int FOLD = 4;',candidate)
        self.assertIn(b'2 * ROWS * 128 * sizeof(half)',CODE['fragment']())
        self.assertNotIn(b'atomicAdd',CODE['fragment']())


if __name__=='__main__':unittest.main()
