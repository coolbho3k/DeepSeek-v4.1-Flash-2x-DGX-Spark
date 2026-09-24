from pathlib import Path
import sys
import unittest

from test_moe_fused_gateup import parents

HERE=Path(__file__).resolve().parents[1]/'release/experimental/moe_critical_path'
sys.path.insert(0,str(HERE))
import persistent_pipeline


class PersistentPipeline(unittest.TestCase):
    def test_scratch_and_publication(self):
        w,k,a,q=persistent_pipeline.transform(*parents())
        self.assertIn(b'goal50_coop_experiment() { return 201; }',w)
        self.assertIn(b'persistent_grid=prop.multiProcessorCount*info[4]',w)
        self.assertIn(b'cuda::memory_order_acq_rel',q)
        self.assertIn(b'store(run + 1, cuda::memory_order_release)',q)
        self.assertIn(b'load(cuda::memory_order_acquire)',q)
        self.assertIn(b'PIPE_B_NEXT + PIPE_RUNS <= 1296',q)
        self.assertNotIn(b'grid.sync',q)
        self.assertNotIn(b'p.ctr_b[i] = 0',a)
        retry=q[q.index(b'__nanosleep(128);'):q.index(b'continue;',q.index(b'__nanosleep(128);'))]
        self.assertIn(b'__syncthreads();',retry)

    def test_down_task_preserves_arithmetic(self):
        w,k,a,q=persistent_pipeline.transform(*parents())
        original=parents()[1].decode()
        start=original.index('    int nrows = 0;',original.index('void exl3_moe_coop_b_kernel'))
        tail=original[start:original.index('\n}  // namespace goal50_fixed_coop_ns',start)]
        task=k.decode().split('void persistent_b_task',1)[1]
        self.assertIn(tail,task)
        self.assertIn(b'constexpr int FOLD = 4;',k)

    def test_layout_disjoint_and_within_original_counters(self):
        ranges=[range(0,3),range(16,160),range(160,304),range(304,448)]
        joined=[i for interval in ranges for i in interval]
        self.assertEqual(len(joined),len(set(joined)))
        self.assertLess(max(joined),1296)

    def test_two_block_bound_is_an_explicit_separate_candidate(self):
        one=persistent_pipeline.transform(*parents())
        two=persistent_pipeline.transform(*parents(),resident_blocks=2)
        self.assertEqual(one[:3],two[:3])
        self.assertIn(b'__launch_bounds__(THREADS, 2)',two[3])
        self.assertNotIn(b'__launch_bounds__(THREADS, 2)',one[3])


if __name__=='__main__':unittest.main()
