from pathlib import Path
import sys
import unittest
from test_moe_fused_gateup import parents

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'release/experimental/moe_critical_path'))
import dequant_pipeline


class DequantPipeline(unittest.TestCase):
    def test_isolated_pinned_parent(self):
        wrapper,kernel=parents();w,k,q=dequant_pipeline.transform(wrapper,kernel)
        self.assertIn(b'return 301;',w)
        self.assertNotIn(b'persistent_kernel',k)
        self.assertNotIn(b'fused_a_kernel',k)
        self.assertIn(b'if constexpr (REG && bits == 3)',k)
        with self.assertRaises(ValueError):dequant_pipeline.transform(wrapper,kernel+b'\n')

    def test_fold_and_accumulation_preserved(self):
        wrapper,kernel=parents();w,k,q=dequant_pipeline.transform(wrapper,kernel)
        self.assertIn(b'PF == 4 && FOLD == 4',q)
        self.assertIn(b'(d + 1) % FOLD == 0 || i + 1 == myn',q)
        self.assertIn(b'acc0[t][f].x += __low2float(ch[t][f][0]);',q)
        self.assertIn(b'acc0[t][f].y += __high2float(ch[t][f][0]);',q)
        # Epilogue and all A/B kernel arithmetic remains byte-identical;
        # only names change for trace evidence that this variant executed.
        tail=b'    // Cross-warp reduction over the k splits.'
        restore=k.replace(b'exl3_moe_coop_dq_a_kernel',b'exl3_moe_coop_a_kernel').replace(b'exl3_moe_coop_dq_b_kernel',b'exl3_moe_coop_b_kernel')
        self.assertEqual(kernel[kernel.index(tail):],restore[restore.index(tail):])

    def test_ring_and_decoded_bank_schedule(self):
        for length in range(0,81):
            ring=[i if i<length else None for i in range(4)]
            decoded=[0 if length else None,None];observed=[]
            for base in range(0,length,4):
                for d in range(4):
                    i=base+d
                    if i>=length:break
                    if i+4<length:ring[d]=i+4
                    if i+1<length:decoded[(d+1)%2]=ring[(d+1)%4]
                    observed.append(decoded[d%2])
            self.assertEqual(observed,list(range(length)))


if __name__=='__main__':unittest.main()
