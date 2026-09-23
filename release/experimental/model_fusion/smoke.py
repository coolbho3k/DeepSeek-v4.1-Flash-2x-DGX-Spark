# SPDX-License-Identifier: AGPL-3.0-only
"""Check text and whole-image prefix reuse on the final, identified server."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--deployment-sha256',required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier smoke evidence')
    raw=args.deployment.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.deployment_sha256:raise ValueError('Changed deployment')
    config=json.loads(raw);kit=Path(config['nodes'][0]['kit'])
    sys.path.insert(0,str(kit/'tools'));import portable_pair as pair
    sys.path.insert(0,str(ROOT/'probes'))
    import check_serving_long_context as retrieval
    import check_serving_speed as speed
    import check_serving_prefix_reuse as prefix
    import check_serving_generation as generation
    import check_serving_image_prefix_fixed as image_prefix
    retrieval.BASE=speed.BASE=generation.BASE='http://127.0.0.1:'+str(config['api']['port'])
    run=Path(config['nodes'][0]['runs'])/config['run_id']/'pair'
    ready=json.loads((run/'health-ready.json').read_bytes())
    report=dict(status='running',run_id=config['run_id'],checks=[],broad_quality_benchmark=False)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    def inspect():
        observed=pair.results(pair.both(config,'inspect'))
        reason=pair.stop_reason(observed,ready['containers'],ready['started_at'])
        if reason:raise RuntimeError(reason)
        if any(r['memory']['MemAvailable']<2**30 for r in observed):raise RuntimeError('Request admission margin unavailable')
        return observed
    with (run/'request-probe.lock').open('a') as lock,(run/'watch.lock').open('r') as watch:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:fcntl.flock(watch,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise RuntimeError('Missing independent memory watchdog')
        report['initial_hosts']=inspect();save()
        try:
            for conversation in (False,True):
                inspect()
                out=ROOT/'reports'/f'model-fusion-final-prefix-{int(conversation)}-v1.json'
                sys.argv=['prefix','--reference',str(args.reference),'--output',str(out)]
                if conversation:sys.argv+=['--conversation']
                prefix.main()
                result=json.loads(out.read_bytes())
                report['checks'].append(dict(kind='prefix',conversation=conversation,report=str(out),status=result['status'],cached_tokens=result['cache']['cached_tokens']));save()
            inspect()
            out=ROOT/'reports/model-fusion-final-image-prefix-v1.json'
            sys.argv=['image-prefix','--output',str(out)]
            image_prefix.main();result=json.loads(out.read_bytes())
            if not all(c['exact_match'] for c in result['cases']):raise ValueError('Document smoke answer changed')
            report['checks'].append(dict(kind='image_prefix',report=str(out),status=result['status'],exact_answers=len(result['cases']),safe_completed_image_reuse=result['safe_completed_image_reuse_observed']))
            report.update(status='final_prefix_and_image_smoke_pass',final_hosts=inspect())
        except BaseException as error:report.update(status='failed',error=repr(error));raise
        finally:save()
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_hosts')}),flush=True)


if __name__=='__main__':main()
