"""Explicit indexer format for the fixed private full-FP4 serving candidate."""
import json

OPTION='--attention-config'
VALUE='{"indexer_kv_dtype":"mxfp4"}'


def configure_argv(arguments):
    """Return a copy with exactly one matching native AttentionConfig option."""
    args=list(arguments)
    found=False
    i=1
    while i<len(args):
        token=args[i]
        if token==OPTION or token.startswith(OPTION+'='):
            if found:
                raise ValueError('Duplicate attention configuration in fixed FP4 candidate')
            if token==OPTION:
                if i+1>=len(args):raise ValueError('Missing attention configuration')
                i+=1
                value=args[i]
            else:
                value=token[len(OPTION)+1:]
            # Only this exact field is supported by the immutable candidate.
            # Reject duplicate JSON keys too; never silently discard an override.
            pairs=json.loads(value,object_pairs_hook=lambda items:items)
            if pairs!=[('indexer_kv_dtype','mxfp4')]:
                raise ValueError('This candidate requires exactly indexer_kv_dtype=mxfp4')
            found=True
        elif (token.startswith(OPTION) or token.startswith('--indexer-kv-dtype')
              or token.startswith('--use-fp4-indexer-cache')):
            raise ValueError('Conflicting indexer-format override in fixed FP4 candidate')
        i+=1
    if not found:args.extend((OPTION,VALUE))
    return args
