"""Preserve official V4.1 image pixels; budget alignment padding separately."""
import hashlib
from pathlib import Path
import threading

UPSTREAM_SHA256 = '4ce1a019d74b1933706432c8aa1c347be78882019f25a956599665cf6d105d63'
SCHEDULER_SHA256 = 'e5e1c18b1d7a6ea73adbb4921f64a35b00a96abe673691bf6a7f57281a524519'
ORIGINAL_ENCODER_SCHEDULE = None
_registered = False
_lock = threading.Lock()


def source_safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    from vllm.models.deepseek_v4_1.common import mm_preprocess as native
    # Exact official image_processor.safe_resize math. Do not subtract the
    # leading alignment pad from the image's own trained token/pixel budget.
    n_llm_h, n_llm_w = native.llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if native.num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = native.solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = native.llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert native.num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def maximum_profile_image_size(info):
    from vllm.multimodal.parse import ImageSize
    config = info.get_hf_config()
    budget, patch, down = config.vision_max_n_token, config.vision_patch_size, config.vision_downsample_ratio
    if (any(type(value) is not int or value <= 0 for value in (budget, patch, down))
            or budget < 4 or config.vision_max_wh_ratio is not None):
        raise ValueError('Maximum-patch profiling requires the pinned uncapped-aspect V4.1 image contract')
    # H*(W+1)+2 <= budget implies H*W <= budget-3. Each LLM cell
    # contains at most down**2 ViT patches. H=1,W=budget-3 attains both
    # bounds; a square does not, because every extra row costs a newline.
    cell = patch * down
    width, height = (budget - 3) * cell, cell
    if width * height < config.vision_min_pixels:
        raise ValueError('Profiling shape must not trigger the source minimum-pixel upscaler')
    return ImageSize(width=width, height=height)


def register():
    global _registered, ORIGINAL_ENCODER_SCHEDULE
    from vllm.models.deepseek_v4_1.common import mm_preprocess as native
    from vllm.v1.core.sched import scheduler
    with _lock:
        if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != UPSTREAM_SHA256:
            raise RuntimeError('Official image preprocessing has not been reviewed for this runtime revision')
        if native.COMPRESS_PAD_TO != 2:
            raise RuntimeError('Unexpected image compressor alignment')
        if hashlib.sha256(Path(scheduler.__file__).read_bytes()).hexdigest() != SCHEDULER_SHA256:
            raise RuntimeError('Atomic V4.1 image scheduling has not been reviewed for this runtime revision')
        if _registered:
            if (native.safe_resize is not source_safe_resize
                    or scheduler.Scheduler._try_schedule_encoder_inputs is not schedule_whole_images
                    or native.DeepseekV4VLProcessingInfo.get_image_size_with_most_features is not maximum_profile_image_size):
                raise RuntimeError('Registered source image resize/scheduling/profiling hook changed')
        else:
            if native.safe_resize.__module__ != native.__name__ or native.safe_resize.__name__ != 'safe_resize':
                raise RuntimeError('Unexpected existing image resize hook')
            original = scheduler.Scheduler._try_schedule_encoder_inputs
            if original.__module__ != scheduler.__name__ or original.__name__ != '_try_schedule_encoder_inputs':
                raise RuntimeError('Unexpected existing image scheduling hook')
            profile_size = native.DeepseekV4VLProcessingInfo.get_image_size_with_most_features
            if profile_size.__module__ != native.__name__ or profile_size.__name__ != 'get_image_size_with_most_features':
                raise RuntimeError('Unexpected existing image profiling-size hook')
            ORIGINAL_ENCODER_SCHEDULE = original
            native.safe_resize = source_safe_resize
            scheduler.Scheduler._try_schedule_encoder_inputs = schedule_whole_images
            native.DeepseekV4VLProcessingInfo.get_image_size_with_most_features = maximum_profile_image_size
            _registered = True


def schedule_whole_images(scheduler, request, num_computed_tokens, num_new_tokens,
                          encoder_compute_budget, shift_computed_tokens=0):
    if num_new_tokens == 0:
        return ORIGINAL_ENCODER_SCHEDULE(scheduler, request, num_computed_tokens, 0,
                                         encoder_compute_budget, shift_computed_tokens)
    config = scheduler.vllm_config.model_config.hf_config
    if config.model_type == 'deepseek_v41' and getattr(config, 'vision_n_layers', 0):
        validate_config(scheduler.vllm_config)
        if shift_computed_tokens:
            raise ValueError('Initial native image scheduling requires zero speculative token shift')
        # The native method checks its encoder-cache hits/duplicate IDs before
        # its no-chunk guard. Those cache only ViT outputs, not the LLM's
        # bidirectional image attention; cap every image span before that path.
        stop = num_computed_tokens + num_new_tokens
        for feature in request.mm_features or ():
            if feature.modality != 'image':
                continue
            start = feature.mm_position.offset
            end = start + feature.mm_position.length
            if start < stop < end:
                if num_computed_tokens >= start:
                    raise ValueError('Step budget cannot cover the remaining native image span')
                num_new_tokens = start - num_computed_tokens
                stop = start
    return ORIGINAL_ENCODER_SCHEDULE(scheduler, request, num_computed_tokens, num_new_tokens,
                                     encoder_compute_budget, shift_computed_tokens)


def validate_config(current):
    config = current.model_config.hf_config
    if not getattr(config, 'vision_n_layers', 0):
        return
    scheduler = current.scheduler_config
    maximum_span = config.vision_max_n_token + 1  # V4.1 has at most one alignment pad.
    if (not scheduler.disable_chunked_mm_input or scheduler.max_num_batched_tokens < maximum_span
            or scheduler.max_num_seqs != 1
            or 0 < scheduler.long_prefill_token_threshold < maximum_span
            or (scheduler.max_num_scheduled_tokens is not None and scheduler.max_num_scheduled_tokens < maximum_span)):
        raise ValueError('Initial DS41 vision serving requires unsplit image spans, one request, '
                         f'and a prefill budget/threshold covering at least{maximum_span} tokens; '
                         'use serving/conservative.yaml')
    if current.speculative_config is not None:
        raise ValueError('Initial native vision serving has no validated speculative decoding path')
