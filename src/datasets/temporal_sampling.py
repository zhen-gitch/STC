import random


SUPPORTED_TEMPORAL_SAMPLING_STRATEGIES = {
    "stride_head",
    "uniform",
    "first",
    "middle",
    "random",
}


def normalize_temporal_sampling_strategy(strategy):
    strategy = str(strategy or "stride_head").strip().lower()
    aliases = {
        "default": "stride_head",
        "stride": "stride_head",
        "head": "first",
        "uniform_fixed": "uniform",
        "middle_crop": "middle",
        "random_crop": "random",
    }
    strategy = aliases.get(strategy, strategy)
    if strategy not in SUPPORTED_TEMPORAL_SAMPLING_STRATEGIES:
        supported = sorted(SUPPORTED_TEMPORAL_SAMPLING_STRATEGIES)
        raise ValueError(f"Unsupported PROCESS_TEMPORAL.SAMPLING_STRATEGY='{strategy}'. Supported values: {supported}")
    return strategy


def model_max_len(max_seq_len, sample_step):
    sample_step = max(1, int(sample_step))
    return max(1, int(max_seq_len) // sample_step)


def select_temporal_indices(
    frame_count,
    sample_step=10,
    max_seq_len=2000,
    strategy="stride_head",
    seed=None,
):
    """Return frame indices for a temporal sampling strategy.

    The default `stride_head` intentionally matches the historical dataset
    behavior: `frames[::SAMPLE_STEP][:MAX_SEQ_LEN // SAMPLE_STEP]`.
    """
    frame_count = max(0, int(frame_count))
    if frame_count <= 0:
        return []

    sample_step = max(1, int(sample_step))
    max_len = model_max_len(max_seq_len=max_seq_len, sample_step=sample_step)
    strategy = normalize_temporal_sampling_strategy(strategy)

    if strategy == "stride_head":
        return list(range(0, frame_count, sample_step))[:max_len]

    selected_count = min(frame_count, max_len)
    if strategy == "uniform":
        if selected_count <= 1:
            return [0]
        return sorted({round(idx * (frame_count - 1) / (selected_count - 1)) for idx in range(selected_count)})

    if strategy == "first":
        start = 0
    elif strategy == "middle":
        start = max(0, (frame_count - selected_count) // 2)
    elif strategy == "random":
        max_start = max(0, frame_count - selected_count)
        rng = random.Random(seed)
        start = rng.randint(0, max_start) if max_start else 0
    else:
        raise AssertionError(f"Unhandled temporal sampling strategy: {strategy}")

    return list(range(start, start + selected_count))
