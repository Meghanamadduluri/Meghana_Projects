"""Portfolio-level heuristics for retrieval confidence (no extra model calls)."""


def retrieval_confidence_percent(sources: list[dict]) -> tuple[float, str]:
    """
    Map the strongest match (minimum distance from Chroma) to a 0–100 score and a label.

    Chroma returns distance (lower = more similar). This is a rough UI heuristic, not a
    calibrated probability.
    """
    if not sources:
        return 0.0, "none"
    distances = [s.get("score") for s in sources if s.get("score") is not None]
    if not distances:
        return 50.0, "unknown"
    best = min(distances)
    pct = max(0.0, min(100.0, 100.0 / (1.0 + float(best))))
    if pct >= 55:
        label = "high"
    elif pct >= 30:
        label = "medium"
    else:
        label = "low"
    return round(pct, 1), label
