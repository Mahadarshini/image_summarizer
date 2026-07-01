"""
FR-6 AI-Based Change Summary
-------------------------------
Generates a concise, human-readable paragraph describing the detected
changes: overall result, major regions, approximate locations, severity,
and (optionally) a confidence indicator.

Two modes:
  1. Rule-based (default, offline, no API key required) - deterministic
     natural-language generation from the statistics.
  2. LLM-enhanced (optional) - if an ANTHROPIC_API_KEY environment variable
     is set, the statistics are handed to Claude to produce a more fluent,
     analyst-style paragraph. Falls back to rule-based mode automatically
     if no key is present or the call fails.
"""

import os
from statistics_module import DiffStatistics


def _severity_label(percent_changed: float) -> str:
    if percent_changed < 1:
        return "minimal"
    if percent_changed < 5:
        return "minor"
    if percent_changed < 15:
        return "moderate"
    if percent_changed < 35:
        return "significant"
    return "extensive"


def _confidence_label(ssim_score: float) -> str:
    # SSIM close to 1 => images are structurally very similar outside the
    # detected regions => higher confidence the detected regions are the
    # genuine changes rather than noise.
    if ssim_score > 0.97:
        return "high"
    if ssim_score > 0.90:
        return "moderate"
    return "low"


def generate_rule_based_summary(stats: DiffStatistics) -> str:
    """FR-6: Deterministic, template-driven paragraph generation."""
    n = stats.num_changed_regions
    severity = _severity_label(stats.percent_changed)
    confidence = _confidence_label(stats.ssim_similarity_score)

    if n == 0:
        return (
            "The comparison found no significant differences between the two images. "
            f"The images are structurally near-identical (SSIM similarity score: "
            f"{stats.ssim_similarity_score})."
        )

    locations = [r["location"] for r in stats.regions]
    unique_locations = sorted(set(locations), key=locations.index)
    location_phrase = ", ".join(unique_locations[:-1]) + (
        f" and {unique_locations[-1]}" if len(unique_locations) > 1 else unique_locations[0]
    )

    largest = stats.regions[0]

    opening = (
        f"The comparison identified {n} changed region{'s' if n != 1 else ''} "
        f"between the two images, affecting approximately {stats.percent_changed}% "
        f"of the total image area."
    )

    location_sentence = (
        f"The most significant change is located in the {largest['location']} region "
        f"(region #{largest['id']}), covering about {largest['percent_of_image']}% of the image."
    )

    if n > 1:
        spread_sentence = (
            f"Additional changes were detected across the {location_phrase} area"
            f"{'s' if len(unique_locations) > 1 else ''} of the frame."
        )
    else:
        spread_sentence = ""

    severity_sentence = (
        f"Overall, the modifications are classified as {severity} in extent "
        f"(structural similarity score: {stats.ssim_similarity_score})."
    )

    confidence_sentence = f"Confidence in these detections is {confidence}."

    parts = [opening, location_sentence, spread_sentence, severity_sentence, confidence_sentence]
    return " ".join(p for p in parts if p)


def generate_llm_summary(stats: DiffStatistics, model: str = "claude-sonnet-4-6") -> str:
    """
    FR-6 (enhanced): Use the Claude API to turn the statistics into a more
    fluent, analyst-style summary. Requires ANTHROPIC_API_KEY to be set.
    Raises on any failure so the caller can fall back to the rule-based path.
    """
    import anthropic  # imported lazily so the package is optional

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    prompt = f"""You are summarizing the output of an automated image-difference
detection system. Given the structured statistics below, write ONE concise
paragraph (3-5 sentences) describing: the overall comparison result, the
major changed regions and their approximate locations (top/bottom/left/
right/center), the severity/extent of change, and a confidence note. Do not
invent objects or details that aren't implied by the statistics - describe
regions generically (e.g. "a changed region") unless labels are given.

Statistics (JSON):
{stats.to_dict()}
"""

    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def generate_summary(stats: DiffStatistics, use_llm: bool = False) -> str:
    """
    FR-6: Main entry point. Tries the LLM path only if explicitly requested
    AND an API key is available; otherwise (or on any failure) uses the
    deterministic rule-based generator.
    """
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return generate_llm_summary(stats)
        except Exception as exc:  # noqa: BLE001
            fallback = generate_rule_based_summary(stats)
            return f"{fallback}\n\n[Note: LLM summary generation failed ({exc}); showing rule-based summary instead.]"
    return generate_rule_based_summary(stats)
