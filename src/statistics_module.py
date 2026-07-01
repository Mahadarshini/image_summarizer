"""
FR-5 Difference Statistics
-----------------------------
Computes the quantitative statistics required by the spec.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from diff_detector import DiffResult


@dataclass
class DiffStatistics:
    num_changed_regions: int
    percent_changed: float
    total_changed_area_px: int
    total_image_area_px: int
    ssim_similarity_score: float
    regions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_statistics(result: DiffResult) -> DiffStatistics:
    """FR-5: Build the statistics object from a DiffResult."""
    height, width = result.image_shape
    total_area = height * width

    region_dicts = []
    for r in result.regions:
        region_dicts.append({
            "id": r.id,
            "bbox_xywh": [r.x, r.y, r.w, r.h],
            "area_px": r.area_px,
            "percent_of_image": round((r.area_px / total_area) * 100, 3) if total_area else 0.0,
            "location": r.location_label,
        })

    return DiffStatistics(
        num_changed_regions=len(result.regions),
        percent_changed=result.percent_changed,
        total_changed_area_px=result.total_changed_pixels,
        total_image_area_px=total_area,
        ssim_similarity_score=result.ssim_score,
        regions=region_dicts,
    )
