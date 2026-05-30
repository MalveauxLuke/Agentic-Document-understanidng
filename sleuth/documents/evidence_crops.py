from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


LOCAL_CROP_REGIONS = {
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "upper_half",
    "lower_half",
    "left_half",
    "right_half",
}
ORIGINAL_PAGE_REGIONS = {"full_page", "uncertain", "not_applicable"}
ALLOWED_CROP_REGIONS = LOCAL_CROP_REGIONS | ORIGINAL_PAGE_REGIONS


@dataclass(frozen=True)
class EvidenceCrop:
    crop_region: str
    image_path: str
    crop_location: str
    bbox: tuple[int, int, int, int] | None
    is_generated_crop: bool


def normalize_crop_region(value: str | None) -> str:
    region = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if region in ALLOWED_CROP_REGIONS:
        return region
    return "not_applicable"


def crop_bbox_for_region(width: int, height: int, crop_region: str) -> tuple[int, int, int, int] | None:
    region = normalize_crop_region(crop_region)
    mid_x = width // 2
    mid_y = height // 2
    if region == "upper_left":
        return (0, 0, mid_x, mid_y)
    if region == "upper_right":
        return (mid_x, 0, width, mid_y)
    if region == "lower_left":
        return (0, mid_y, mid_x, height)
    if region == "lower_right":
        return (mid_x, mid_y, width, height)
    if region == "upper_half":
        return (0, 0, width, mid_y)
    if region == "lower_half":
        return (0, mid_y, width, height)
    if region == "left_half":
        return (0, 0, mid_x, height)
    if region == "right_half":
        return (mid_x, 0, width, height)
    return None


def build_evidence_crop(
    image_path: str,
    crop_region: str,
    evidence_item_index: int,
) -> EvidenceCrop:
    source_path = Path(image_path)
    region = normalize_crop_region(crop_region)
    if region in ORIGINAL_PAGE_REGIONS:
        return EvidenceCrop(
            crop_region=region,
            image_path=str(source_path),
            crop_location="full original page",
            bbox=None,
            is_generated_crop=False,
        )

    with Image.open(source_path) as image:
        width, height = image.size
        bbox = crop_bbox_for_region(width, height, region)
        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return EvidenceCrop(
                crop_region="not_applicable",
                image_path=str(source_path),
                crop_location="full original page",
                bbox=None,
                is_generated_crop=False,
            )

        crop_dir = source_path.parent / "_evidence_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"{source_path.stem}_item_{evidence_item_index:03d}_{region}{source_path.suffix}"
        if not crop_path.exists():
            image.crop(bbox).save(crop_path)

    return EvidenceCrop(
        crop_region=region,
        image_path=str(crop_path),
        crop_location=f"{region} bbox={bbox}",
        bbox=bbox,
        is_generated_crop=True,
    )
