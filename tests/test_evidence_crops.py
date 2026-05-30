from pathlib import Path

from PIL import Image

from sleuth.documents.evidence_crops import build_evidence_crop, crop_bbox_for_region


def test_crop_bbox_for_all_regions():
    assert crop_bbox_for_region(100, 80, "upper_left") == (0, 0, 50, 40)
    assert crop_bbox_for_region(100, 80, "upper_right") == (50, 0, 100, 40)
    assert crop_bbox_for_region(100, 80, "lower_left") == (0, 40, 50, 80)
    assert crop_bbox_for_region(100, 80, "lower_right") == (50, 40, 100, 80)
    assert crop_bbox_for_region(100, 80, "upper_half") == (0, 0, 100, 40)
    assert crop_bbox_for_region(100, 80, "lower_half") == (0, 40, 100, 80)
    assert crop_bbox_for_region(100, 80, "left_half") == (0, 0, 50, 80)
    assert crop_bbox_for_region(100, 80, "right_half") == (50, 0, 100, 80)
    assert crop_bbox_for_region(100, 80, "full_page") is None
    assert crop_bbox_for_region(100, 80, "uncertain") is None
    assert crop_bbox_for_region(100, 80, "not_applicable") is None


def test_build_evidence_crop_writes_generated_crop(tmp_path: Path):
    image_path = tmp_path / "page_0001.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    crop = build_evidence_crop(str(image_path), "right_half", 3)

    assert crop.is_generated_crop is True
    assert crop.bbox == (50, 0, 100, 80)
    assert crop.crop_region == "right_half"
    assert Path(crop.image_path).exists()
    assert "_evidence_crops" in crop.image_path


def test_build_evidence_crop_uses_original_page_for_non_crop_regions(tmp_path: Path):
    image_path = tmp_path / "page_0001.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    for region in ["full_page", "uncertain", "not_applicable"]:
        crop = build_evidence_crop(str(image_path), region, 0)
        assert crop.is_generated_crop is False
        assert crop.bbox is None
        assert crop.image_path == str(image_path)
