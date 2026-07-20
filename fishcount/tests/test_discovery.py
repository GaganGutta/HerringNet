from pathlib import Path

from fishcount.batch import discover_images
from helpers import write_image


def test_finds_nested_images_sorted_and_skips_non_images(tmp_path: Path) -> None:
    write_image(tmp_path / "b.jpg")
    write_image(tmp_path / "a.PNG")  # extension matching is case-insensitive
    write_image(tmp_path / "sub" / "deep" / "c.jpeg")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    found = discover_images(tmp_path)
    names = [path.relative_to(tmp_path).as_posix() for path in found]
    assert names == ["a.PNG", "b.jpg", "sub/deep/c.jpeg"]


def test_excludes_output_dir_inside_input(tmp_path: Path) -> None:
    write_image(tmp_path / "a.jpg")
    write_image(tmp_path / "output" / "run" / "annotated" / "a.jpg")  # a previous run

    found = discover_images(tmp_path, exclude_dir=tmp_path / "output" / "run")
    assert [path.name for path in found] == ["a.jpg"]


def test_exclude_dir_outside_input_changes_nothing(tmp_path: Path) -> None:
    write_image(tmp_path / "in" / "a.jpg")

    found = discover_images(tmp_path / "in", exclude_dir=tmp_path / "elsewhere")
    assert [path.name for path in found] == ["a.jpg"]
