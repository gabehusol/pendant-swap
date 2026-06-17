"""pendant-swap CLI -- thin adapter over the core engine.

All business logic lives in pendant_swap/*.  This file only parses args,
calls the engine, and writes output files.

API key for the generate command:
  - Pass --api-key on the command line, OR
  - Set GEMINI_API_KEY in your environment / .env file.
The key is never stored, logged, or committed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="pendant-swap",
    help="Pendant swap toolkit -- composite, generate, QA, and finish.",
    no_args_is_help=True,
)

# Default calibration values (DESIGN.md section 11)
_REF_PX = 130
_REF_MM = 13.0
_TARGET = 21.0
_TOL    = 28


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------

@app.command()
def prep(
    model:     Path  = typer.Option(..., "--model",   "-m", help="Model photo path."),
    pendant:   Path  = typer.Option(..., "--pendant", "-p", help="Pendant product photo path."),
    target_mm: float = typer.Option(_TARGET, "--target-mm",  help="Target pendant height (mm)."),
    ref_px:    int   = typer.Option(_REF_PX, "--ref-px",     help="Reference pendant pixel height."),
    ref_mm:    float = typer.Option(_REF_MM, "--ref-mm",     help="Reference pendant real height (mm)."),
    hang_x:    Optional[int] = typer.Option(None, "--x",     help="Hang point x (default: image centre)."),
    hang_y:    Optional[int] = typer.Option(None, "--y",     help="Hang point y (default: 60pct of height)."),
    tolerance: int   = typer.Option(_TOL,    "--tolerance",  help="Background-removal tolerance."),
    out:       Path  = typer.Option(Path("output"), "--out", help="Output directory."),
) -> None:
    """Remove background, compute scale, and save a placement guide overlay."""
    from PIL import Image as PILImage
    from pendant_swap.cutout import remove_background, trim_to_alpha
    from pendant_swap.guide import make_guide
    from pendant_swap.scale import pixels_per_mm, target_pixels
    from pendant_swap.types import Point

    out.mkdir(parents=True, exist_ok=True)

    model_img   = PILImage.open(model).convert("RGB")
    raw_pendant = PILImage.open(pendant).convert("RGB")

    cutout = remove_background(raw_pendant, tolerance=tolerance)
    cutout = trim_to_alpha(cutout)
    cutout.save(out / "cutout.png")
    typer.echo("Saved cutout -> %s" % (out / "cutout.png"))

    ppm       = pixels_per_mm(ref_px, ref_mm)
    target_px = int(round(target_pixels(target_mm, ppm)))
    typer.echo("Scale: %.2f px/mm  target %d px" % (ppm, target_px))

    w, h  = model_img.size
    hx    = hang_x if hang_x is not None else w // 2
    hy    = hang_y if hang_y is not None else int(h * 0.60)
    guide = make_guide(model_img, cutout, target_px_height=target_px, hang_xy=Point(hx, hy))
    guide.save(out / "guide.jpg", quality=92)
    typer.echo("Saved guide -> %s  (hang_xy=(%d,%d))" % (out / "guide.jpg", hx, hy))


# ---------------------------------------------------------------------------
# composite
# ---------------------------------------------------------------------------

@app.command()
def composite(
    model:     Path  = typer.Option(..., "--model",   "-m"),
    pendant:   Path  = typer.Option(..., "--pendant", "-p"),
    target_mm: float = typer.Option(_TARGET, "--target-mm"),
    ref_px:    int   = typer.Option(_REF_PX, "--ref-px"),
    ref_mm:    float = typer.Option(_REF_MM, "--ref-mm"),
    hang_x:    Optional[int] = typer.Option(None, "--x"),
    hang_y:    Optional[int] = typer.Option(None, "--y"),
    rotate:    float = typer.Option(0.0, "--rotate",   help="Clockwise rotation (degrees)."),
    top_crop:  int   = typer.Option(0,   "--top-crop", help="Pixels to crop from top of pendant."),
    tolerance: int   = typer.Option(_TOL, "--tolerance"),
    out:       Path  = typer.Option(Path("output"), "--out"),
) -> None:
    """Real-pixel composite -- places actual pendant pixels. No API key needed."""
    from PIL import Image as PILImage
    from pendant_swap.composite import composite_pendant
    from pendant_swap.cutout import remove_background, trim_to_alpha
    from pendant_swap.scale import pixels_per_mm, target_pixels
    from pendant_swap.types import Point

    out.mkdir(parents=True, exist_ok=True)

    model_img   = PILImage.open(model).convert("RGB")
    raw_pendant = PILImage.open(pendant).convert("RGB")
    cutout      = trim_to_alpha(remove_background(raw_pendant, tolerance=tolerance))

    ppm       = pixels_per_mm(ref_px, ref_mm)
    target_px = int(round(target_pixels(target_mm, ppm)))

    w, h = model_img.size
    hx   = hang_x if hang_x is not None else w // 2
    hy   = hang_y if hang_y is not None else int(h * 0.60)

    result = composite_pendant(
        model_img, cutout,
        scale_width_px=target_px,
        hang_xy=Point(hx, hy),
        rotate_deg=rotate,
        top_crop_px=top_crop,
    )
    out_path = out / "composite.jpg"
    result.save(out_path, quality=92)
    typer.echo("Saved composite -> %s" % out_path)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

@app.command()
def generate(
    model:       Path  = typer.Option(..., "--model",   "-m"),
    pendant:     Path  = typer.Option(..., "--pendant", "-p"),
    target_mm:   float = typer.Option(_TARGET, "--target-mm"),
    ref_px:      int   = typer.Option(_REF_PX, "--ref-px"),
    ref_mm:      float = typer.Option(_REF_MM, "--ref-mm"),
    hang_x:      Optional[int] = typer.Option(None, "--x"),
    hang_y:      Optional[int] = typer.Option(None, "--y"),
    rotate:      float = typer.Option(0.0,  "--rotate"),
    top_crop:    int   = typer.Option(0,    "--top-crop"),
    tolerance:   int   = typer.Option(_TOL, "--tolerance"),
    max_retries: int   = typer.Option(4,    "--max-retries", help="Max generate->QA retry attempts."),
    model_id:    str   = typer.Option("gemini-3.1-flash-image", "--model-id",
                             help="Gemini model ID (e.g. gemini-3.1-flash-image or gemini-3-pro-image)."),
    api_key:     Optional[str] = typer.Option(None, "--api-key", help="Gemini API key (or set GEMINI_API_KEY)."),
    out:         Path  = typer.Option(Path("output"), "--out"),
) -> None:
    """AI generate -> measure -> retry loop. Requires a Gemini API key."""
    from pendant_swap.loop import run_swap
    from pendant_swap.types import SwapParams

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        typer.echo("Error: no API key. Pass --api-key or set GEMINI_API_KEY.", err=True)
        raise typer.Exit(1)

    out.mkdir(parents=True, exist_ok=True)

    w_model, h_model = _image_size(model)
    hx = hang_x if hang_x is not None else w_model // 2
    hy = hang_y if hang_y is not None else int(h_model * 0.60)

    params = SwapParams(
        model_path=str(model),
        pendant_path=str(pendant),
        target_mm=target_mm,
        ref_px_height=ref_px,
        ref_mm=ref_mm,
        hang_x=hx,
        hang_y=hy,
        rotate_deg=rotate,
        top_crop_px=top_crop,
        tolerance=tolerance,
        max_retries=max_retries,
        mode="generate",
        api_key=key,
        model_id=model_id,
    )

    typer.echo("Running generate loop (max %d attempts)..." % (max_retries + 1))
    try:
        result = run_swap(params)
    except RuntimeError as exc:
        typer.echo("Error: %s" % exc, err=True)
        raise typer.Exit(1)

    result.final_image.save(out / "result.jpg", quality=92)
    typer.echo("Saved result -> %s  (attempt %d)" % (
        out / "result.jpg", result.chosen_attempt + 1))

    for i, report in enumerate(result.qa_reports):
        typer.echo("\n-- Attempt %d QA --\n%s" % (i + 1, report.summary))


# ---------------------------------------------------------------------------
# qa
# ---------------------------------------------------------------------------

@app.command()
def qa(
    result_img: Path  = typer.Option(..., "--result",    help="Result image to measure."),
    target_mm:  float = typer.Option(_TARGET, "--target-mm"),
    ref_px:     int   = typer.Option(_REF_PX, "--ref-px"),
    ref_mm:     float = typer.Option(_REF_MM, "--ref-mm"),
    search:     str   = typer.Option(..., "--search",    help="Search bbox: 'X0 Y0 X1 Y1'."),
    annotate:   bool  = typer.Option(False, "--annotate", help="Save annotated copy."),
    out:        Path  = typer.Option(Path("output"), "--out"),
) -> None:
    """Measure a result image and print a pass/fail QA report."""
    from PIL import Image as PILImage
    from pendant_swap.qa import qa_report as _qa_report
    from pendant_swap.scale import pixels_per_mm

    coords = [int(v) for v in search.split()]
    if len(coords) != 4:
        typer.echo("Error: --search expects 'X0 Y0 X1 Y1'", err=True)
        raise typer.Exit(1)

    img    = PILImage.open(result_img).convert("RGB")
    ppm    = pixels_per_mm(ref_px, ref_mm)
    report = _qa_report(img, target_mm=target_mm, ppm=ppm,
                        search_bbox=tuple(coords), annotate=annotate)

    typer.echo(report.summary)

    if annotate and report.annotated_image:
        out.mkdir(parents=True, exist_ok=True)
        ann_path = out / ("qa_" + Path(result_img).stem + ".jpg")
        report.annotated_image.save(ann_path, quality=92)
        typer.echo("Annotated -> %s" % ann_path)


# ---------------------------------------------------------------------------
# finish
# ---------------------------------------------------------------------------

@app.command()
def finish(
    image:           Path  = typer.Option(..., "--image", "-i"),
    remove_wm:       Optional[str]   = typer.Option(None, "--remove-watermark",
                         help="Watermark bbox: 'X0 Y0 X1 Y1'."),
    method:          str             = typer.Option("inpaint", "--method",
                         help="Watermark method: inpaint | corner_patch."),
    crop:            Optional[str]   = typer.Option(None, "--crop",
                         help="Crop margins: 'TOP RIGHT BOTTOM LEFT'."),
    upscale_factor:  Optional[float] = typer.Option(None, "--upscale",
                         help="Upscale factor (e.g. 2.0)."),
    out:             Path            = typer.Option(Path("output"), "--out"),
) -> None:
    """Watermark removal, crop, and Lanczos upscale."""
    from PIL import Image as PILImage
    from pendant_swap.finish import crop_margins, remove_watermark, upscale

    out.mkdir(parents=True, exist_ok=True)
    img = PILImage.open(image).convert("RGB")

    if remove_wm:
        coords = [int(v) for v in remove_wm.split()]
        if len(coords) != 4:
            typer.echo("Error: --remove-watermark expects 'X0 Y0 X1 Y1'", err=True)
            raise typer.Exit(1)
        img = remove_watermark(img, tuple(coords), method=method)
        typer.echo("Watermark removed (%s)." % method)

    if crop:
        t, r, b, l = [int(v) for v in crop.split()]
        img = crop_margins(img, top=t, right=r, bottom=b, left=l)
        typer.echo("Cropped: top=%d right=%d bottom=%d left=%d" % (t, r, b, l))

    if upscale_factor:
        img = upscale(img, upscale_factor)
        typer.echo("Upscaled %.1fx -> %s" % (upscale_factor, img.size))

    out_path = out / ("finished_" + Path(image).stem + ".jpg")
    img.save(out_path, quality=92)
    typer.echo("Saved -> %s" % out_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        return im.size


if __name__ == "__main__":
    app()
