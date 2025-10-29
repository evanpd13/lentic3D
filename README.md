# lentic3D

Lentic3D provides a command-line pipeline that prepares lenticular-ready composites from a single photograph.

## Dependencies

Install the runtime requirements with pip:

```bash
pip install opencv-python numpy Pillow
```

`opencv-python` powers the warping and feature detection, `numpy` handles array manipulation, and `Pillow` is useful for manual inspection or further editing of the generated images.

## Usage

```bash
python lentic3d_pipeline.py \
  path/to/source.jpg \
  --output-dir outputs/my_subject \
  --knob 6.5 \
  --lenticule-width 5 \
  --metadata-json outputs/my_subject/metadata.json
```

### Arguments

* `source` – input image path.
* `--output-dir` – directory where the left/center/right frames, lenticular composite, and metadata are written (defaults to `./outputs`).
* `--knob` – angular knob in degrees. Positive values rotate to the right, negative values to the left. The pipeline internally generates left/right views using ±knob while keeping the center frame unmodified.
* `--radius` – optional override for the depth radius. When omitted the tool derives a radius from the detected subject bounding box and persists it for reuse.
* `--lenticule-width` – vertical strip width in pixels used while interlacing left/center/right views across the final canvas.
* `--metadata-json` – optional metadata path. Defaults to `<output-dir>/metadata.json` and is automatically created or updated on each run.

### Depth reuse behavior

When no radius override is provided the pipeline estimates a subject bounding box using Canny edges and morphological closing, then derives a radius from the longest box dimension. The resulting radius, knob value, lenticule width, and bounding box are stored in the metadata JSON. Future runs that point to the same metadata file will reuse the persisted radius so that subtle configuration tweaks (for example changing the knob angle) keep a consistent depth unless you explicitly pass a new `--radius` value.

## Outputs

Running the CLI saves four images alongside the metadata JSON:

* `<stem>_left.png`
* `<stem>_center.png`
* `<stem>_right.png`
* `<stem>_lenticular.png`

Each file shares the original resolution. The lenticular composite interlaces vertical strips in the order left → center → right, repeating across the canvas using the provided lenticule width.

## Smoke test

You can verify the pipeline locally using any photograph with a clearly defined subject:

1. Run the CLI with a moderate knob, for example `--knob 5 --lenticule-width 4`.
2. Open the generated `<stem>_left.png`, `<stem>_center.png`, and `<stem>_right.png` in an image viewer. You should observe the subject shifting laterally while maintaining scale.
3. Inspect `<stem>_lenticular.png` to confirm alternating strips from the three views. Zoom in to ensure the strip order repeats left/center/right.
4. Re-run the CLI without specifying `--radius` and confirm that the metadata JSON radius remains unchanged between runs.

For automated regression checks you can incorporate the CLI into a script that compares output file existence and dimensions, e.g.:

```bash
python lentic3d_pipeline.py example.jpg --output-dir smoke && \
python - <<'PY'
from pathlib import Path
from PIL import Image
base = Path('smoke')
expected = ['example_left.png', 'example_center.png', 'example_right.png', 'example_lenticular.png']
for name in expected:
    path = base / name
    assert path.exists(), f"Missing {path}"
    with Image.open(path) as img:
        img.verify()
print('Smoke test passed')
PY
```

This lightweight check verifies the CLI runs end-to-end and the images are readable without corrupting their contents.
