## CrackSegmentation 1.5.0

### What's new in 1.5.0

- **Merge crack branches** (right sidebar, Tools menu): join nearby cracks into one crack with branches. Click the **main** crack (its line or its number), then every crack to add as one of its **branches**. A branch that stops short of the main crack is joined to it along the crack edge (or with a straight bridge, as Link does); a crack that is too far away is refused. A crack that already has branches brings them along. The cracks keep working with every edit tool, their markers read N, N.1, N.2 ..., and in the JSON they share one `group_id` (flags `branch_main` / `branch`). JSON and masks are written after every branch, and **Undo** removes the last one. **Shift+click** on the button or **Shift+C** ends the mode.
- **XML reference** (left sidebar, View menu): shows the damage regions of `XML files/<photo>.xml` (Pascal VOC-style boxes and polygons) over the photo as a tracing guide, cracks in yellow and other damage classes in cyan. On screen only: never saved to the JSON, the masks or the exported images.

### Downloads

- **macOS (Apple Silicon)**: `CrackSegmentation-1.5.0-macos-arm64.dmg`. Not notarized by Apple: on first launch, open **System Settings → Privacy & Security** and click **Open Anyway** (only needed once).
- **Windows**: `CrackSegmentation-1.5.0-windows-Setup.exe`, or the portable `CrackSegmentation-1.5.0-windows-portable.zip`. Unsigned: if SmartScreen flags it, click **More info → Run anyway**.
- **Linux (Debian/Ubuntu, amd64)**: `cracksegmentation_1.5.0_amd64.deb`. Install with `sudo apt install ./cracksegmentation_1.5.0_amd64.deb`.
- **Linux (other distributions, amd64)**: `CrackSegmentation-1.5.0-linux-x86_64.tar.gz`. Portable, no package manager required: extract it and run `./CrackSegmentation/CrackSegmentation`.
