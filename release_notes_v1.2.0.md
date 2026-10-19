## CrackSegmentation 1.2.0

### What's new in 1.2.0

- **Interrupted crack**: when you click the last point of a crack, the route is checked against the photo. If it crosses a stretch with no crack (a gap), an orange **INTERRUPTED CRACK** message appears on screen and the trace is stored as **separate cracks**, one per continuous piece, without the gap. **Save** writes them as separate shapes in the JSON, with the gap left black in the PNG masks (in Uncertain mode they are written at once, as every uncertain trace). A single **Undo** removes the whole trace. Short breaks of a few pixels and stray dark specks inside a gap are ignored; the middle of wide cracks counts as crack.
- **Hideable sidebars**: a small flat sidebar icon at the top of each sidebar hides it. Hide the left one and the photo extends to the left, the right one and it extends to the right; hide both and the photo takes the whole width. A thin strip with the icon stays on that side to bring the sidebar back.

### Downloads

- **macOS (Apple Silicon)**: `CrackSegmentation-1.2.0-macos-arm64.dmg`. Not notarized by Apple: on first launch, open **System Settings → Privacy & Security** and click **Open Anyway** (only needed once).
- **Windows**: `CrackSegmentation-1.2.0-windows-Setup.exe`, or the portable `CrackSegmentation-1.2.0-windows-portable.zip`. Unsigned: if SmartScreen flags it, click **More info → Run anyway**.
- **Linux (Debian/Ubuntu, amd64)**: `cracksegmentation_1.2.0_amd64.deb`. Install with `sudo apt install ./cracksegmentation_1.2.0_amd64.deb`.
- **Linux (other distributions, amd64)**: `CrackSegmentation-1.2.0-linux-x86_64.tar.gz`. Portable, no package manager required: extract it and run `./CrackSegmentation/CrackSegmentation`.
