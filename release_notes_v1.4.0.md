## CrackSegmentation 1.4.0

### What's new in 1.4.0

- **Edit extremes** (right sidebar, Tools menu, key **4**): lengthen or shorten a crack you have already traced. The crack numbers disappear and every crack end becomes a large red dot. Click (or press and drag) a dot: the other dots disappear; then click the point the crack should reach and it is lengthened along the crack edge, like a normal trace; click on the crack itself to shorten it. Click the same dot again to release it. Clicks away from the dots still trace new cracks, Cmd/Ctrl+C and Cmd/Ctrl+U keep working, and the dots can be dragged with Zoom window on too. **Undo** reverts each change; press the button or **4** again and the numbers come back and the JSON and masks are written.
- **Hide a crack's markers with a click**: with the Crack tool, click on a crack line to hide its numbered marker and end dot; click the crack again to show them. Hidden markers cannot delete the crack. Shift+click or Cmd/Ctrl+click on the line starts a trace there instead.
- **Hide uncertain overlay / Hide normal overlay** (left sidebar, View menu): hide only the red fill of the uncertain cracks, or only the blue fill of the normal ones. **N** still hides both.

### Downloads

- **macOS (Apple Silicon)**: `CrackSegmentation-1.4.0-macos-arm64.dmg`. Not notarized by Apple: on first launch, open **System Settings → Privacy & Security** and click **Open Anyway** (only needed once).
- **Windows**: `CrackSegmentation-1.4.0-windows-Setup.exe`, or the portable `CrackSegmentation-1.4.0-windows-portable.zip`. Unsigned: if SmartScreen flags it, click **More info → Run anyway**.
- **Linux (Debian/Ubuntu, amd64)**: `cracksegmentation_1.4.0_amd64.deb`. Install with `sudo apt install ./cracksegmentation_1.4.0_amd64.deb`.
- **Linux (other distributions, amd64)**: `CrackSegmentation-1.4.0-linux-x86_64.tar.gz`. Portable, no package manager required: extract it and run `./CrackSegmentation/CrackSegmentation`.
