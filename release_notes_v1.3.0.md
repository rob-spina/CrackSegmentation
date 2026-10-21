## CrackSegmentation 1.3.0

### What's new in 1.3.0

- **Make uncertain / Make normal**: change the class of a crack you have already traced. Press **Ctrl+C** (**Cmd+C** on macOS), or Ctrl/Cmd+click the **Crack tool** sidebar button, then click a blue crack, on its line or on its number: it becomes uncertain (red). Press **Ctrl+U** (**Cmd+U**), or Ctrl/Cmd+click **Uncertain crack**, then click a red crack: it becomes a normal crack (blue) again. One crack per press: after the click everything works as usual again (for example deleting the crack by its marker); press the same keys again, or choose any other command, to cancel. JSON and masks (including `-crack_uncertain_mask.png`) are updated after the change, and **Undo** reverts it. Both commands are also in the Tools menu.
- **No more automatic uncertain cracks**: cracks are uncertain only when you decide so (Uncertain crack, Warn uncertain, Make uncertain). `CRACK_UNCERTAIN_AUTO = True` in `config.py` restores the old rule.
- **Warn uncertain** (right sidebar): press it before tracing; a new crack below the reliability/confidence limits opens a window proposing to save it as uncertain (Yes / No).
- **Crack labels** (right sidebar): rest the cursor on a crack number or line to see its length, width, reliability, confidence, views and status in a small window.
- **Ctrl/Cmd+click** with the Crack tool starts a trace next to a crack's end marker without deleting that crack.
- Active modes are listed in a yellow box in the status bar instead of on the photo.
- Other Ctrl+letter combinations no longer trigger unrelated commands.
- **Tracing next to an edge**: with Shift+clicked points, a faint crack running a few pixels from a strong edge or shadow line (e.g. right under a cornice) is now followed, instead of the route jumping onto the edge; a click is snapped onto the nearest line, not the strongest one.

### Downloads

- **macOS (Apple Silicon)**: `CrackSegmentation-1.3.0-macos-arm64.dmg`. Not notarized by Apple: on first launch, open **System Settings → Privacy & Security** and click **Open Anyway** (only needed once).
- **Windows**: `CrackSegmentation-1.3.0-windows-Setup.exe`, or the portable `CrackSegmentation-1.3.0-windows-portable.zip`. Unsigned: if SmartScreen flags it, click **More info → Run anyway**.
- **Linux (Debian/Ubuntu, amd64)**: `cracksegmentation_1.3.0_amd64.deb`. Install with `sudo apt install ./cracksegmentation_1.3.0_amd64.deb`.
- **Linux (other distributions, amd64)**: `CrackSegmentation-1.3.0-linux-x86_64.tar.gz`. Portable, no package manager required: extract it and run `./CrackSegmentation/CrackSegmentation`.
