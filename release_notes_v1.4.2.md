## CrackSegmentation 1.4.2

### Fixed in 1.4.2

- **User manual in the installed app**: the manual did not open (*FileNotFoundError* in the manual panel) because the app was missing data files that recent versions of the PDF library (PyMuPDF) need. They are now included in every build.

Includes everything new in 1.4.0: **Edit extremes** (key **4**), hiding a crack's markers with a click, and **Hide uncertain overlay** / **Hide normal overlay**.

### Downloads

- **macOS (Apple Silicon)**: `CrackSegmentation-1.4.2-macos-arm64.dmg`. Not notarized by Apple: on first launch, open **System Settings → Privacy & Security** and click **Open Anyway** (only needed once).
- **Windows**: `CrackSegmentation-1.4.2-windows-Setup.exe`, or the portable `CrackSegmentation-1.4.2-windows-portable.zip`. Unsigned: if SmartScreen flags it, click **More info → Run anyway**.
- **Linux (Debian/Ubuntu, amd64)**: `cracksegmentation_1.4.2_amd64.deb`. Install with `sudo apt install ./cracksegmentation_1.4.2_amd64.deb`.
- **Linux (other distributions, amd64)**: `CrackSegmentation-1.4.2-linux-x86_64.tar.gz`. Portable, no package manager required: extract it and run `./CrackSegmentation/CrackSegmentation`.
