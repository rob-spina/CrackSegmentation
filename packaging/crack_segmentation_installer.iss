; Inno Setup script for CrackSegmentation.
;
; Produces a normal double-click Windows installer (Setup.exe) that copies
; the PyInstaller onedir build into Program Files, adds Start Menu /
; Desktop shortcuts, and registers an uninstaller -- instead of handing
; the operator a raw folder to run from.
;
; Requires PyInstaller to have already been run (see build_windows.bat,
; which runs both steps in order). Requires Inno Setup itself to compile
; this script: free download, https://jrsoftware.org/isdl.php -- install
; it once on the Windows machine you're building on, then either open
; this file in the Inno Setup Compiler GUI and click Compile, or run
; ISCC.exe against it from the command line (build_windows.bat does this
; automatically if it finds ISCC.exe already installed).

#define MyAppName "CrackSegmentation"
#define MyAppVersion "1.0.0"
#define MyAppExeName "CrackSegmentation.exe"

[Setup]
AppId={{B6B6C6C2-7B6E-4B7B-9A7B-CRACKSEG0001}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=CrackSegmentation-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
; No code-signing certificate is applied here -- Windows SmartScreen will
; likely warn on first run of the installer AND of the installed app
; itself ("Windows protected your PC"). This is expected for an unsigned
; installer, not a sign the build is broken; the operator clicks "More
; info" -> "Run anyway" once. A real certificate removes this warning but
; costs money and is out of scope here.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Crea un collegamento sul Desktop"; GroupDescription: "Collegamenti aggiuntivi:"

[Files]
; Everything PyInstaller put in dist\CrackSegmentation\ (the .exe plus its
; bundled Python/OpenCV/Tk/scikit-image DLLs and support files).
Source: "dist\CrackSegmentation\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Disinstalla {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Avvia {#MyAppName} ora"; Flags: nowait postinstall skipifsilent
