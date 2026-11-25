; Inno Setup script for AIR Detection (Windows).
;
; Packages the PyInstaller onedir build (dist\AIR Detection\) into a
; single "AIR Detection Setup v1.1.0.exe" installer. Before copying any files,
; InitializeSetup() below runs the standalone compatibility checker
; (build_assets\check_compatibility.py, built separately into
; build_assets\dist\Check Compatibility.exe) so a student on an
; unsupported machine gets a clear alert - with the actual reason
; (missing VC++ runtime, 32-bit Windows, too little RAM/disk, OS too old)
; - before wasting time on a install that likely won't run.
;
; build_windows.bat / build_windows.py run all of this. Build order:
;   0. Nuitka compiles ai_detection_lab into build\nuitka\ai_detection_lab.*.pyd (no .py/.pyc ships)
;   1. pyinstaller --noconfirm AIR_Detection_Windows.spec        (produces dist\AIR Detection\)
;   2. pyinstaller --onefile --windowed --name "Check Compatibility"
;      --icon build_assets\app_icon.ico
;      --distpath build_assets\dist --workpath build_assets\build
;      --specpath build_assets build_assets\check_compatibility.py  (produces build_assets\dist\Check Compatibility.exe)
;   3. ISCC installer.iss                                           (produces dist_installer\AIR Detection Setup.exe)

#define MyAppName "AIR Detection"
#define MyAppVersion "1.1.0"
#define MyAppExeName "AIR Detection.exe"
#define MyCheckerExeName "Check Compatibility.exe"
; build_windows.py passes /DDistDir= and /DCheckerDir= (its build output lives outside OneDrive)
#ifndef DistDir
  #define DistDir "dist"
#endif
#ifndef CheckerDir
  #define CheckerDir "build_assets\dist"
#endif

[Setup]
AppId={{99191486-E4A3-4181-B761-8F162C3EEC41}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=dist_installer
OutputBaseFilename=AIR Detection Setup v{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
SetupIconFile=build_assets\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
WizardStyle=modern
#ifdef BigSetup
; build_windows.py passes /DBigSetup when the app (CUDA PyTorch) is too big
; for one setup .exe (~2.1 GB): the setup becomes .exe + .bin files.
DiskSpanning=yes
DiskSliceSize=max
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; The real app, installed to {app}.
Source: "{#DistDir}\AIR Detection\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#CheckerDir}\{#MyCheckerExeName}"; DestDir: "{app}"; Flags: ignoreversion
; A second copy extracted to {tmp} only, so InitializeSetup can run the
; compatibility check BEFORE any of the app's files above are copied.
Source: "{#CheckerDir}\{#MyCheckerExeName}"; DestDir: "{tmp}"; Flags: dontcopy

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Check System Compatibility"; Filename: "{app}\{#MyCheckerExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
  CheckerPath, CheckerArgs: String;
begin
  Result := True;
  ExtractTemporaryFile('{#MyCheckerExeName}');
  CheckerPath := ExpandConstant('{tmp}\{#MyCheckerExeName}');
  // Run WITHOUT --silent: the checker shows its own MessageBox naming the
  // exact reason(s) a hard check failed (or that everything passed) -
  // that IS the "alert explaining why" this whole tool exists for. This
  // call blocks until the student closes that box, then we only need the
  // exit code (0 = all hard checks passed) to decide whether to also ask
  // about continuing anyway.
  CheckerArgs := '';
  if WizardSilent then CheckerArgs := '--silent';
  if Exec(CheckerPath, CheckerArgs, '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
  begin
    if ResultCode <> 0 then
    begin
      if (not WizardSilent) and (MsgBox('This machine failed the compatibility check (see the reason in the message box you just closed).' + #13#10 +
                'Continue installing anyway? The program may fail to launch.',
                mbConfirmation, MB_YESNO) = IDNO) then
        Result := False;
    end;
  end
  else
  begin
    // Checker itself couldn't even be launched - do not block install on
    // that; just let the student proceed without a compatibility report.
    Result := True;
  end;
end;
