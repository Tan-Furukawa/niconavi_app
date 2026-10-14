#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef MyFileVersion
  #define MyFileVersion "0.0.0.0"
#endif
#ifndef SourceDir
  #error SourceDir must be passed by build_windows.ps1
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif

#define MyAppName "NicoNavi"
#define MyAppPublisher "Tan Furukawa"
#define MyAppURL "https://tan-furukawa.github.io/niconavi/"

[Setup]
AppId={{D9378598-DB95-4AF2-891D-A09906741661}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
VersionInfoVersion={#MyFileVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\NicoNavi
DefaultGroupName=NicoNavi
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=NicoNavi-{#MyAppVersion}-Setup-x64
SetupIconFile={#SourceDir}\niconavi.ico
UninstallDisplayIcon={app}\niconavi.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\NicoNavi"; Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\niconavi.ico"; AppUserModelID: "io.github.tan_furukawa.NicoNavi"
Name: "{autodesktop}\NicoNavi"; Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\niconavi.ico"; AppUserModelID: "io.github.tan_furukawa.NicoNavi"; Tasks: desktopicon

[Run]
Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,NicoNavi}"; Flags: nowait postinstall skipifsilent
