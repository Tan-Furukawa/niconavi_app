Both packages use an uv-managed CPython distribution from
`python-build-standalone` and install the exact runtime dependencies exported
from `uv.lock`. PyInstaller is not used. Build outputs go under `dist/` and are
ignored by Git.

The package icon is `src/niconavi_app/assets/favicon.png`. Its SHA-256 is
`70acd223da029c98f4361632a3cffe9cabb29cb759b176ebf63fcfcd1e7fced4`, which
matches the image used on the NicoNavi website home page:
<https://tan-furukawa.github.io/niconavi/images/avatar.png>.

## Windows x64 installer

Build on Windows x64. Install:

- uv
- Inno Setup 6

From PowerShell at the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\build_windows.ps1
```

The result is:

```text
dist\windows\NicoNavi-<version>-Setup-x64.exe
```

Use `-StageOnly` to prepare and smoke-test the portable application directory
without compiling the Inno Setup installer. `-PythonVersion`, `-AppVersion`,
`-OutputDirectory`, `-PackagingCacheDirectory`, and `-IsccPath` can override
their corresponding defaults.

The installed application is per-user and does not request administrator
rights. The required Microsoft Visual C++ runtime is bundled with the
application, so users do not need to install it separately. Mutable application
data is stored under `%LOCALAPPDATA%\NicoNavi`.

### Build with GitHub Actions

The `Build Windows installer` workflow runs the same build on
`windows-latest`. Open the repository's **Actions** tab, choose the workflow,
and select **Run workflow**. It also runs automatically for tags matching
`v*`.

The job creates `Windows/NicoNavi-Setup.exe`. Download the `NicoNavi-Windows` artifact
from the completed workflow run. After extraction it contains:

```text
NicoNavi-Setup.exe
NicoNavi-Setup.exe.sha256
```

Action artifacts are stored by GitHub for 14 days and then deleted
automatically. They are not added to the Git history. This workflow does not
create a GitHub Release or upload files to one.

## Ubuntu/Linux x86_64 AppImage

Build on Linux x86_64 with uv and curl installed:

```bash
./packaging/build_appimage.sh
```

The result is:

```text
dist/appimage/NicoNavi-<version>-x86_64.AppImage
```

The script caches the standalone Python plus pinned, checksum-verified releases
of `appimagetool` and the AppImage Type 2 runtime in the user cache. FUSE is not
required while building. Mutable application data is stored under
`${XDG_DATA_HOME:-~/.local/share}/niconavi`.

Run the completed image with:

```bash
chmod +x dist/appimage/NicoNavi-*-x86_64.AppImage
./dist/appimage/NicoNavi-*-x86_64.AppImage
```

# NicoNaviのパッケージング

[日本語](#niconaviのパッケージング) | [English](#english)

Windows版とLinux版のどちらも、`python-build-standalone`から提供される
uv管理のCPythonを使用します。実行時の依存パッケージには、`uv.lock`から
書き出した固定バージョンを使用します。

ビルド結果は`dist/`以下に作成され、Gitの管理対象には含まれません。

アプリのアイコンには`src/niconavi_app/assets/favicon.png`を使用します。
SHA-256は
`70acd223da029c98f4361632a3cffe9cabb29cb759b176ebf63fcfcd1e7fced4`です。
これはNicoNavi公式サイトのトップページで使用されている画像と同一です。

<https://tan-furukawa.github.io/niconavi/images/avatar.png>

## Windows x64インストーラー

### Windows上でビルドする場合

Windows x64環境に次のソフトウェアをインストールしてください。

- uv
- Inno Setup 6

リポジトリのルートディレクトリでPowerShellを開き、次のコマンドを実行します。

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\build_windows.ps1
```

ビルド結果は次の場所に作成されます。

```text
dist\windows\NicoNavi-<バージョン>-Setup-x64.exe
```

`-StageOnly`を指定すると、Inno Setupによるインストーラーの生成を省略し、
ポータブルアプリ用ディレクトリの作成と簡易動作確認だけを行います。

必要に応じて、次の引数で既定値を変更できます。

- `-PythonVersion`
- `-AppVersion`
- `-OutputDirectory`
- `-PackagingCacheDirectory`
- `-IsccPath`

インストールはユーザー単位で行われるため、管理者権限は要求されません。
実行に必要なMicrosoft Visual C++ランタイムはアプリに同梱されるため、
利用者が別途インストールする必要はありません。
設定、ログ、アップロードファイルなどの変更可能なデータは、
`%LOCALAPPDATA%\NicoNavi`以下に保存されます。

### GitHub Actionsでビルドする場合

`Build Windows installer`ワークフローを使用すると、GitHub Actionsの
`windows-latest`環境で同じビルドを実行できます。

手動で実行する手順は次のとおりです。

1. GitHub上でリポジトリの「Actions」タブを開きます。
2. 「Build Windows installer」を選択します。
3. 「Run workflow」を選択して実行します。
4. ビルド完了後、実行結果ページ下部の「Artifacts」から
   `NicoNavi-Windows`をダウンロードします。

`v*`に一致するタグをpushした場合も、自動的にビルドされます。

ワークフロー内では`Windows/NicoNavi-Setup.exe`が作成されます。ダウンロードした
`NicoNavi-Windows`を展開すると、次のファイルが含まれています。

```text
NicoNavi-Setup.exe
NicoNavi-Setup.exe.sha256
```

`NicoNavi-Setup.exe.sha256`には、`NicoNavi-Setup.exe`が破損していないか確認するための
SHA-256ハッシュが記録されています。

これらはGitHub Releaseではなく、GitHub ActionsのArtifactとしてGitHub側に
14日間保存され、その後自動的に削除されます。Gitの履歴やソースコードには
追加されません。このワークフローはGitHub Releaseの作成や、Releaseへの
ファイルのアップロードを行いません。

## Ubuntu/Linux x86_64 AppImage

Linux x86_64環境にuvとcurlをインストールし、次のコマンドを実行します。

```bash
./packaging/build_appimage.sh
```

ビルド結果は次の場所に作成されます。

```text
dist/appimage/NicoNavi-<バージョン>-x86_64.AppImage
```

ビルドスクリプトは、スタンドアロン版Python、バージョン固定済みの
`appimagetool`、AppImage Type 2ランタイムをユーザーキャッシュに保存します。
`appimagetool`とランタイムは、ダウンロード時にチェックサムで検証されます。
ビルド時にFUSEは必要ありません。

設定、ログ、アップロードファイルなどの変更可能なデータは、
`${XDG_DATA_HOME:-~/.local/share}/niconavi`以下に保存されます。

作成したAppImageは次のように実行できます。

```bash
chmod +x dist/appimage/NicoNavi-*-x86_64.AppImage
./dist/appimage/NicoNavi-*-x86_64.AppImage
```

---
