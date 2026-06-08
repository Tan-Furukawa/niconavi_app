import subprocess
import sys
import os
from pathlib import Path


def main() -> None:
    # Launch the package entry module with the same interpreter regardless of this file's location.
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"

    if not (src_dir / "niconavi_app" / "main.py").exists():
        raise FileNotFoundError(f"Unable to locate {src_dir / 'niconavi_app' / 'main.py'}")

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(src_dir)
        if not env.get("PYTHONPATH")
        else str(src_dir) + os.pathsep + env["PYTHONPATH"]
    )
    result = subprocess.run([sys.executable, "-u", "-m", "niconavi_app.main"], env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
