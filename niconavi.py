import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Launch src/main.py with the same interpreter regardless of this file's location.
    project_root = Path(__file__).resolve().parent
    target = project_root / "src" / "main.py"

    if not target.exists():
        raise FileNotFoundError(f"Unable to locate {target}")

    result = subprocess.run([sys.executable, "-u", str(target), "--desktop"])
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
