from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import ensure_dir, load_config, project_path


def main() -> None:
    cfg = load_config()
    target = project_path(cfg["audio"]["librispeech_dir"])
    ensure_dir(target.parent)
    if target.exists() and any(target.rglob("*.flac")):
        print(f"LibriSpeech appears available at {target}")
        return
    print("LibriSpeech test-clean was not found.")
    print("Download manually from https://www.openslr.org/12 or place files at:")
    print(target)
    print("The POC can still run with synthetic placeholder audio.")


if __name__ == "__main__":
    main()
