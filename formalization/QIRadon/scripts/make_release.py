#!/usr/bin/env python3
"""Package the verified source and human-readable note, without build caches."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUTPUT = DIST / "qi_radon_lean.zip"


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/check_structure.py")], check=True)
    pdf = DIST / "qi_radon_theorems.pdf"
    if not pdf.is_file():
        raise SystemExit("Render docs/qi_radon_theorems.tex to dist/ first.")
    files = [ROOT / name for name in (
        "README.md", "STATUS.md", "lakefile.toml", "lake-manifest.json",
        "lean-toolchain", ".gitignore", "QIRadon.lean", "Audit.lean",
        "verification/README.md", "verification/axioms.txt",
        "verification/structure.json", "dist/qi_radon_theorems.pdf")]
    files += sorted((ROOT / "QIRadon").rglob("*.lean"))
    files += sorted((ROOT / "docs").glob("*.md"))
    files += sorted((ROOT / "docs").glob("*.tex"))
    files += sorted((ROOT / "scripts").glob("*.py"))
    # Reproducible experiments are part of the note, not dependencies of Lean.
    for suffix in ("*.py", "*.json", "*.md", "*.pdf", "*.png"):
        files += sorted((ROOT / "empirical").rglob(suffix))
    payload = {str(p.relative_to(ROOT)): p.read_bytes() for p in files}
    manifest = {
        "format": 1,
        "source_modules": len(list((ROOT / "QIRadon").rglob("*.lean"))),
        "sha256": {name: hashlib.sha256(data).hexdigest()
                   for name, data in sorted(payload.items())},
        "excluded": ["archive/", ".lake/", "generated logs", "the release ZIP itself"]
    }
    raw_manifest = (json.dumps(manifest, indent=2) + "\n").encode()
    (ROOT / "verification/release_manifest.json").write_bytes(raw_manifest)
    payload["verification/release_manifest.json"] = raw_manifest
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo("QI-Radon/" + name, date_time=(2026, 10, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    with zipfile.ZipFile(OUTPUT) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed.")
        for name, expected in manifest["sha256"].items():
            actual = hashlib.sha256(z.read("QI-Radon/" + name)).hexdigest()
            if actual != expected:
                raise RuntimeError("Payload checksum mismatch: " + name)
    print(f"Created {OUTPUT.name}: {len(payload)} files, {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
