"""Build a native launcher in this repo's Codex geometry-reader output folder."""
from pathlib import Path
import plistlib
import subprocess

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1] / "results/checkpoint_G_interactive/geometry_reader_codex/Geometry Reader Codex.app"
if DEST.exists():
    print(f"Launcher already present: {DEST}")
else:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    command = str(HERE / "launch.command").replace("\\", "\\\\").replace('"', '\\"')
    source = 'on run\n do shell script "/bin/bash " & quoted form of "' + command + '"\nend run'
    subprocess.run(["/usr/bin/osacompile", "-o", str(DEST), "-e", source], check=True)
    info = DEST / "Contents/Info.plist"
    data = plistlib.loads(info.read_bytes())
    data.update(CFBundleName="Geometry Reader Codex", CFBundleDisplayName="Geometry Reader Codex",
                CFBundleIdentifier="local.sam.precisionmlps.geometry-reader-codex", LSUIElement=True)
    info.write_bytes(plistlib.dumps(data))
    subprocess.run(["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(DEST)], check=True)
    print(DEST)
