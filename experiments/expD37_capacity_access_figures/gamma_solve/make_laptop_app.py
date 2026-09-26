"""Build a small Mac launcher; computation and data remain in the mini repo."""
from pathlib import Path
import plistlib
import shlex
import subprocess
import tempfile
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FINAL=ROOT/'results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/Gamma Solve.app'
FINAL.parent.joinpath('data').mkdir(parents=True,exist_ok=True)
build_dir=Path(tempfile.mkdtemp(prefix='launcher_build_',dir=FINAL.parent/'data'))
DEST=build_dir/'Gamma Solve.app'
# A native AppleScript applet avoids Launch Services rejecting a shell-only app.
applescript='''on run
    set launcherPath to (POSIX path of (path to me)) & "Contents/Resources/launch.sh"
    do shell script "/bin/bash " & quoted form of launcherPath
end run'''
subprocess.run(['/usr/bin/osacompile','-o',str(DEST),'-e',applescript],check=True)
info_path=DEST/'Contents/Info.plist'
info=plistlib.loads(info_path.read_bytes())
info.update(CFBundleName='Gamma Solve',CFBundleDisplayName='Gamma Solve',
            CFBundleIdentifier='local.sam.precisionmlps.gamma-solve',LSUIElement=True)
info_path.write_bytes(plistlib.dumps(info))
launcher=DEST/'Contents/Resources/launch.sh'
start_command='bash '+shlex.quote(str(HERE/'start.sh'))
script='''#!/bin/bash
set -euo pipefail
fail() {
  /usr/bin/osascript -e 'display alert "Gamma Solve could not connect to the mini" message "Check that Tailscale is connected and that ssh home works. The app keeps its calculations and data on the mini." as critical'
  exit 1
}
/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=8 home __START__ >/dev/null 2>&1 || fail
if ! /usr/bin/curl -fsS --max-time 2 http://127.0.0.1:18067/api/info >/dev/null 2>&1; then
  /usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=8 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -fNT -L 127.0.0.1:18067:127.0.0.1:8067 home || fail
fi
/usr/bin/open http://127.0.0.1:18067
'''
launcher.write_text(script.replace('__START__',shlex.quote(start_command)))
launcher.chmod(0o755)
subprocess.run(['/usr/bin/codesign','--force','--deep','--sign','-',str(DEST)],check=True)
if FINAL.exists():
    previous=plistlib.loads((FINAL/'Contents/Info.plist').read_bytes())
    if previous.get('CFBundleIdentifier')!='local.sam.precisionmlps.gamma-solve':
        raise RuntimeError('Refusing to replace an unrelated app bundle.')
    shutil.rmtree(FINAL)
DEST.rename(FINAL)
build_dir.rmdir()
print(FINAL)
