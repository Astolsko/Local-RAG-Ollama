import subprocess
import sys
from pathlib import Path

def test_frontend_js_suite():
    # Run vitest run command
    project_root = Path(__file__).resolve().parent.parent
    frontend_dir = project_root / "frontend"
    
    # Run npm test inside the frontend directory
    # On Windows, we use shell=True to execute npm safely
    result = subprocess.run(
        ["npm", "run", "test"],
        cwd=str(frontend_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        shell=True
    )
    
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    
    assert result.returncode == 0, f"Frontend vitest tests failed!\\nStdout:\\n{result.stdout}\\nStderr:\\n{result.stderr}"
