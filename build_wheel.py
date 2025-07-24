#!/usr/bin/env python3
"""
Simple build script for Vibequake wheel.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

def create_wheel():
    """Create a wheel file for the vibequake package."""
    
    # Create dist directory
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)
    
    # Create wheel directory structure
    wheel_name = "vibequake-0.1.0-py3-none-any.whl"
    wheel_dir = dist_dir / wheel_name
    
    # Create wheel contents
    wheel_contents = {
        "vibequake/": [
            "src/vibequake/__init__.py",
            "src/vibequake/core.py", 
            "src/vibequake/utils.py",
            "src/vibequake/api.py",
            "src/vibequake/cli.py"
        ],
        "vibequake-0.1.0.data/scripts/": [
            "src/vibequake/cli.py"
        ],
        "vibequake-0.1.0.dist-info/": [
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "CHANGELOG.md"
        ]
    }
    
    # Create wheel file
    with zipfile.ZipFile(wheel_dir, 'w', zipfile.ZIP_DEFLATED) as wheel:
        
        # Add package files
        for dest_dir, files in wheel_contents.items():
            for file_path in files:
                if Path(file_path).exists():
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Add to wheel
                    wheel_path = f"{dest_dir}{Path(file_path).name}"
                    wheel.writestr(wheel_path, content)
        
        # Add metadata
        metadata = """Metadata-Version: 2.1
Name: vibequake
Version: 0.1.0
Summary: Analyze vibrational resonance and predict catastrophic failure in mechanical systems
Home-page: https://github.com/vibequake/vibequake
Author: Vibequake Team
Author-email: team@vibequake.dev
License: MIT
Classifier: Development Status :: 4 - Beta
Classifier: Intended Audience :: Science/Research
Classifier: Intended Audience :: Developers
Classifier: License :: OSI Approved :: MIT License
Classifier: Operating System :: OS Independent
Classifier: Programming Language :: Python :: 3
Classifier: Programming Language :: Python :: 3.8
Classifier: Programming Language :: Python :: 3.9
Classifier: Programming Language :: Python :: 3.10
Classifier: Programming Language :: Python :: 3.11
Classifier: Programming Language :: Python :: 3.12
Classifier: Topic :: Scientific/Engineering :: Physics
Classifier: Topic :: Scientific/Engineering :: Mechanical
Classifier: Topic :: Software Development :: Libraries :: Python Modules
Requires-Python: >=3.8
Requires-Dist: numpy>=1.21.0
Requires-Dist: scipy>=1.7.0
Requires-Dist: matplotlib>=3.5.0
Requires-Dist: pandas>=1.3.0
Requires-Dist: fastapi>=0.104.0
Requires-Dist: uvicorn[standard]>=0.24.0
Requires-Dist: pydantic>=2.0.0
Requires-Dist: python-multipart>=0.0.6

"""
        wheel.writestr("vibequake-0.1.0.dist-info/METADATA", metadata)
        
        # Add WHEEL file
        wheel_content = """Wheel-Version: 1.0
Generator: vibequake-build-script
Root-Is-Purelib: true
Tag: py3-none-any
"""
        wheel.writestr("vibequake-0.1.0.dist-info/WHEEL", wheel_content)
        
        # Add entry points
        entry_points = """[console_scripts]
vibequake = vibequake.cli:main

"""
        wheel.writestr("vibequake-0.1.0.dist-info/entry_points.txt", entry_points)
    
    print(f"✅ Wheel created: {wheel_dir}")
    return wheel_dir

if __name__ == "__main__":
    wheel_path = create_wheel()
    print(f"🎉 Vibequake wheel built successfully!")
    print(f"📦 Wheel location: {wheel_path}")
    print(f"📏 Wheel size: {wheel_path.stat().st_size / 1024:.1f} KB")