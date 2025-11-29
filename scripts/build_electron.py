#!/usr/bin/env python3
"""
Build the Electron desktop application.

This script:
1. Checks prerequisites (node, npm, electron-builder)
2. Builds the UI (if needed)
3. Runs electron-builder to create platform-specific distributables

Usage:
    python scripts/build_electron.py [--platform mac|win|linux|all] [--skip-ui-build]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
UI_DIR = PROJECT_ROOT / "ui"
ELECTRON_DIR = PROJECT_ROOT / "electron"


def check_command(cmd: str) -> bool:
    """Check if a command is available."""
    try:
        subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            check=True,
            timeout=10
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_prerequisites() -> bool:
    """Check if all required tools are available."""
    print("🔍 Checking prerequisites...")
    required = {
        "node": "Node.js",
        "npm": "npm"
    }
    missing = []
    for cmd, name in required.items():
        if check_command(cmd):
            version = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            print(f"  ✅ {name}: {version}")
        else:
            print(f"  ❌ {name}: not found")
            missing.append(name)
    if missing:
        print(f"\n❌ Missing required tools: {', '.join(missing)}")
        print("Please install Node.js and npm: https://nodejs.org/")
        return False
    # Check electron-builder in electron directory
    print("  🔍 Checking electron-builder...")
    try:
        result = subprocess.run(
            ["npm", "list", "electron-builder"],
            cwd=ELECTRON_DIR,
            capture_output=True,
            text=True,
            check=False
        )
        if "electron-builder" in result.stdout:
            print("  ✅ electron-builder: installed")
            return True
        else:
            print("  ⚠️  electron-builder: not found, installing...")
            subprocess.run(
                ["npm", "install"],
                cwd=ELECTRON_DIR,
                check=True
            )
            print("  ✅ electron-builder: installed")
            return True
    except subprocess.CalledProcessError:
        print("  ❌ Failed to check/install electron-builder")
        return False


def build_ui(skip: bool = False) -> bool:
    """Build the UI if needed."""
    if skip:
        print("⏭️  Skipping UI build (--skip-ui-build flag)")
        return True
    print("\n📦 Building UI...")
    ui_dist = UI_DIR / "dist"
    if ui_dist.exists() and any(ui_dist.iterdir()):
        print("  ℹ️  UI dist directory exists, checking if rebuild is needed...")
        # Could add more sophisticated checking here
    try:
        # Check if node_modules exists
        if not (UI_DIR / "node_modules").exists():
            print("  📥 Installing UI dependencies...")
            subprocess.run(
                ["npm", "install"],
                cwd=UI_DIR,
                check=True
            )
        print("  🔨 Running UI build...")
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=UI_DIR,
            check=True
        )
        if result.returncode == 0:
            print("  ✅ UI build completed")
            return True
        else:
            print("  ❌ UI build failed")
            return False
    except subprocess.CalledProcessError as e:
        print(f"  ❌ UI build failed: {e}")
        return False


def cleanup_mounted_dmgs():
    """Unmount any DMG volumes from previous builds."""
    print("  🧹 Cleaning up any mounted DMG volumes...")
    try:
        # Get list of mounted disk images using hdiutil
        hdiutil_result = subprocess.run(
            ["hdiutil", "info"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if hdiutil_result.returncode != 0:
            print("    ℹ️  Could not query disk images")
            return
        
        # Parse hdiutil output to find disk images and their mount points
        lines = hdiutil_result.stdout.split('\n')
        disk_images = []
        current_image = None
        
        for i, line in enumerate(lines):
            # Look for image file paths
            if 'image-path' in line.lower() or 'image-alias' in line.lower():
                # Extract the image path
                parts = line.split(':')
                if len(parts) > 1:
                    image_path = parts[-1].strip()
                    if 'Lauren AI' in image_path or 'lauren-ai' in image_path.lower():
                        current_image = {'path': image_path, 'devices': [], 'volumes': []}
                        disk_images.append(current_image)
            
            # Look for device identifiers (like /dev/disk4)
            if current_image and ('/dev/disk' in line):
                device = line.strip().split()[0] if line.strip().split() else None
                if device and device.startswith('/dev/disk'):
                    current_image['devices'].append(device)
            
            # Look for volume mount points
            if current_image and '/Volumes/' in line:
                volume_path = line.strip().split()[-1] if line.strip().split() else None
                if volume_path and volume_path.startswith('/Volumes/'):
                    current_image['volumes'].append(volume_path)
        
        # Also check /Volumes directly for any Lauren AI volumes
        volumes_dir = Path("/Volumes")
        if volumes_dir.exists():
            for volume in volumes_dir.iterdir():
                if 'Lauren AI' in volume.name or 'lauren-ai' in volume.name.lower():
                    # Check if this volume is already in our list
                    found = False
                    for img in disk_images:
                        if str(volume) in img['volumes']:
                            found = True
                            break
                    if not found:
                        # Add as standalone volume
                        disk_images.append({
                            'path': None,
                            'devices': [],
                            'volumes': [str(volume)]
                        })
        
        # Unmount found volumes and detach devices
        unmounted = False
        for img in disk_images:
            # First, try unmounting volumes
            for volume in img['volumes']:
                try:
                    print(f"    Unmounting volume {volume}...")
                    detach_result = subprocess.run(
                        ["hdiutil", "detach", volume, "-quiet"],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10
                    )
                    if detach_result.returncode == 0:
                        unmounted = True
                        print(f"    ✅ Unmounted {volume}")
                    else:
                        # Try force unmount
                        force_result = subprocess.run(
                            ["hdiutil", "detach", volume, "-force", "-quiet"],
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=10
                        )
                        if force_result.returncode == 0:
                            unmounted = True
                            print(f"    ✅ Force unmounted {volume}")
                except (subprocess.TimeoutExpired, Exception) as e:
                    print(f"    ⚠️  Error unmounting {volume}: {e}")
            
            # Then, try detaching devices directly (this is what electron-builder does)
            for device in img['devices']:
                try:
                    print(f"    Detaching device {device}...")
                    detach_result = subprocess.run(
                        ["hdiutil", "detach", device, "-quiet"],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10
                    )
                    if detach_result.returncode == 0:
                        unmounted = True
                        print(f"    ✅ Detached {device}")
                    else:
                        # Try force detach
                        force_result = subprocess.run(
                            ["hdiutil", "detach", device, "-force", "-quiet"],
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=10
                        )
                        if force_result.returncode == 0:
                            unmounted = True
                            print(f"    ✅ Force detached {device}")
                        else:
                            print(f"    ⚠️  Could not detach {device} (may be in use)")
                except (subprocess.TimeoutExpired, Exception) as e:
                    print(f"    ⚠️  Error detaching {device}: {e}")
        
        if not disk_images:
            print("    ℹ️  No mounted DMG volumes found")
    except Exception as e:
        print(f"    ⚠️  Error during cleanup: {e}")


def build_electron(platform: str = "all") -> bool:
    """Build the Electron application."""
    print(f"\n🚀 Building Electron app for platform: {platform}")
    
    # Clean up any mounted DMG volumes before building (macOS only)
    if platform in ("mac", "all") and sys.platform == "darwin":
        cleanup_mounted_dmgs()
    
    platform_map = {
        "mac": "build:mac",
        "win": "build:win",
        "linux": "build:linux",
        "all": "build"
    }
    if platform not in platform_map:
        print(f"  ❌ Invalid platform: {platform}")
        print(f"  Valid options: {', '.join(platform_map.keys())}")
        return False
    script = platform_map[platform]
    try:
        result = subprocess.run(
            ["npm", "run", script],
            cwd=ELECTRON_DIR,
            check=True
        )
        if result.returncode == 0:
            print(f"  ✅ Electron build completed for {platform}")
            
            # Clean up any DMG volumes created during the build (macOS only)
            if platform in ("mac", "all") and sys.platform == "darwin":
                print("  🧹 Cleaning up build artifacts...")
                cleanup_mounted_dmgs()
            
            dist_dir = ELECTRON_DIR / "dist"
            if dist_dir.exists():
                print(f"  📦 Build artifacts in: {dist_dir}")
                # List built files
                for item in dist_dir.iterdir():
                    if item.is_file():
                        size_mb = item.stat().st_size / (1024 * 1024)
                        print(f"    - {item.name} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"  ❌ Electron build failed for {platform}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Electron build failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build the Electron desktop application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build for all platforms
  python scripts/build_electron.py

  # Build for macOS only
  python scripts/build_electron.py --platform mac

  # Build for Windows only
  python scripts/build_electron.py --platform win

  # Build for Linux only
  python scripts/build_electron.py --platform linux

  # Skip UI build (use existing dist)
  python scripts/build_electron.py --skip-ui-build
        """
    )
    parser.add_argument(
        "--platform",
        choices=["mac", "win", "linux", "all"],
        default="all",
        help="Platform to build for (default: all)"
    )
    parser.add_argument(
        "--skip-ui-build",
        action="store_true",
        help="Skip UI build step (use existing dist)"
    )
    args = parser.parse_args()
    print("=" * 60)
    print("Electron Build Script")
    print("=" * 60)
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    # Build UI
    if not build_ui(skip=args.skip_ui_build):
        sys.exit(1)
    # Build Electron
    if not build_electron(platform=args.platform):
        sys.exit(1)
    print("\n" + "=" * 60)
    print("✅ Build completed successfully!")
    print("=" * 60)
    dist_dir = ELECTRON_DIR / "dist"
    if dist_dir.exists():
        print(f"\n📦 Build artifacts are in: {dist_dir.absolute()}")


if __name__ == "__main__":
    main()

