"""
Version Information Module

Provides version endpoint with git SHA, build time, and dependency lock hash.
"""

import hashlib
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class VersionInfo:
    """Version information provider."""

    def __init__(self, project_root: Optional[str] = None) -> None:
        """
        Initialize version info.

        Args:
            project_root: Root directory of the project
        """
        if project_root is None:
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parent.parent
        else:
            self.project_root = Path(project_root)
        self._git_sha: Optional[str] = None
        self._build_time: Optional[str] = None
        self._dependency_hash: Optional[str] = None

    def get_git_sha(self) -> str:
        """
        Get current git commit SHA.

        Returns:
            Git SHA or 'unknown' if not available
        """
        if self._git_sha is not None:
            return self._git_sha
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._git_sha = result.stdout.strip()
            else:
                self._git_sha = "unknown"
        except Exception:
            self._git_sha = "unknown"
        return self._git_sha

    def get_git_short_sha(self) -> str:
        """
        Get short git commit SHA (first 7 characters).

        Returns:
            Short git SHA
        """
        sha = self.get_git_sha()
        if sha != "unknown":
            return sha[:7]
        return sha

    def get_build_time(self) -> str:
        """
        Get build time.

        For now, returns the modification time of this file.
        In a real build system, this would be set during build.

        Returns:
            ISO format timestamp
        """
        if self._build_time is not None:
            return self._build_time
        try:
            build_time_env = os.getenv("BUILD_TIME")
            if build_time_env:
                self._build_time = build_time_env
                return self._build_time
            this_file = Path(__file__)
            mtime = this_file.stat().st_mtime
            self._build_time = datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            self._build_time = datetime.now().isoformat()
        return self._build_time

    def get_dependency_hash(self) -> str:
        """
        Get hash of dependency lock file.

        Returns:
            SHA256 hash of requirements-lock.txt or 'unknown'
        """
        if self._dependency_hash is not None:
            return self._dependency_hash
        try:
            lock_file = self.project_root / "requirements-lock.txt"
            if not lock_file.exists():
                lock_file = self.project_root / "requirements.txt"
            if lock_file.exists():
                content = lock_file.read_bytes()
                self._dependency_hash = hashlib.sha256(content).hexdigest()[:16]
            else:
                self._dependency_hash = "unknown"
        except Exception:
            self._dependency_hash = "unknown"
        return self._dependency_hash

    def get_version_info(self) -> Dict[str, Any]:
        """
        Get complete version information.

        Returns:
            Dictionary with version information
        """
        return {
            "git_sha": self.get_git_sha(),
            "git_short_sha": self.get_git_short_sha(),
            "build_time": self.get_build_time(),
            "dependency_lock_hash": self.get_dependency_hash(),
            "python_version": self._get_python_version(),
        }

    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys

        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


_version_info: Optional[VersionInfo] = None


def get_version_info() -> VersionInfo:
    """Get or create the global version info instance."""
    global _version_info
    if _version_info is None:
        _version_info = VersionInfo()
    return _version_info


def create_version_routes(app) -> None:
    """
    Register version routes with Flask app.

    Args:
        app: Flask application instance
    """
    from flask import Blueprint, jsonify

    version_bp = Blueprint("version", __name__)

    @version_bp.route("/version", methods=["GET"])
    def version() -> None:
        """
        Get version information.

        Response:
        {
            "git_sha": "abc123...",
            "git_short_sha": "abc123",
            "build_time": "2025-10-01T12:00:00",
            "dependency_lock_hash": "def456...",
            "python_version": "3.11.0"
        }
        """
        version_info = get_version_info()
        return (jsonify(version_info.get_version_info()), 200)

    app.register_blueprint(version_bp)
    return version_bp
