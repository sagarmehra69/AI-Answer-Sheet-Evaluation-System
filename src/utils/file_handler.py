# src/utils/file_handler.py

import os
import shutil
import json
from pathlib import Path
from typing import List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def list_images(folder: str, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    """
    Return sorted list of all image file paths in a folder.

    Args:
        folder: Directory to scan
        extensions: Allowed image extensions

    Returns:
        List of full file paths
    """
    if not os.path.exists(folder):
        logger.warning(f"Folder not found: {folder}")
        return []

    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(extensions)
    ]
    logger.info(f"Found {len(files)} images in {folder}")
    return files


def save_json(data: dict, path: str) -> None:
    """Save a dictionary as a JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON → {path}")


def load_json(path: str) -> Optional[dict]:
    """Load a JSON file. Returns None if file not found."""
    if not os.path.exists(path):
        logger.error(f"JSON file not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_file(src: str, dst: str) -> None:
    """Copy file from src to dst, creating parent dirs as needed."""
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
    logger.debug(f"Copied {src} → {dst}")


def get_filename(path: str, with_ext: bool = True) -> str:
    """Extract filename from full path."""
    base = os.path.basename(path)
    return base if with_ext else os.path.splitext(base)[0]
