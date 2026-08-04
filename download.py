"""Google Drive checkpoint fetching with on-disk caching.

Weights live on Drive as "anyone with the link" shares. The first time a
checkpoint is needed it is downloaded into ``models/`` and reused from there
for every later run, so nothing is re-fetched across sessions.
"""

import os
import re

import gdown

MODELS_DIR = "models"

# Matches the file ID in the two Drive URL shapes people paste:
#   https://drive.google.com/file/d/<ID>/view?usp=sharing
#   https://drive.google.com/uc?id=<ID>
_ID_PATTERNS = [
    re.compile(r"/file/d/([A-Za-z0-9_-]+)"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]+)"),
]

# A bare file ID: no slashes, no scheme, Drive's alphabet only.
_BARE_ID = re.compile(r"^[A-Za-z0-9_-]{10,}$")


def extract_file_id(url_or_id):
    """Accept either a full Drive share URL or a bare file ID, return the ID."""
    value = url_or_id.strip()
    for pattern in _ID_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1)
    if _BARE_ID.match(value):
        return value
    raise ValueError(
        f"Could not read a Google Drive file ID out of {url_or_id!r}. "
        "Pass a share URL like https://drive.google.com/file/d/<ID>/view "
        "or the bare file ID."
    )


def local_path(filename):
    return os.path.join(MODELS_DIR, filename)


def is_cached(filename):
    path = local_path(filename)
    return os.path.exists(path) and os.path.getsize(path) > 0


def ensure_checkpoint(url_or_id, filename, progress_hook=None):
    """Return a local path to ``filename``, downloading it only if missing.

    ``progress_hook`` is called with a human-readable message when a download
    actually starts, so the UI can show a specific spinner instead of a
    generic one.
    """
    path = local_path(filename)
    if is_cached(filename):
        return path

    os.makedirs(MODELS_DIR, exist_ok=True)
    if progress_hook is not None:
        progress_hook(f"Downloading {filename}…")

    file_id = extract_file_id(url_or_id)
    gdown.download(id=file_id, output=path, quiet=False)

    if not is_cached(filename):
        raise RuntimeError(
            f"Download of {filename} (Drive id {file_id}) produced no file. "
            "Check that the share link is set to 'anyone with the link'."
        )
    return path
