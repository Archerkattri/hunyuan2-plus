"""Dependency-free helpers that keep user-influenced filesystem paths safe.

The API server receives task ids and mesh-type parameters from remote
callers, so those values must be canonicalized before they are used to
build paths under the output directory (CodeQL `py/path-injection`
hardening).
"""

import os
import uuid

# Mesh formats a caller may request via the "type" parameter. Only these
# constants may flow into export paths / temporary-file suffixes.
ALLOWED_MESH_TYPES = ("glb", "gltf", "obj", "ply", "stl")


def validated_mesh_type(value):
    """Return the requested mesh type if allowlisted, else raise ValueError.

    The returned string always comes from ALLOWED_MESH_TYPES rather than
    from the caller's input, so only known-safe constants reach the
    filesystem.
    """
    for allowed in ALLOWED_MESH_TYPES:
        if value == allowed:
            return allowed
    raise ValueError(f"Unsupported mesh type: {value!r}")


def validated_uid(value):
    """Return the canonical form of a task id, else raise ValueError.

    Task ids are UUIDs minted server-side; anything that does not parse as
    a UUID is rejected. The canonical form contains only hex digits and
    hyphens, so it cannot carry path separators or traversal sequences.
    """
    return str(uuid.UUID(str(value)))


def resolve_output_path(save_dir, filename):
    """Join *filename* under *save_dir*, rejecting path traversal.

    Returns the absolute path when it stays strictly within *save_dir*,
    else raises ValueError.
    """
    base = os.path.realpath(save_dir)
    path = os.path.realpath(os.path.join(base, filename))
    if not path.startswith(base + os.sep):
        raise ValueError(f"Invalid filename: {filename!r}")
    return path
