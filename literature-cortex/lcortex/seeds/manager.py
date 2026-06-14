"""Seed library manager.

Manages the built-in seed library and user extensions. Seeds are the initial
knowledge nodes that bootstrap the literature cortex — covering foundational
concepts across L1 (axioms) through L4 (physical).

Usage:
    from lcortex.seeds.manager import init_seeds, list_seeds
    init_seeds("/path/to/vault")
    seeds = list_seeds("/path/to/vault")
"""

import json
import shutil
from pathlib import Path
from typing import Optional

# ── Built-in seed directory ────────────────────────────────────────────────
_BUILTIN_SEEDS = Path(__file__).resolve().parent


def init_seeds(vault_path: str) -> dict:
    """Initialize seeds in a vault directory.

    Copies built-in seed files (meta/ + L1-L4/) into the vault.
    Returns a summary dict of what was copied.
    """
    vault = Path(vault_path)
    seeds_target = vault / "seeds"

    summary = {"meta": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0, "total": 0}

    # Copy meta directory
    meta_src = _BUILTIN_SEEDS / "meta"
    if meta_src.exists():
        meta_target = seeds_target / "meta"
        meta_target.mkdir(parents=True, exist_ok=True)
        for f in meta_src.glob("*.json"):
            shutil.copy2(str(f), str(meta_target / f.name))
            summary["meta"] += 1
            summary["total"] += 1

    # Copy level directories
    for level in ["L1", "L2", "L3", "L4"]:
        level_src = _BUILTIN_SEEDS / level
        if not level_src.exists():
            continue
        level_target = seeds_target / level
        level_target.mkdir(parents=True, exist_ok=True)
        for f in level_src.glob("*.md"):
            shutil.copy2(str(f), str(level_target / f.name))
            summary[level] += 1
            summary["total"] += 1

    # Also create user seeds directory
    user_target = seeds_target / "user"
    user_target.mkdir(parents=True, exist_ok=True)

    # Copy seeds/README.md if it exists
    readme_src = _BUILTIN_SEEDS / "README.md"
    if readme_src.exists():
        shutil.copy2(str(readme_src), str(seeds_target / "README.md"))

    return summary


def load_user_seeds(user_seeds_dir: str, vault_path: str) -> int:
    """Merge user seeds from a directory into the vault.

    Rules:
      - User seed with same name as built-in → user version takes precedence
      - User seed with new name → appended to corresponding level
      - Files in user_seeds_dir/ are expected to be .md with YAML frontmatter

    Returns:
        Number of user seeds merged.
    """
    user_src = Path(user_seeds_dir)
    vault_seeds = Path(vault_path) / "seeds"
    user_target = vault_seeds / "user"

    if not user_src.exists():
        return 0

    count = 0
    user_target.mkdir(parents=True, exist_ok=True)

    for f in user_src.glob("**/*.md"):
        if f.name.startswith("."):
            continue
        dest = user_target / f.name
        shutil.copy2(str(f), str(dest))
        count += 1

    return count


def list_seeds(vault_path: str, level_filter: Optional[str] = None) -> list[dict]:
    """List all available seeds, organized by knowledge level.

    Returns a list of {name, level, path, builtin} dicts sorted by level.
    """
    vault_seeds = Path(vault_path) / "seeds"
    results = []

    if not vault_seeds.exists():
        return results

    # Built-in seeds
    for level_dir in sorted(vault_seeds.iterdir()):
        if not level_dir.is_dir():
            continue
        level_name = level_dir.name.upper()
        if not level_name.startswith("L") and level_name not in ("META", "USER"):
            continue
        if level_filter and level_name != level_filter.upper():
            continue

        for f in sorted(level_dir.glob("*.md")):
            label = level_name if level_name.startswith("L") else f"builtin/{level_dir.name}"
            results.append({
                "name": f.stem,
                "level": label,
                "path": str(f),
                "builtin": level_dir.name != "user",
            })

    # User seeds
    user_dir = vault_seeds / "user"
    if user_dir.exists() and (not level_filter or level_filter.upper() == "USER"):
        for f in sorted(user_dir.glob("*.md")):
            results.append({
                "name": f.stem,
                "level": "user",
                "path": str(f),
                "builtin": False,
            })

    return results


def get_seed_path(vault_path: str, name: str) -> Optional[Path]:
    """Find a seed file by name. Searches user seeds first, then built-in."""
    vault_seeds = Path(vault_path) / "seeds"

    for search in ["user", ""]:
        d = vault_seeds / search if search else vault_seeds
        if not d.exists():
            continue
        for f in d.rglob(f"{name}.md"):
            return f
        for f in d.rglob(f"{name}"):
            if f.suffix == ".md":
                return f

    return None
