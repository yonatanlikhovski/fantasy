from pathlib import Path
import shutil
import pandas as pd

REPORT = "duplicate_player_files.csv"
BACKUP_ROOT = Path("data/duplicate_backups")


def score_file(path: Path):
    """
    Higher score = file we prefer to keep.

    Priority:
    1. More rows
    2. File with real-looking Basketball Reference id, not guessed broken id
    3. Newer modified time
    """
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        rows = len(df)
        cols = len(df.columns)
    except Exception:
        rows = -1
        cols = -1

    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0

    # Prefer files with more data
    return (rows, cols, mtime)


def backup_path_for(original: Path):
    rel = original.relative_to(Path("."))
    return BACKUP_ROOT / rel


def main():
    report = pd.read_csv(REPORT)

    if report.empty:
        print("No duplicates in duplicate_player_files.csv. Nothing to do.")
        return

    needed = {"player_norm", "season", "file"}
    missing_cols = needed - set(report.columns)
    if missing_cols:
        raise ValueError(f"{REPORT} is missing columns: {missing_cols}")

    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

    moved_rows = []
    kept_rows = []

    for (player_norm, season), group in report.groupby(["player_norm", "season"]):
        files = [Path(p) for p in group["file"].tolist()]
        existing_files = [p for p in files if p.exists()]

        if len(existing_files) <= 1:
            print(f"[SKIP] {player_norm} {season}: only one existing file")
            continue

        scored = [(score_file(p), p) for p in existing_files]
        scored.sort(reverse=True)

        keep = scored[0][1]
        delete_candidates = [p for _, p in scored[1:]]

        print("=" * 80)
        print(f"PLAYER: {group['player'].iloc[0] if 'player' in group.columns else player_norm}")
        print(f"SEASON: {season}")
        print(f"KEEP: {keep}")

        kept_rows.append({
            "player_norm": player_norm,
            "season": season,
            "kept_file": str(keep),
            "kept_score": str(score_file(keep)),
        })

        for path in delete_candidates:
            dst = backup_path_for(path)
            dst.parent.mkdir(parents=True, exist_ok=True)

            print(f"MOVE DUPLICATE: {path} -> {dst}")

            shutil.move(str(path), str(dst))

            moved_rows.append({
                "player_norm": player_norm,
                "season": season,
                "moved_from": str(path),
                "moved_to": str(dst),
                "kept_file": str(keep),
            })

    pd.DataFrame(kept_rows).to_csv("duplicates_kept.csv", index=False, encoding="utf-8")
    pd.DataFrame(moved_rows).to_csv("duplicates_moved_to_backup.csv", index=False, encoding="utf-8")

    print()
    print(f"Moved duplicate files: {len(moved_rows)}")
    print("Backup folder:")
    print(f"  {BACKUP_ROOT}")
    print("Reports written:")
    print("  duplicates_kept.csv")
    print("  duplicates_moved_to_backup.csv")


if __name__ == "__main__":
    main()