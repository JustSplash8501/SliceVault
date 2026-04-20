from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download full GTEx v8 recount3 project files")
    parser.add_argument(
        "--table",
        default="data/raw/recount3/gtex_v8/recount3_raw_project_files_with_default_annotation.csv",
        help="Path to recount3 project URL table CSV",
    )
    parser.add_argument(
        "--outdir",
        default="data/raw/recount3/gtex_v8/files",
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--include",
        default="gene,exon,jxn_MM,jxn_RR,jxn_ID",
        help="Comma-separated columns to download",
    )
    parser.add_argument(
        "--limit-projects",
        type=int,
        default=0,
        help="Optional cap on number of tissue projects for testing (0=all)",
    )
    args = parser.parse_args()

    table = Path(args.table)
    if not table.exists():
        raise FileNotFoundError(f"Table not found: {table}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(table)
    gtex = df[
        (df["organism"] == "human")
        & (df["file_source"] == "gtex")
        & (df["project_type"] == "data_sources")
    ].copy()

    gtex = gtex.sort_values("project")
    if args.limit_projects > 0:
        gtex = gtex.head(args.limit_projects)

    include_cols = [x.strip() for x in args.include.split(",") if x.strip()]

    manifest_records: list[dict[str, str]] = []
    for _, row in gtex.iterrows():
        project = str(row["project"])
        project_dir = outdir / project
        project_dir.mkdir(parents=True, exist_ok=True)

        for col in include_cols:
            url = str(row[col])
            fname = url.rsplit("/", 1)[-1]
            dest = project_dir / fname
            manifest_records.append(
                {
                    "project": project,
                    "asset": col,
                    "url": url,
                    "local_path": str(dest),
                }
            )

    manifest = pd.DataFrame(manifest_records)
    manifest_path = outdir / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    print(f"Wrote manifest: {manifest_path} ({len(manifest)} files)", flush=True)

    for rec in manifest_records:
        dest = Path(rec["local_path"])
        # Always run with -C - so interrupted downloads are resumed and complete
        # files are quickly validated by the server via Content-Range handling.
        run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(dest), rec["url"]])

    print("Download complete", flush=True)


if __name__ == "__main__":
    main()
