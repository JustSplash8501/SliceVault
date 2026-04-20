# recount3 GTEx v8 Download Notes

Source references:
- recount3 docs: https://rna.recount.bio/docs/quick-access.html
- raw project URL table used here:
  https://raw.githubusercontent.com/LieberInstitute/recount3-docs/master/docs/recount3_raw_project_files_with_default_annotation.csv

This directory contains:
- `gtex.recount_project.MD.gz`: sample-level GTEx metadata from recount3
- `recount3_raw_project_files_with_default_annotation.csv`: full project URL index
- `files/manifest.tsv`: resolved download manifest for this workspace
- `files/<TISSUE>/...`: downloaded GTEx tissue files

Downloader script:
- `scripts/download_recount3_gtex_v8.py`

Splicing-focused full GTEx (all tissues) download command:

```bash
python3 scripts/download_recount3_gtex_v8.py --include jxn_MM,jxn_RR,jxn_ID
```

Resume-safe: re-running the command continues interrupted transfers via `curl -C -`.
