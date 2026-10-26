# Data moves from ../tmlr_paper (2026-10-20)

Same filesystem (/work, NFS4 vast): `mv` = rename, no data copied. inode/size/mtime of all 449 files were
recorded before (`.stat_before.txt`) and after (`.stat_after.txt`) the move and are identical. SHA-256
checksums are computed afterwards into `llama2.json`.

| From (../tmlr_paper/) | To (this repo) |
|---|---|
| outputs/data/fineweb_edu_llama2/ | data/llama2/ |
| outputs/data/tok/ | data/llama2_tok/ |
| outputs/data/eval/triviaqa_v2/ | data/eval/triviaqa_v2/ |
| outputs/data/raw/fineweb_edu (symlink) | data/raw/fineweb_edu -> /work/wac20/ctrl_tokens/text_conditioned/data/fineweb_edu_raw/sample/100BT (new symlink; raw data belongs to ctrl_tokens and was not moved) |

Aliases added in data/llama2/: validation.{bin,docs.npy} -> analysis.*, test.{bin,docs.npy} -> val.* (see MIGRATION_REPORT.md §1).
