# CGO 2026 artifact: floating-point branch-flip detection

Scores three detectors (FPChecker, EFTSanitizer, NSan) against a brtrace
ground-truth census on LULESH, AMG, QuickSilver and the NAS Parallel Benchmarks,
and reproduces Table 4 and Appendix Tables 11-12 of the paper.

## 1. Build the image (~30 min)

    cd cgo2026_artifact
    docker build -t cgo2026-artifact .
    # Podman:
    podman build --format docker -t cgo2026-artifact .

The image is Rocky Linux 8 (glibc 2.28). The build clones FPChecker
(branch `v0.7_new_runtime_branch_flip`) and EFTSanitizer (pinned commit,
patched), creates three conda environments (LLVM 19.1.7 for FPChecker,
brtrace and NSan; LLVM 10 for EFTSanitizer), builds the four tools and the
NSan compiler-rt runtime, and runs an NSan self-test.

Alternatively load the prebuilt image: `docker load -i cgo2026-artifact.tar.gz`.

Podman notes: `--format docker` is required (the Dockerfile uses `SHELL`).
Rootless Podman without subordinate UIDs needs
`ignore_chown_errors = "true"` under `[storage.options.overlay]` in
`~/.config/containers/storage.conf`. Behind a TLS-inspecting proxy add
`-v /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:ro`
to the build command.

## 2. Run the experiments

    mkdir results
    docker run -it --rm -v $PWD/results:/opt/cgo2026_artifact/branch_flip/results cgo2026-artifact
    # Podman: add :Z to the volume

Inside the container (`/opt/cgo2026_artifact/branch_flip`):

    bash run_experiments.sh --main      # paper Table 4, compared row by row with the reference (~3 h)
    bash run_experiments.sh --full      # appendix Tables 11-12 (+ Table 4), compared the same way (~10 h)

Both run the brtrace census first, then each tool (EFTSanitizer + FPChecker
for `--main`; NSan too for `--full`), score it, print the table(s) with a
`match`/`MISMATCH` status per row, and end with `RESULT: OK` when every row
matches. The terminal shows one line per step; full output is in
`results/run.log`.

If you want something else:

    bash run_experiments.sh --full --quick          # skip QuickSilver and NAS SP (~2 h)
    bash run_experiments.sh --full --bench amg      # one benchmark: lulesh amg quicksilver bt cg ep is lu mg sp
    bash run_experiments.sh --tools fpchecker --skip-gt   # one tool, reuse an existing census

## 3. Compare with the paper

The last step prints the table(s) and writes to `results/`:

| file | content |
|---|---|
| `table_main.{txt,tex,json}` | Table 4 (FP32, EFTSanitizer and FPChecker interval rule at eta=1e-6) |
| `table_full.{txt,tex,json}` | Tables 11 (FP32) and 12 (FP64): EFTSanitizer, NSan, FPChecker interval rule at three eta and shadow rule |
| `{fpc,eftsan,nsan}_metrics.json` | scorer output per tool |
| `compare.txt` | row-by-row check against the committed reference |

Every row carries a status: `match` (all four counts identical to the
reference in `fpchecker_bf/.../branch_flip/expected/`) or `MISMATCH`.
The run ends with

    RESULT: OK -- all results match expected

Tables can be rebuilt without rerunning:

    python3 branch_flip_tables.py --main --full     # from results/
    python3 branch_flip_tables.py --score --full    # rerun the scorers first

## What the tables show

TP/FP/TN/FN are counted per branch execution, joined on
(module_id, site_id, execution index) with the census. 
`Crashed` marks EFTSanitizer's build failure on QuickSilver
FP32. NAS IS has no floating-point-controlled branches and is all zeros.

## Layout inside the container

    /opt/cgo2026_artifact/branch_flip/          run_experiments.sh, branch_flip_tables.py, results/
    /opt/cgo2026_artifact/fpchecker_bf/         FPChecker repo
      cpu_checking/error_analysis/env_setup/    create_*/activate_* environment scripts
      cpu_checking/error_analysis/branch_flip/
        benchmarks/                             fp32 / fp64 / long double source trees
        expected/                               reference metrics
        experiments/gt_experiments/             brtrace, census harnesses, scorers
        experiments/{fpchecker,eftsan,nsan}_experiments/   per-tool harnesses
    /opt/cgo2026_artifact/EFTSanitizer/         pinned commit + patch