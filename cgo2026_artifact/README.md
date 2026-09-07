# CGO 2026 artifact:

Scores three detectors (FPChecker, EFTSanitizer, NSan) against a brtrace
ground-truth census on LULESH, AMG, QuickSilver and the NAS Parallel Benchmarks,
and reproduces Table 4 and Appendix Tables 11-12 of the paper.

Requirements: Docker or Podman, x86-64 Linux host, ~15 GB disk, network access
during the build (GitHub, conda-forge, repo.anaconda.com). Everything runs single-threaded; more cores do not help.

## 1. Build the image (~30 min)

    git clone --branch v0.7_new_runtime_branch_flip --single-branch https://github.com/llnl/FPChecker.git
    cd FPChecker/cgo2026_artifact
    docker build -t cgo2026-artifact .
    # Podman (the --format flag is required):
    podman build --format docker -t cgo2026-artifact .

The build clones FPChecker and EFTSanitizer (pinned commit, patched) inside
the image, creates three conda environments (LLVM 19.1.7 for FPChecker,
brtrace and NSan; LLVM 10 for EFTSanitizer), builds the four tools and the
NSan compiler-rt runtime, and ends with `Successfully tagged`. The base image
is Rocky Linux 8.

Podman troubleshooting:
- `SHELL is not supported`: add `--format docker`.
- `setgroups failed` or ownership errors during the build (rootless Podman
  without subordinate UIDs): put

        [storage.options.overlay]
        ignore_chown_errors = "true"

  in `~/.config/containers/storage.conf`.
- `Peer certificate cannot be authenticated` (TLS-inspecting proxy): add
  `-v /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:ro`
  to the build command.

## 2. Start a container

    mkdir -p results
    docker run -it --rm -v $PWD/results:/opt/cgo2026_artifact/branch_flip/results cgo2026-artifact
    # Podman: append :Z to the volume, i.e. ...results:Z

This opens a shell inside the container in `/opt/cgo2026_artifact/branch_flip`.
`results/` on the host is mounted there, so everything the run writes survives
after the container exits. `--rm` removes the container itself on exit; the
image stays.

## 3. Run

Inside the container:

    bash run_experiments.sh --main      # paper Table 4  (~3 h)
    bash run_experiments.sh --full      # appendix Tables 11-12, plus Table 4  (~10 h)

Each run: checks that the four tools are built, runs the brtrace census,
runs each tool (EFTSanitizer and FPChecker for `--main`; NSan as well for
`--full`), scores it, and prints the table(s) with a `match`/`MISMATCH`
status per row. The terminal shows one line per step, e.g.

    == FPChecker  (0h16m elapsed)
       LULESH                                    done   147s
       AMG                                       done   168s
       ...
    RESULT: OK -- all results match expected

Full output of every step is in `results/run.log`. To leave a long run
unattended, start the container under `tmux`/`screen`, or use
`docker run -d` and `docker logs -f`.

If you want something else:

    bash run_experiments.sh --full --quick          # skip QuickSilver and NAS SP (~2 h)
    bash run_experiments.sh --full --bench amg      # one benchmark: lulesh amg quicksilver bt cg ep is lu mg sp
    bash run_experiments.sh --main --bench lu       # quickest end-to-end check (~1 min)
    bash run_experiments.sh --tools fpchecker --skip-gt   # one tool, reuse the census of a previous run

## 4. Compare with the paper

Files in `results/` after a run:

| file | content |
|---|---|
| `table_main.{txt,tex,json}` | Table 4: EFTSanitizer and FPChecker (interval rule, eta=1e-6), FP32 |
| `table_full.{txt,tex,json}` | Tables 11 (FP32) and 12 (FP64): EFTSanitizer, NSan, FPChecker interval rule at three eta and shadow rule |
| `{fpc,eftsan,nsan}_metrics.json` | scorer output per tool |
| `compare.txt` | the row-by-row check |
| `run.log` | everything the harnesses printed |

Every table row is compared with the reference results committed in
`fpchecker_bf/cpu_checking/error_analysis/branch_flip/expected/`: `match`
means all four counts (TP, FP, TN, FN) are identical; anything else is
`MISMATCH`. The `.tex` files are the tables in the paper's column order.

To rebuild the tables from the JSONs without rerunning anything:

    python3 branch_flip_tables.py --main --full          # reads results/
    python3 branch_flip_tables.py --score --full         # rerun the scorers first

## What the tables show

TP/FP/TN/FN are counted per branch execution, joined on
(module_id, site_id, execution index) with the census. `Crashed` marks
EFTSanitizer's build failure on QuickSilver FP32. NAS IS has no
floating-point-controlled branches and is all zeros.

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