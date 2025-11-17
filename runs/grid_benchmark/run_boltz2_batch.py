import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    from typing import Literal
    import polars as pl
    import os
    import subprocess
    from pydantic import BaseModel
    from pathlib import Path
    from itertools import product
    from tqdm import tqdm
    import time
    import json
    return (
        BaseModel,
        Literal,
        Path,
        json,
        mo,
        os,
        pl,
        product,
        subprocess,
        time,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Constants
    """)
    return


@app.cell
def _(Path, os):
    BOLTZ2_EXECUTABLE = Path(os.environ["BOLTZ2_EXECUTABLE"])

    ROOT_DIR = Path(os.environ["ROOT_DIR"])
    SMALL_MOLECULES_DIR = ROOT_DIR / "data" / "small_molecules"
    RESULTS_FILE = ROOT_DIR / "results" / "boltz2_batch_benchmark_results.jsonl"
    OUT_DIR_BASE = ROOT_DIR / "boltz2_batch_benchmark"
    return BOLTZ2_EXECUTABLE, OUT_DIR_BASE, RESULTS_FILE, SMALL_MOLECULES_DIR


@app.cell
def _():
    # RECYCLING_STEPS = [1, 3, 6]
    # SAMPLING_STEPS = [50, 200, 600]
    # DIFFUSION_SAMPLES = [1, 3, 5]
    # SAMPLING_STEPS_AFFINITY = [500, 200, 600]
    # DIFFUSION_SAMPLES_AFFINITY = [1, 5, 10]

    RECYCLING_STEPS = [1, 2]
    SAMPLING_STEPS = [10, 20]
    DIFFUSION_SAMPLES = [1, 2]
    SAMPLING_STEPS_AFFINITY = [10, 20]
    DIFFUSION_SAMPLES_AFFINITY = [1, 2]

    BATCH_SIZE = ["4", "8"]
    return (
        BATCH_SIZE,
        DIFFUSION_SAMPLES,
        DIFFUSION_SAMPLES_AFFINITY,
        RECYCLING_STEPS,
        SAMPLING_STEPS,
        SAMPLING_STEPS_AFFINITY,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Functions
    """)
    return


@app.cell
def _(BaseModel, Literal, OUT_DIR_BASE):
    class Boltz2Params(BaseModel):
        out_dir: str = OUT_DIR_BASE
        devices: int = 1
        accelerator: Literal["gpu", "cpu", "tpu"] = "gpu"
        recycling_steps: int = 3
        sampling_steps: int = 200
        diffusion_samples: int = 1
        max_parallel_samples: int = 5
        step_scale: float = 1.638
        output_format: Literal["pdb", "mmcif"] = "mmcif"
        num_workers: int = 2
        sampling_steps_affinity: int = 200
        diffusion_samples_affinity: int = 5
        override: bool = True
        use_msa_server: bool = False
        msa_server_url: str = "https://api.colabfold.com"

        batch_size: int = 1
    return (Boltz2Params,)


@app.cell
def _(RESULTS_FILE, json):
    def load_completed() -> set[str]:
        if not RESULTS_FILE.exists():
            return set()
        with open(RESULTS_FILE, "r") as f:
            return {json.loads(line)["id"] for line in f}
    return (load_completed,)


@app.cell
def _(
    BATCH_SIZE,
    DIFFUSION_SAMPLES,
    DIFFUSION_SAMPLES_AFFINITY,
    RECYCLING_STEPS,
    SAMPLING_STEPS,
    SAMPLING_STEPS_AFFINITY,
    product,
):
    def param_grid() -> tuple[dict[str, int], ...]:
        return tuple(
            {
                "recycling_steps": r,
                "sampling_steps": s,
                "diffusion_samples": d,
                "sampling_steps_affinity": sa,
                "diffusion_samples_affinity": da,
                "batch_size": bs
            }
            for r, s, d, sa, da, bs in product(
                RECYCLING_STEPS,
                SAMPLING_STEPS,
                DIFFUSION_SAMPLES,
                SAMPLING_STEPS_AFFINITY,
                DIFFUSION_SAMPLES_AFFINITY,
                BATCH_SIZE
            )
        )
    return (param_grid,)


@app.cell
def _(Boltz2Params, subprocess, time):
    def run_boltz2(boltz2_executalbe: str, small_molecules_dir: str, params: Boltz2Params) -> float:
        # Build command arguments
        cmd_args = [boltz2_executalbe]
        cmd_args.extend(["predict"])

        # Add parameters from Boltz2Params model
        cmd_args.extend(["--out_dir", params.out_dir])
        cmd_args.extend(["--devices", str(params.devices)])
        cmd_args.extend(["--accelerator", params.accelerator])
        cmd_args.extend(["--recycling_steps", str(params.recycling_steps)])
        cmd_args.extend(["--sampling_steps", str(params.sampling_steps)])
        cmd_args.extend(["--diffusion_samples", str(params.diffusion_samples)])
        cmd_args.extend(["--max_parallel_samples", str(params.max_parallel_samples)])
        cmd_args.extend(["--step_scale", str(params.step_scale)])
        cmd_args.extend(["--output_format", params.output_format])
        cmd_args.extend(["--num_workers", str(params.num_workers)])
        cmd_args.extend(["--sampling_steps_affinity", str(params.sampling_steps_affinity)])
        cmd_args.extend(["--diffusion_samples_affinity", str(params.diffusion_samples_affinity)])

        # Handle boolean flags
        if params.override:
            cmd_args.append("--override")
        if params.use_msa_server:
            cmd_args.append("--use_msa_server")
            cmd_args.extend(["--msa_server_url", params.msa_server_url])

        # Enable batch mode
        if int(params.batch_size) > 1:
            cmd_args.extend(["--screening_mode"])
            cmd_args.extend(["--batch_size", str(params.batch_size)])

        # Add input YAML
        cmd_args.append(small_molecules_dir)

        start = time.perf_counter()
        subprocess.run(cmd_args, check=True)
        end = time.perf_counter()

        return end - start
    return (run_boltz2,)


@app.cell
def _(mo):
    mo.md(r"""
    # Run
    """)
    return


@app.cell
def _(RESULTS_FILE, load_completed, param_grid):
    RESULTS_FILE.touch(exist_ok=True)

    all_params = param_grid()
    done_ids = load_completed()

    print(f"Number of combinations to check: {len(all_params)}; {len(done_ids)} is already done; {len(all_params) - len(done_ids)} to complete")
    return all_params, done_ids


@app.cell
def _(
    BOLTZ2_EXECUTABLE,
    Boltz2Params,
    OUT_DIR_BASE,
    RESULTS_FILE,
    SMALL_MOLECULES_DIR,
    all_params,
    done_ids,
    json,
    run_boltz2,
    tqdm,
):
    for i, p in tqdm(enumerate(all_params, 1), total=len(all_params)):
        run_id = f"{p['recycling_steps']}_{p['sampling_steps']}_{p['diffusion_samples']}_{p['sampling_steps_affinity']}_{p['diffusion_samples_affinity']}_{p['batch_size']}"
        if run_id in done_ids:
            continue

        params = Boltz2Params(**p)
        params.out_dir = OUT_DIR_BASE / run_id
        runtime = run_boltz2(boltz2_executalbe=str(BOLTZ2_EXECUTABLE),
                             small_molecules_dir=str(SMALL_MOLECULES_DIR),
                             params=params)
        result = {
            "id": run_id,
            "params": p,
            "runtime_sec": runtime,
        }
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")
    return


@app.cell
def _(RESULTS_FILE, pl):
    pl.read_ndjson(RESULTS_FILE).unnest('params')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

