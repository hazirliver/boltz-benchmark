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
    import yaml
    import requests
    import contextlib
    return (
        BaseModel,
        Literal,
        Path,
        contextlib,
        json,
        mo,
        os,
        product,
        requests,
        time,
        tqdm,
        yaml,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Constants
    """)
    return


@app.cell
def _(Path, os):
    BOLTZ2_NIM_URL = os.environ.get(
        "BOLTZ2_NIM_URL",
        "http://localhost:8000/biology/mit/boltz2/predict",
    )

    ROOT_DIR = Path(os.environ["ROOT_DIR"])
    SMALL_MOLECULES_DIR = ROOT_DIR / "data" / "small_molecules"
    RESULTS_FILE = ROOT_DIR / "results" / "boltz2_nim_benchmark_results.jsonl"
    OUT_DIR_BASE = ROOT_DIR / "boltz2_nim_benchmark"
    return BOLTZ2_NIM_URL, OUT_DIR_BASE, RESULTS_FILE, SMALL_MOLECULES_DIR


@app.cell
def _():
    RECYCLING_STEPS = [1, 2]
    SAMPLING_STEPS = [10, 20]
    DIFFUSION_SAMPLES = [1, 2]
    SAMPLING_STEPS_AFFINITY = [10, 20]
    DIFFUSION_SAMPLES_AFFINITY = [1, 2]

    BATCH_SIZE = ["NIM"]
    return (
        BATCH_SIZE,
        DIFFUSION_SAMPLES,
        DIFFUSION_SAMPLES_AFFINITY,
        RECYCLING_STEPS,
        SAMPLING_STEPS,
        SAMPLING_STEPS_AFFINITY,
    )


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

        batch_size: int | str = "NIM"
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
def _(Path, yaml):
    def cli_yaml_to_nim_payload(yaml_path: Path) -> dict:
        with open(yaml_path, "r") as f:
            cli_obj = yaml.safe_load(f)

        polymers: list[dict] = []
        ligands: list[dict] = []

        sequences = cli_obj.get("sequences", [])

        for entry in sequences:
            if "protein" in entry:
                p = entry["protein"]
                seq = p.get("sequence")
                pid = p.get("id", None)

                poly: dict = {
                    "molecule_type": "protein",
                    "sequence": seq,
                }
                if pid:
                    poly["id"] = pid

                msa_path = p.get("msa")
                if msa_path:
                    a3m_text = Path(msa_path).read_text()
                    poly["msa"] = {
                        "user": {
                            "a3m": {
                                "format": "a3m",
                                "alignment": a3m_text,
                                "rank": 0,
                            }
                        }
                    }
                polymers.append(poly)

            if "ligand" in entry:
                l = entry["ligand"]
                smi = l.get("smiles")
                lid = l.get("id", None)
                ligand: dict = {}
                if lid:
                    ligand["id"] = lid

                ligand["smiles"] = smi
                ligands.append(ligand)

        payload: dict = {"polymers": polymers}
        if ligands:
            payload["ligands"] = ligands

        return payload
    return (cli_yaml_to_nim_payload,)


@app.cell
def _(Boltz2Params, Path, cli_yaml_to_nim_payload, contextlib, requests, time):
    def call_boltz2_nim(
        boltz2_nim_url: str,
        small_molecules_dir: str,
        params: Boltz2Params,
    ) -> float:
        headers = {"content-type": "application/json"}
        out_dir = Path(params.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        allowed_nim_params = {
            "recycling_steps",
            "sampling_steps",
            "diffusion_samples",
            "step_scale",
            "output_format",
            "sampling_steps_affinity",
            "diffusion_samples_affinity",
        }

        start = time.perf_counter()

        for yaml_file in Path(small_molecules_dir).glob("*.yaml"):
            base_payload = cli_yaml_to_nim_payload(yaml_file)

            # Dump all params, then filter to allowed NIM keys only
            param_dict = params.model_dump()
            nim_param_dict = {
                k: v for k, v in param_dict.items()
                if k in allowed_nim_params
            }

            # Merge into request body
            base_payload.update(nim_param_dict)

            resp = requests.post(
                boltz2_nim_url,
                headers=headers,
                json=base_payload,
                timeout=1800,
            )

            # If something is still off, this will show you the server message
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                # Helpful debug print; remove or log as needed
                print(f"Request failed for {yaml_file.name}")
                print("Status:", resp.status_code)
                with contextlib.suppress(Exception):
                    print("Response:", resp.json())
                raise

            result = resp.json()

            # Persist returned structures
            structures = result.get("structures", [])
            for i, structure in enumerate(structures):
                content = structure.get("structure")
                if not content:
                    continue
                name = structure.get("name") or f"{yaml_file.stem}_{i}"
                suffix = "cif" if params.output_format == "mmcif" else params.output_format
                (out_dir / f"{name}.{suffix}").write_text(content)

        end = time.perf_counter()
        return end - start
    return (call_boltz2_nim,)


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
    BOLTZ2_NIM_URL,
    Boltz2Params,
    OUT_DIR_BASE,
    RESULTS_FILE,
    SMALL_MOLECULES_DIR,
    all_params,
    call_boltz2_nim,
    done_ids,
    json,
    tqdm,
):
    for i, p in tqdm(enumerate(all_params, 1), total=len(all_params)):
        run_id = f"{p['recycling_steps']}_{p['sampling_steps']}_{p['diffusion_samples']}_{p['sampling_steps_affinity']}_{p['diffusion_samples_affinity']}_{p['batch_size']}"
        if run_id in done_ids:
            continue

        params = Boltz2Params(**p)
        params.out_dir = OUT_DIR_BASE / run_id
        runtime = call_boltz2_nim(boltz2_nim_url=str(BOLTZ2_NIM_URL),
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
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

