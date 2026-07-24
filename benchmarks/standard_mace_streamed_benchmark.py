"""Benchmark standard compact MACE streamed-edge modes."""

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time


THREAD_COUNT = os.environ.get("SYMMETRIX_BENCHMARK_THREADS", "1")
BLAS_THREAD_COUNT = os.environ.get("SYMMETRIX_BENCHMARK_BLAS_THREADS", "1")
for variable in ("KOKKOS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = THREAD_COUNT
for variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = BLAS_THREAD_COUNT
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")

import numpy as np
from ase.build import bulk

from symmetrix import Symmetrix


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_csv(value, cast, option, parser):
    try:
        parsed = [cast(item) for item in value.split(",")]
    except ValueError:
        parser.error(f"{option} contains an invalid value")
    if not parsed:
        parser.error(f"{option} cannot be empty")
    return parsed


def _summary(samples):
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _evaluate(model, atoms, backend, dtype, mode, warmups, repeats):
    calculator = Symmetrix(
        model,
        use_kokkos=backend == "kokkos",
        dtype=dtype,
        streamed_edges=mode,
    )
    if not calculator.evaluator.supports_streamed_edges:
        raise RuntimeError("The model is not an ordinary format-v2 compact MACE model")
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], inputs[5].flatten(), inputs[6])
    for _ in range(warmups):
        calculator.evaluator.compute_node_energies_forces(*native_args)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        calculator.evaluator.compute_node_energies_forces(*native_args)
        samples.append(1000.0*(time.perf_counter()-start))
    results = calculator._collect_mace_results(atoms, inputs)
    scalar_bytes = 4 if dtype == "float32" else 8
    r0_elements = int(calculator.evaluator.R0_storage_size)
    r1_elements = int(calculator.evaluator.R1_storage_size)
    return {
        "mode": mode,
        "directed_edges": len(inputs[6]),
        "timing": _summary(samples),
        "energy_eV": float(results["energy"]),
        "forces_eV_per_A": np.asarray(results["forces"]).tolist(),
        "edge_radial_storage": {
            "R0_elements": r0_elements,
            "R1_elements": r1_elements,
            "bytes": scalar_bytes*(r0_elements+r1_elements),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--backend", choices=("serial", "kokkos"), default="kokkos")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--modes", default="legacy,r1,all")
    parser.add_argument("--sizes", default="2,3,4")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-force-error", type=float, default=2e-5)
    parser.add_argument("--min-speedup", type=float)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.backend == "serial" and args.dtype != "float64":
        parser.error("the serial backend supports only float64")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats must be positive")

    modes = _parse_csv(args.modes, str, "--modes", parser)
    if any(mode not in ("legacy", "r1", "all") for mode in modes):
        parser.error("--modes must contain only legacy,r1,all")
    if "legacy" not in modes:
        parser.error("--modes must include legacy as the reference")
    sizes = _parse_csv(args.sizes, int, "--sizes", parser)
    if any(size < 1 for size in sizes):
        parser.error("--sizes values must be positive")

    model = args.model.resolve()
    report = {
        "model": {
            "path": str(model),
            "size_bytes": model.stat().st_size,
            "sha256": _sha256(model),
        },
        "backend": args.backend,
        "dtype": args.dtype,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "KOKKOS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
            )
        },
        "systems": [],
    }
    failed = False
    for size in sizes:
        atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((size, size, size))
        records = {
            mode: _evaluate(
                model,
                atoms,
                args.backend,
                args.dtype,
                mode,
                args.warmups,
                args.repeats,
            )
            for mode in modes
        }
        reference = records["legacy"]
        reference_forces = np.asarray(reference["forces_eV_per_A"])
        directed_edges = reference["directed_edges"]
        for mode, record in records.items():
            forces = np.asarray(record.pop("forces_eV_per_A"))
            record.pop("directed_edges")
            record["energy_error_eV"] = abs(record["energy_eV"]-reference["energy_eV"])
            record["force_max_error_eV_per_A"] = float(
                np.max(np.abs(forces-reference_forces))
            )
            record["speedup_vs_legacy"] = (
                reference["timing"]["median_ms"]/record["timing"]["median_ms"]
            )
            failed |= record["force_max_error_eV_per_A"] > args.max_force_error
            if args.min_speedup is not None and mode != "legacy":
                failed |= record["speedup_vs_legacy"] < args.min_speedup
        report["systems"].append(
            {
                "supercell_repeat": size,
                "atoms": len(atoms),
                "directed_edges": directed_edges,
                "modes": records,
            }
        )

    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.write_text(serialized+"\n")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
