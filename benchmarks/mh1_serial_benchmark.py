"""Benchmark native serial MACE-MH-1 evaluator and ASE execution."""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time


THREAD_COUNT = os.environ.get("SYMMETRIX_BENCHMARK_THREADS", "1")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for variable in THREAD_VARIABLES:
    os.environ[variable] = THREAD_COUNT

import numpy as np
from ase.build import bulk

from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _cpu_model():
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or None


def _native_build_metadata(extension_path):
    cache = extension_path.parent / "CMakeCache.txt"
    keys = {
        "CMAKE_BUILD_TYPE",
        "SYMMETRIX_KOKKOS",
        "SYMMETRIX_SPHERICART_CUDA",
        "Kokkos_ENABLE_OPENMP",
        "Kokkos_ENABLE_SERIAL",
    }
    values = {}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if not line or line.startswith(("#", "//")) or "=" not in line:
                continue
            declaration, value = line.split("=", 1)
            key = declaration.split(":", 1)[0]
            if key in keys:
                values[key] = value
    return {"cmake_cache": str(cache) if cache.is_file() else None, "values": values}


def _samples_summary(samples):
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _benchmark(calculator, atoms, repeats):
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], inputs[5].flatten(), inputs[6])

    calculator.evaluator.compute_node_energies_forces(*native_args)
    evaluator_samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        calculator.evaluator.compute_node_energies_forces(*native_args)
        evaluator_samples.append(1000.0 * (time.perf_counter() - start))
    evaluator_results = calculator._collect_mace_results(atoms, inputs)

    calculator.calculate(atoms.copy(), properties=["energy", "forces"])
    ase_samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        calculator.calculate(atoms.copy(), properties=["energy", "forces"])
        ase_samples.append(1000.0 * (time.perf_counter() - start))

    return {
        "atoms": len(atoms),
        "directed_edges": len(inputs[6]),
        "evaluator": _samples_summary(evaluator_samples),
        "ase": _samples_summary(ase_samples),
        "energy_eV": float(evaluator_results["energy"]),
        "force_l2_eV_per_A": float(np.linalg.norm(evaluator_results["forces"])),
        "force_sum_eV_per_A": np.sum(evaluator_results["forces"], axis=0).tolist(),
    }


def _model_metadata(path):
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path, help="Extracted MACE-MH-1 JSON model")
    parser.add_argument("--reference-model", type=pathlib.Path, help="Optional native MH-0 JSON")
    parser.add_argument("--max-reference-ratio", type=float)
    parser.add_argument(
        "--cpu",
        type=int,
        help="CPU to pin; required unless taskset already restricts the process to one CPU",
    )
    parser.add_argument("--sizes", default="1,2,3")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.max_reference_ratio is not None and args.reference_model is None:
        parser.error("--max-reference-ratio requires --reference-model")
    if hasattr(os, "sched_getaffinity"):
        available_cpus = os.sched_getaffinity(0)
        if args.cpu is not None:
            if args.cpu not in available_cpus:
                parser.error(f"--cpu {args.cpu} is outside the current affinity {sorted(available_cpus)}")
            os.sched_setaffinity(0, {args.cpu})
        elif len(available_cpus) != 1:
            parser.error("use --cpu or launch under taskset to pin this benchmark to one CPU")

    calculator = Symmetrix(args.model, use_kokkos=False, dtype="float64")
    if not getattr(calculator.evaluator, "uses_mh1_fast_path", False):
        raise RuntimeError("Model does not match the specialized MACE-MH-1 CPU architecture")
    reference = (
        Symmetrix(args.reference_model, use_kokkos=False, dtype="float64")
        if args.reference_model else None
    )

    extension_path = pathlib.Path(native_symmetrix.__file__).resolve()
    metadata = {
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "native_extension": str(extension_path),
        "native_build": _native_build_metadata(extension_path),
        "model": _model_metadata(args.model),
        "reference_model": _model_metadata(args.reference_model) if args.reference_model else None,
        "timing_scope": {
            "evaluator": "prebuilt graph; native energy and analytic-force kernel only",
            "ase": "Symmetrix.calculate including graph construction and result collection",
        },
    }

    results = []
    for size in (int(value) for value in args.sizes.split(",")):
        atoms = bulk("Si", "diamond", a=5.43).repeat((size,) * 3)
        record = _benchmark(calculator, atoms, args.repeats)
        if reference is not None:
            reference_record = _benchmark(reference, atoms, args.repeats)
            record["reference"] = reference_record
            record["evaluator_reference_ratio"] = (
                record["evaluator"]["median_ms"]
                / reference_record["evaluator"]["median_ms"]
            )
            record["ase_reference_ratio"] = (
                record["ase"]["median_ms"] / reference_record["ase"]["median_ms"]
            )
            if (
                args.max_reference_ratio is not None
                and record["evaluator_reference_ratio"] > args.max_reference_ratio
            ):
                raise RuntimeError(
                    f"MH-1/MH-0 evaluator ratio {record['evaluator_reference_ratio']:.3f} "
                    f"exceeds {args.max_reference_ratio:.3f} for {len(atoms)} atoms"
                )
        results.append(record)
        print(json.dumps(record), flush=True)

    output = {"metadata": metadata, "systems": results}
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
