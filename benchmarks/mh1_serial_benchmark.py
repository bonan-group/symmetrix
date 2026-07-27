"""Benchmark native serial or Kokkos CPU MACE-MH-1 execution."""

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
BLAS_THREAD_COUNT = os.environ.get("SYMMETRIX_BENCHMARK_BLAS_THREADS", "1")
KOKKOS_THREAD_VARIABLES = (
    "KOKKOS_NUM_THREADS",
    "OMP_NUM_THREADS",
)
BLAS_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for variable in KOKKOS_THREAD_VARIABLES:
    os.environ[variable] = THREAD_COUNT
for variable in BLAS_THREAD_VARIABLES:
    os.environ[variable] = BLAS_THREAD_COUNT
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")
THREAD_VARIABLES = KOKKOS_THREAD_VARIABLES + BLAS_THREAD_VARIABLES + (
    "OMP_PROC_BIND",
    "OMP_PLACES",
)

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
        "Kokkos_ENABLE_CUDA",
        "Kokkos_ENABLE_HIP",
        "Kokkos_ENABLE_SYCL",
        "Kokkos_ENABLE_OPENMPTARGET",
        "KokkosKernels_ENABLE_TPL_BLAS",
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


def _thread_affinities():
    task_directory = pathlib.Path("/proc/self/task")
    if not task_directory.is_dir():
        return None
    affinities = {}
    for task in task_directory.iterdir():
        status = task / "status"
        try:
            lines = status.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("Cpus_allowed_list"):
                affinities[task.name] = line.partition(":")[2].strip()
                break
    return affinities


def _current_rss_bytes():
    statm = pathlib.Path("/proc/self/statm")
    if not statm.is_file():
        raise RuntimeError("lifecycle RSS measurement requires Linux /proc/self/statm")
    resident_pages = int(statm.read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _samples_summary(samples, atom_count):
    return {
        "median_ms": statistics.median(samples),
        "median_ms_per_atom": statistics.median(samples) / atom_count,
        "min_ms": min(samples),
        "min_ms_per_atom": min(samples) / atom_count,
        "max_ms": max(samples),
        "max_ms_per_atom": max(samples) / atom_count,
        "samples_ms": samples,
    }


def _peak_rss_bytes():
    status = pathlib.Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def _gpu_process_memory_mib():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    used_mib = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            try:
                used_mib += int(fields[1])
            except ValueError:
                return None
    return used_mib


def _benchmark(calculator, atoms, warmups, repeats, include_ase):
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], inputs[5].flatten(), inputs[6])

    rss_before = _current_rss_bytes()
    cuda = (
        calculator.use_kokkos
        and native_symmetrix._kokkos_default_execution_space() == "Cuda"
    )
    gpu_memory_before = _gpu_process_memory_mib() if cuda else None
    for _ in range(warmups):
        calculator.evaluator.compute_node_energies_forces(*native_args)
    evaluator_samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        calculator.evaluator.compute_node_energies_forces(*native_args)
        evaluator_samples.append(1000.0 * (time.perf_counter() - start))
    evaluator_results = calculator._collect_mace_results(atoms, inputs)

    ase_samples = []
    if include_ase:
        for _ in range(warmups):
            calculator.calculate(atoms.copy(), properties=["energy", "forces"])
        for _ in range(repeats):
            start = time.perf_counter()
            calculator.calculate(atoms.copy(), properties=["energy", "forces"])
            ase_samples.append(1000.0 * (time.perf_counter() - start))

    edge_workspace_bytes = getattr(
        calculator.evaluator, "edge_workspace_bytes", None
    )
    return {
        "atoms": len(atoms),
        "directed_edges": len(inputs[6]),
        "evaluator": _samples_summary(evaluator_samples, len(atoms)),
        "ase": _samples_summary(ase_samples, len(atoms)) if ase_samples else None,
        "rss_before_mib": rss_before / 2**20,
        "rss_after_mib": _current_rss_bytes() / 2**20,
        "peak_rss_mib": (
            _peak_rss_bytes() / 2**20 if _peak_rss_bytes() is not None else None
        ),
        "gpu_process_memory_before_mib": gpu_memory_before,
        "gpu_process_memory_after_mib": _gpu_process_memory_mib() if cuda else None,
        "edge_workspace_rows": calculator.evaluator.edge_workspace_rows,
        "edge_workspace_bytes": edge_workspace_bytes,
        "edge_workspace_mib": (
            edge_workspace_bytes / 2**20
            if edge_workspace_bytes is not None else None
        ),
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


def _parse_sizes(value, parser, option):
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError:
        parser.error(f"{option} must be a comma-separated integer list")
    if not sizes or any(size < 1 for size in sizes):
        parser.error(f"{option} values must be positive")
    return sizes


def _lifecycle(calculator, sizes, cycles):
    systems = [bulk("Si", "diamond", a=5.43).repeat((size,) * 3) for size in sizes]
    for _ in range(3):
        for atoms in systems:
            calculator.calculate(atoms.copy(), properties=["energy", "forces", "stress"])
    rss_bytes = []
    for _ in range(cycles):
        for atoms in systems:
            calculator.calculate(atoms.copy(), properties=["energy", "forces", "stress"])
        rss_bytes.append(_current_rss_bytes())
    return {
        "sizes": sizes,
        "atoms": [len(atoms) for atoms in systems],
        "cycles": cycles,
        "evaluations": cycles * len(systems),
        "rss_mib": [value / 2**20 for value in rss_bytes],
        "min_rss_mib": min(rss_bytes) / 2**20,
        "max_rss_mib": max(rss_bytes) / 2**20,
        "growth_mib": (max(rss_bytes) - min(rss_bytes)) / 2**20,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path, help="Extracted MACE-MH-1 JSON model")
    parser.add_argument("--backend", choices=("serial", "kokkos"), default="serial")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--reference-model", type=pathlib.Path, help="Optional native MH-0 JSON")
    parser.add_argument("--max-reference-ratio", type=float)
    parser.add_argument(
        "--cpu",
        type=int,
        help="Single CPU to pin (equivalent to a one-entry --cpus list)",
    )
    parser.add_argument(
        "--cpus",
        help="Comma-separated CPU affinity for Kokkos threads",
    )
    parser.add_argument("--sizes", default="1,2,3")
    parser.add_argument(
        "--atom-counts",
        help="Comma-separated exact AlN atom counts from 256,864,4000",
    )
    parser.add_argument(
        "--streamed-edges",
        choices=("legacy", "r1", "all"),
        default="all",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--evaluator-only", action="store_true")
    parser.add_argument(
        "--allow-generic",
        action="store_true",
        help="Allow the generic nonlinear evaluator for a control measurement",
    )
    parser.add_argument(
        "--lifecycle-sizes",
        help="Comma-separated supercell sizes to alternate for RSS measurement",
    )
    parser.add_argument("--lifecycle-cycles", type=int, default=20)
    parser.add_argument("--max-lifecycle-growth-mib", type=float)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats must be positive")
    if args.atom_counts is not None and args.sizes != "1,2,3":
        parser.error("--atom-counts and an explicit --sizes are mutually exclusive")
    if args.lifecycle_cycles < 1:
        parser.error("--lifecycle-cycles must be positive")
    if args.max_lifecycle_growth_mib is not None and args.lifecycle_sizes is None:
        parser.error("--max-lifecycle-growth-mib requires --lifecycle-sizes")
    if args.max_reference_ratio is not None and args.reference_model is None:
        parser.error("--max-reference-ratio requires --reference-model")
    if args.cpu is not None and args.cpus is not None:
        parser.error("--cpu and --cpus are mutually exclusive")
    if hasattr(os, "sched_getaffinity"):
        available_cpus = os.sched_getaffinity(0)
        requested_cpus = None
        if args.cpu is not None:
            requested_cpus = {args.cpu}
        elif args.cpus is not None:
            try:
                requested_cpus = {int(value) for value in args.cpus.split(",")}
            except ValueError:
                parser.error("--cpus must be a comma-separated integer list")
            if not requested_cpus:
                parser.error("--cpus cannot be empty")
        if requested_cpus is not None:
            try:
                os.sched_setaffinity(0, requested_cpus)
            except OSError as error:
                parser.error(f"could not apply requested CPU affinity: {error}")
            applied_cpus = os.sched_getaffinity(0)
            if applied_cpus != requested_cpus:
                parser.error(
                    f"requested CPUs {sorted(requested_cpus)} produced affinity "
                    f"{sorted(applied_cpus)}"
                )
        elif len(available_cpus) != int(THREAD_COUNT):
            parser.error(
                "use --cpu/--cpus or launch under taskset with exactly "
                f"{THREAD_COUNT} available CPUs"
            )

    use_kokkos = args.backend == "kokkos"
    calculator = Symmetrix(
        args.model,
        use_kokkos=use_kokkos,
        dtype=args.dtype,
        streamed_edges=args.streamed_edges,
    )
    if (
        not args.allow_generic
        and not getattr(calculator.evaluator, "uses_mh1_fast_path", False)
    ):
        raise RuntimeError(
            f"Model does not match the specialized MACE-MH-1 {args.backend} architecture"
        )
    reference = (
        Symmetrix(args.reference_model, use_kokkos=use_kokkos, dtype=args.dtype)
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
        "thread_affinities": _thread_affinities(),
        "backend": args.backend,
        "dtype": args.dtype,
        "scalar_size_bytes": getattr(calculator.evaluator, "scalar_size_bytes", 8),
        "uses_mh1_fast_path": bool(calculator.evaluator.uses_mh1_fast_path),
        "streamed_edges": calculator.streamed_edges,
        "warmups": args.warmups,
        "repeats": args.repeats,
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

    if args.atom_counts is None:
        systems = [
            bulk("Si", "diamond", a=5.43).repeat((size,) * 3)
            for size in _parse_sizes(args.sizes, parser, "--sizes")
        ]
    else:
        atom_counts = _parse_sizes(args.atom_counts, parser, "--atom-counts")
        repeat_by_atoms = {256: 4, 864: 6, 4000: 10}
        invalid = [count for count in atom_counts if count not in repeat_by_atoms]
        if invalid:
            parser.error(f"unsupported --atom-counts values: {invalid}")
        systems = [
            bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat(
                (repeat_by_atoms[count],) * 3
            )
            for count in atom_counts
        ]

    results = []
    for atoms in systems:
        record = _benchmark(
            calculator, atoms, args.warmups, args.repeats, not args.evaluator_only
        )
        if reference is not None:
            reference_record = _benchmark(
                reference, atoms, args.warmups, args.repeats, not args.evaluator_only
            )
            record["reference"] = reference_record
            record["evaluator_reference_ratio"] = (
                record["evaluator"]["median_ms"]
                / reference_record["evaluator"]["median_ms"]
            )
            record["ase_reference_ratio"] = (
                record["ase"]["median_ms"] / reference_record["ase"]["median_ms"]
                if record["ase"] is not None else None
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

    lifecycle = None
    if args.lifecycle_sizes is not None:
        lifecycle = _lifecycle(
            calculator,
            _parse_sizes(args.lifecycle_sizes, parser, "--lifecycle-sizes"),
            args.lifecycle_cycles,
        )
        print(json.dumps({"lifecycle": lifecycle}), flush=True)
        if (
            args.max_lifecycle_growth_mib is not None
            and lifecycle["growth_mib"] > args.max_lifecycle_growth_mib
        ):
            raise RuntimeError(
                f"lifecycle RSS growth {lifecycle['growth_mib']:.3f} MiB exceeds "
                f"{args.max_lifecycle_growth_mib:.3f} MiB"
            )

    output = {"metadata": metadata, "systems": results, "lifecycle": lifecycle}
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
