"""Benchmark compact MACEField execution through native CPU or Kokkos."""

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import threading
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
from symmetrix import symmetrix as native_symmetrix


ATOM_COUNT_TO_REPEAT = {256: 4, 864: 6, 4000: 10}


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


def _per_atom_summary(samples, atom_count):
    return _summary([sample / atom_count for sample in samples])


def _difference_summary(actual, reference):
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    reference = np.asarray(reference, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(difference * difference))),
        "relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-300)
        ),
    }


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
    found = False
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            try:
                used_mib += int(fields[1])
                found = True
            except ValueError:
                return None
    return used_mib if found else 0


def _process_memory_mib():
    values = {"current": None, "peak": None}
    try:
        with pathlib.Path("/proc/self/status").open() as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key == "VmRSS":
                    values["current"] = int(value.split()[0]) / 1024
                elif key == "VmHWM":
                    values["peak"] = int(value.split()[0]) / 1024
    except (OSError, ValueError):
        pass
    return values


class _GpuMemorySampler:
    def __init__(self, interval):
        self.interval = interval
        self.peak_mib = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.is_set():
            value = _gpu_process_memory_mib()
            if value is not None:
                self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()


def _field_arguments(calculator, atoms, electric_field):
    inputs = calculator._mace_inputs(atoms)
    return inputs, (
        *inputs[:5],
        inputs[5].flatten(),
        inputs[6],
        electric_field,
    )


def _evaluate(
    model,
    atoms,
    backend,
    dtype,
    mode,
    electric_field,
    warmups,
    repeats,
    memory_sample_interval,
    responses,
    finite_difference_control,
    finite_difference_step,
    finite_difference_scheme,
):
    process_memory_before = _process_memory_mib()
    gpu_memory_before = _gpu_process_memory_mib()
    calculator = Symmetrix(
        model,
        use_kokkos=backend == "kokkos",
        dtype=dtype,
        streamed_edges=mode,
    )
    if not calculator.evaluator.has_field_coupling:
        raise RuntimeError("The benchmark requires a field-aware MACEField model")
    inputs, native_args = _field_arguments(calculator, atoms, electric_field)
    process_memory_after_setup = _process_memory_mib()
    gpu_memory_after_setup = _gpu_process_memory_mib()

    for _ in range(warmups):
        calculator.evaluator.compute_node_energies_forces_field(*native_args)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        calculator.evaluator.compute_node_energies_forces_field(*native_args)
        samples.append(1000.0 * (time.perf_counter() - start))

    with _GpuMemorySampler(memory_sample_interval) as sampler:
        calculator.evaluator.compute_node_energies_forces_field(*native_args)

    results = calculator._collect_mace_results(atoms, inputs)
    field_adjoint = np.asarray(calculator.evaluator.electric_field_adj, dtype=np.float64)
    scalar_bytes = 4 if dtype == "float32" else 8
    r0_elements = int(calculator.evaluator.R0_storage_size)
    r1_elements = int(calculator.evaluator.R1_storage_size)

    response_timings = {}
    response_agreement = {}
    analytic_hessian = None
    analytic_force_derivative = None
    if responses:
        for name in ("field_hessian", "field_force_derivative"):
            method = (
                calculator.evaluator.compute_electric_field_hessian
                if name == "field_hessian"
                else calculator.evaluator.compute_electric_field_force_derivative
            )
            response_gpu_before = _gpu_process_memory_mib()
            start = time.perf_counter()
            method(*native_args)
            elapsed_ms = 1000.0 * (time.perf_counter() - start)
            with _GpuMemorySampler(memory_sample_interval) as response_sampler:
                method(*native_args)
            response_timings[name] = {
                "elapsed_ms": elapsed_ms,
                "elapsed_ms_per_atom": elapsed_ms / len(atoms),
                "gpu_before_mib": response_gpu_before,
                "gpu_after_mib": _gpu_process_memory_mib(),
                "gpu_sampled_high_water_mib": response_sampler.peak_mib,
            }
        analytic_hessian = np.asarray(
            calculator.evaluator.electric_field_hessian, dtype=np.float64
        ).copy()
        analytic_force_derivative = np.asarray(
            calculator.evaluator.electric_field_force_derivative,
            dtype=np.float64,
        ).copy()
    if finite_difference_control:
        def run_finite_difference_control():
            base_forces = np.asarray(
                calculator.evaluator.node_forces, dtype=np.float64
            ).copy()
            if finite_difference_scheme == "forward":
                calculator.evaluator.compute_node_energies_forces_field(
                    *native_args
                )
                base_adj = np.asarray(
                    calculator.evaluator.electric_field_adj,
                    dtype=np.float64,
                ).copy()
                base_forces = np.asarray(
                    calculator.evaluator.node_forces, dtype=np.float64
                ).copy()
                np.asarray(
                    calculator.evaluator.node_energies, dtype=np.float64
                ).copy()
            else:
                base_adj = None
            hessian = np.empty(9, dtype=np.float64)
            force_derivative = np.empty(
                3 * np.asarray(calculator.evaluator.node_forces).size,
                dtype=np.float64,
            )
            for seed in range(3):
                field_plus = electric_field.copy()
                field_plus[seed] += finite_difference_step
                calculator.evaluator.compute_node_energies_forces_field(
                    *native_args[:-1], field_plus
                )
                adj_plus = np.asarray(
                    calculator.evaluator.electric_field_adj, dtype=np.float64
                ).copy()
                forces_plus = np.asarray(
                    calculator.evaluator.node_forces, dtype=np.float64
                ).copy()
                if finite_difference_scheme == "central":
                    field_minus = electric_field.copy()
                    field_minus[seed] -= finite_difference_step
                    calculator.evaluator.compute_node_energies_forces_field(
                        *native_args[:-1], field_minus
                    )
                    adj_minus = np.asarray(
                        calculator.evaluator.electric_field_adj,
                        dtype=np.float64,
                    ).copy()
                    forces_minus = np.asarray(
                        calculator.evaluator.node_forces, dtype=np.float64
                    ).copy()
                    denominator = 2 * finite_difference_step
                else:
                    adj_minus = base_adj
                    forces_minus = base_forces
                    denominator = finite_difference_step
                hessian[seed::3] = (adj_plus - adj_minus) / denominator
                force_derivative[
                    seed * base_forces.size : (seed + 1) * base_forces.size
                ] = (forces_plus - forces_minus) / denominator
            return hessian, force_derivative
        response_gpu_before = _gpu_process_memory_mib()
        start = time.perf_counter()
        fd_hessian, fd_force_derivative = run_finite_difference_control()
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        with _GpuMemorySampler(memory_sample_interval) as response_sampler:
            run_finite_difference_control()
        response_timings["finite_difference_control"] = {
            "elapsed_ms": elapsed_ms,
            "elapsed_ms_per_atom": elapsed_ms / len(atoms),
            "step": finite_difference_step,
            "scheme": finite_difference_scheme,
            "gpu_before_mib": response_gpu_before,
            "gpu_after_mib": _gpu_process_memory_mib(),
            "gpu_sampled_high_water_mib": response_sampler.peak_mib,
        }
        if analytic_hessian is not None:
            response_agreement = {
                "hessian_analytic_vs_finite_difference": _difference_summary(
                    analytic_hessian, fd_hessian
                ),
                "force_derivative_analytic_vs_finite_difference": (
                    _difference_summary(
                        analytic_force_derivative, fd_force_derivative
                    )
                ),
            }

    force_array = np.asarray(results["forces"], dtype=np.float64)
    process_memory_after_evaluation = _process_memory_mib()
    gpu_memory_after_evaluation = _gpu_process_memory_mib()
    return {
        "mode": mode,
        "atoms": len(atoms),
        "directed_edges": len(inputs[6]),
        "electric_field_V_per_A": electric_field.tolist(),
        "timing": _summary(samples),
        "timing_per_atom": _per_atom_summary(samples, len(atoms)),
        "response_timing_ms": response_timings,
        "response_agreement": response_agreement,
        "energy_eV": float(results["energy"]),
        "force_max_abs_eV_per_A": float(np.max(np.abs(force_array))),
        "force_l2_eV_per_A": float(np.linalg.norm(force_array)),
        "electric_field_adjoint": field_adjoint.tolist(),
        "polarization_per_volume": (-field_adjoint / atoms.get_volume()).tolist(),
        "edge_radial_storage": {
            "R0_elements": r0_elements,
            "R1_elements": r1_elements,
            "bytes": scalar_bytes * (r0_elements + r1_elements),
        },
        "gpu_process_memory_mib": {
            "before_setup": gpu_memory_before,
            "after_setup": gpu_memory_after_setup,
            "after_evaluation": gpu_memory_after_evaluation,
            "sampled_high_water": sampler.peak_mib,
        },
        "process_memory_mib": {
            "before_setup": process_memory_before,
            "after_setup": process_memory_after_setup,
            "after_evaluation": process_memory_after_evaluation,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--backend", choices=("native", "kokkos"), default="kokkos")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--modes", default="legacy,r1,all")
    parser.add_argument("--sizes", default="864")
    parser.add_argument("--electric-field", default="0.01,-0.02,0.03")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--memory-sample-interval", type=float, default=0.02)
    parser.add_argument("--responses", action="store_true")
    parser.add_argument("--finite-difference-control", action="store_true")
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    parser.add_argument(
        "--finite-difference-scheme",
        choices=("forward", "central"),
        default="forward",
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats must be positive")
    if args.memory_sample_interval <= 0:
        parser.error("--memory-sample-interval must be positive")
    if args.finite_difference_step <= 0:
        parser.error("--finite-difference-step must be positive")
    if args.backend == "native" and args.dtype != "float64":
        parser.error("the native CPU evaluator supports only float64")

    modes = _parse_csv(args.modes, str, "--modes", parser)
    if any(mode not in ("legacy", "r1", "all") for mode in modes):
        parser.error("--modes must contain only legacy,r1,all")
    sizes = _parse_csv(args.sizes, int, "--sizes", parser)
    invalid_sizes = [size for size in sizes if size not in ATOM_COUNT_TO_REPEAT]
    if invalid_sizes:
        parser.error(f"unsupported atom counts: {invalid_sizes}; choose 256,864,4000")
    field = np.asarray(
        _parse_csv(args.electric_field, float, "--electric-field", parser),
        dtype=np.float64,
    )
    if field.shape != (3,):
        parser.error("--electric-field must contain exactly three values")

    model = args.model.resolve()
    report = {
        "model": {
            "path": str(model),
            "size_bytes": model.stat().st_size,
            "sha256": _sha256(model),
        },
        "backend": args.backend,
        "execution_space": (
            native_symmetrix._kokkos_default_execution_space()
            if args.backend == "kokkos"
            else "native CPU"
        ),
        "dtype": args.dtype,
        "warmups": args.warmups,
        "repeats": args.repeats,
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

    for atom_count in sizes:
        repeat = ATOM_COUNT_TO_REPEAT[atom_count]
        atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((repeat, repeat, repeat))
        if len(atoms) != atom_count:
            raise RuntimeError(f"expected {atom_count} atoms, built {len(atoms)}")
        records = []
        for mode in modes:
            records.append(
                _evaluate(
                    model,
                    atoms,
                    args.backend,
                    args.dtype,
                    mode,
                    field,
                    args.warmups,
                    args.repeats,
                    args.memory_sample_interval,
                    args.responses,
                    args.finite_difference_control,
                    args.finite_difference_step,
                    args.finite_difference_scheme,
                )
            )
        report["systems"].append({"atoms": atom_count, "records": records})

    output = json.dumps(report, indent=2)
    print(output)
    if args.output is not None:
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
