"""Benchmark native serial MACE-MH-1 energy and force evaluation."""

import argparse
import json
import pathlib
import statistics
import time

import numpy as np
from ase.build import bulk

from symmetrix import Symmetrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Extracted MACE-MH-1 JSON model")
    parser.add_argument("--sizes", default="1,2,3")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    calculator = Symmetrix(args.model, use_kokkos=False, dtype="float64")
    if not getattr(calculator.evaluator, "uses_mh1_fast_path", False):
        raise RuntimeError("Model does not match the specialized MACE-MH-1 CPU architecture")

    results = []
    for size in (int(value) for value in args.sizes.split(",")):
        atoms = bulk("Si", "diamond", a=5.43).repeat((size,) * 3)
        inputs = calculator._mace_inputs(atoms)
        native_args = (*inputs[:5], inputs[5].flatten(), inputs[6])
        calculator.evaluator.compute_node_energies_forces(*native_args)
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            calculator.evaluator.compute_node_energies_forces(*native_args)
            samples.append(1000.0 * (time.perf_counter() - start))
        record = {
            "atoms": len(atoms),
            "directed_edges": len(inputs[6]),
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "samples_ms": samples,
            "energy_eV": float(np.sum(calculator.evaluator.node_energies)),
        }
        results.append(record)
        print(json.dumps(record), flush=True)

    if args.output:
        args.output.write_text(json.dumps({"model": args.model, "systems": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
