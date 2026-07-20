"""Serialization for MACE nonlinear-residual interaction models.

The legacy Symmetrix JSON schema stores a heavily fused two-layer MACE graph.
`RealAgnosticResidualNonLinearInteractionBlock` has a different differentiable
graph, so this module records its e3nn metadata and learned tensors directly.
"""

from __future__ import annotations

from typing import Any

import torch
from e3nn import o3


def _tensor(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "values": value.detach().cpu().numpy(force=True).reshape(-1).tolist(),
    }


def _instructions(module) -> list[dict[str, Any]]:
    serialized = []
    for instruction in module.instructions:
        entry = {}
        for name in (
            "i_in",
            "i_in1",
            "i_in2",
            "i_out",
            "connection_mode",
            "has_weight",
            "path_weight",
            "path_shape",
        ):
            if hasattr(instruction, name):
                value = getattr(instruction, name)
                entry[name] = list(value) if isinstance(value, tuple) else value
        serialized.append(entry)
    return serialized


def _linear(module) -> dict[str, Any]:
    return {
        "irreps_in": str(module.irreps_in),
        "irreps_out": str(module.irreps_out),
        "instructions": _instructions(module),
        "weight": _tensor(module.weight),
        "bias": _tensor(module.bias),
        "output_mask": _tensor(module.output_mask),
    }


def _tensor_product(module) -> dict[str, Any]:
    output = {
        "irreps_in1": str(module.irreps_in1),
        "irreps_in2": str(module.irreps_in2),
        "irreps_out": str(module.irreps_out),
        "instructions": _instructions(module),
        "weight": _tensor(module.weight),
        "output_mask": _tensor(module.output_mask),
    }
    # e3nn does not retain its generated Wigner tensors as module buffers.  Store
    # them explicitly so evaluating this schema never imports e3nn at runtime.
    for instruction in output["instructions"]:
        in1 = module.irreps_in1[instruction["i_in1"]].ir.l
        in2 = module.irreps_in2[instruction["i_in2"]].ir.l
        out = module.irreps_out[instruction["i_out"]].ir.l
        instruction["wigner_3j"] = _tensor(
            o3.wigner_3j(in1, in2, out).to(
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
        )
    return output


def _affine_mlp(module) -> dict[str, Any]:
    layers = []
    for layer in module.net:
        if isinstance(layer, torch.nn.Linear):
            layers.append(
                {
                    "type": "linear",
                    "weight": _tensor(layer.weight),
                    "bias": _tensor(layer.bias),
                }
            )
        elif isinstance(layer, torch.nn.LayerNorm):
            layers.append(
                {
                    "type": "layer_norm",
                    "normalized_shape": list(layer.normalized_shape),
                    "eps": float(layer.eps),
                    "weight": _tensor(layer.weight),
                    "bias": _tensor(layer.bias),
                }
            )
        elif isinstance(layer, torch.nn.SiLU):
            layers.append({"type": "silu"})
        else:
            raise RuntimeError(
                "Native nonlinear-MACE extraction only supports Linear, LayerNorm, and SiLU radial MLP layers; "
                f"found {type(layer).__module__}.{type(layer).__name__}."
            )
    return {"layers": layers}


def _interaction(interaction) -> dict[str, Any]:
    scalar_activations = interaction.equivariant_nonlin.act_scalars.acts
    gate_activations = interaction.equivariant_nonlin.act_gates.acts
    if not all(activation.f is torch.nn.functional.silu for activation in scalar_activations):
        raise RuntimeError("Native nonlinear-MACE extraction requires SiLU scalar gate activations.")
    supported_sigmoids = (torch.sigmoid, torch.nn.functional.sigmoid)
    if not all(activation.f in supported_sigmoids for activation in gate_activations):
        raise RuntimeError("Native nonlinear-MACE extraction requires sigmoid tensor gate activations.")
    return {
        "class": type(interaction).__name__,
        "node_feats_irreps": str(interaction.node_feats_irreps),
        "target_irreps": str(interaction.target_irreps),
        "hidden_irreps": str(interaction.hidden_irreps),
        "edge_irreps": str(interaction.edge_irreps),
        "avg_num_neighbors": float(interaction.avg_num_neighbors),
        "alpha": float(interaction.alpha.detach().cpu()),
        "beta": float(interaction.beta.detach().cpu()),
        "source_embedding": _linear(interaction.source_embedding),
        "target_embedding": _linear(interaction.target_embedding),
        "linear_up": _linear(interaction.linear_up),
        "skip_tp": _linear(interaction.skip_tp),
        "linear_res": _linear(interaction.linear_res),
        "linear_1": _linear(interaction.linear_1),
        "linear_2": _linear(interaction.linear_2),
        "conv_tp": _tensor_product(interaction.conv_tp),
        "conv_tp_weights": _affine_mlp(interaction.conv_tp_weights),
        "density_fn": _affine_mlp(interaction.density_fn),
        "gate": {
            "irreps_in": str(interaction.equivariant_nonlin.irreps_in),
            "irreps_out": str(interaction.equivariant_nonlin.irreps_out),
            "irreps_scalars": str(interaction.equivariant_nonlin.irreps_scalars),
            "irreps_gates": str(interaction.equivariant_nonlin.irreps_gates),
            "irreps_gated": str(interaction.equivariant_nonlin.irreps_gated),
            "mul": _tensor_product(interaction.equivariant_nonlin.mul),
            "scalar_activation": "silu",
            "scalar_activation_constants": [
                float(activation.cst)
                for activation in scalar_activations
            ],
            "gate_activation": "sigmoid",
            "gate_activation_constants": [
                float(activation.cst)
                for activation in gate_activations
            ],
        },
    }


def _product(product) -> dict[str, Any]:
    contractions = []
    for contraction in product.symmetric_contractions.contractions:
        contractions.append(
            {
                "correlation": int(contraction.correlation),
                "weights": [_tensor(weight) for weight in contraction.weights],
                "weights_max": _tensor(contraction.weights_max),
                "u_tensors": [
                    _tensor(contraction.U_tensors(nu))
                    for nu in range(1, contraction.correlation + 1)
                ],
            }
        )
    return {
        "node_feats_irreps": str(product.linear.irreps_in),
        "target_irreps": str(product.linear.irreps_out),
        "use_sc": bool(product.use_sc),
        "use_agnostic_product": bool(getattr(product, "use_agnostic_product", False)),
        "linear": _linear(product.linear),
        "symmetric_contractions": {
            "irreps_in": str(product.symmetric_contractions.irreps_in),
            "irreps_out": str(product.symmetric_contractions.irreps_out),
            "contractions": contractions,
        },
    }


def extract_mace_nonlinear_data(model, atomic_numbers: list[int]) -> dict[str, Any]:
    """Return a versioned native schema for nonlinear residual MACE models."""
    from mace.modules.blocks import RealAgnosticResidualNonLinearInteractionBlock

    if not model.interactions or not all(
        isinstance(interaction, RealAgnosticResidualNonLinearInteractionBlock)
        for interaction in model.interactions
    ):
        raise RuntimeError(
            "MACE_Nonlinear JSON requires every interaction to be "
            "RealAgnosticResidualNonLinearInteractionBlock."
        )
    if len(model.products) != len(model.interactions):
        raise RuntimeError("MACE_Nonlinear model has mismatched interaction and product counts.")
    if hasattr(model, "joint_embedding") or hasattr(model, "embedding_readout"):
        raise RuntimeError(
            "Native nonlinear-MACE extraction does not support joint_embedding or embedding_readout."
        )
    if bool(getattr(model, "use_last_readout_only", False)):
        raise RuntimeError(
            "Native nonlinear-MACE extraction does not support use_last_readout_only models."
        )
    if len(model.readouts) != len(model.interactions):
        raise RuntimeError("MACE_Nonlinear model has mismatched interaction and readout counts.")
    if getattr(model, "pair_repulsion", False) and not hasattr(model, "pair_repulsion_fn"):
        raise RuntimeError("MACE_Nonlinear pair repulsion is missing its basis module.")

    model_atomic_numbers = [int(value) for value in model.atomic_numbers.tolist()]
    if not atomic_numbers:
        atomic_numbers = list(model_atomic_numbers)
    model_indices = []
    for atomic_number in atomic_numbers:
        try:
            model_indices.append(model_atomic_numbers.index(atomic_number))
        except ValueError as exc:
            raise ValueError(
                f"Atomic number {atomic_number} is not supported by this MACE checkpoint."
            ) from exc

    radial = model.radial_embedding
    if type(radial.bessel_fn).__name__ != "BesselBasis":
        raise RuntimeError("Native nonlinear-MACE extraction currently requires BesselBasis.")
    if type(radial.cutoff_fn).__name__ != "PolynomialCutoff":
        raise RuntimeError("Native nonlinear-MACE extraction currently requires PolynomialCutoff.")

    distance_transform = {"type": "none"}
    if radial.distance_transform is not None:
        transform = radial.distance_transform
        if type(transform).__name__ != "AgnesiTransform":
            raise RuntimeError("Native nonlinear-MACE extraction currently requires AgnesiTransform or no transform.")
        distance_transform = {
            "type": "agnesi",
            "a": float(transform.a),
            "q": float(transform.q),
            "p": float(transform.p),
            "covalent_radii": [float(value) for value in transform.covalent_radii.numpy(force=True)],
        }

    from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock

    for readout in model.readouts:
        if type(readout) not in (LinearReadoutBlock, NonLinearReadoutBlock):
            raise RuntimeError(
                "Native nonlinear-MACE extraction only supports LinearReadoutBlock and "
                f"NonLinearReadoutBlock; found {type(readout).__name__}."
            )
        if isinstance(readout, NonLinearReadoutBlock) and not all(
            activation.f is torch.nn.functional.silu
            for activation in readout.non_linearity.acts
        ):
            raise RuntimeError(
                "Native nonlinear-MACE extraction requires SiLU nonlinear readouts."
            )

    output = {
        "symmetrix_format_version": 3,
        "model_type": "MACE_Nonlinear",
        "head": model.heads[0] if hasattr(model, "heads") and len(model.heads) == 1 else None,
        "atomic_numbers": atomic_numbers,
        "model_atomic_numbers": model_atomic_numbers,
        "model_indices": model_indices,
        "num_elements": len(atomic_numbers),
        "r_cut": float(model.r_max),
        "l_max": int(model.spherical_harmonics._lmax),
        "num_interactions": len(model.interactions),
        "atomic_energies": _tensor(model.atomic_energies_fn.atomic_energies),
        "scale_shift": {
            "scale": _tensor(model.scale_shift.scale),
            "shift": _tensor(model.scale_shift.shift),
        },
        "node_embedding": _linear(model.node_embedding.linear),
        "radial_embedding": {
            "apply_cutoff": bool(getattr(radial, "apply_cutoff", True)),
            "basis": {
                "type": "bessel",
                "weights": _tensor(radial.bessel_fn.bessel_weights),
                "prefactor": float(radial.bessel_fn.prefactor),
            },
            "cutoff": {
                "type": "polynomial",
                "r_max": float(radial.cutoff_fn.r_max),
                "p": int(radial.cutoff_fn.p),
            },
            "distance_transform": distance_transform,
        },
        "interactions": [_interaction(interaction) for interaction in model.interactions],
        "products": [_product(product) for product in model.products],
        "readouts": [
            {
                "class": type(readout).__name__,
                "linear": _linear(readout.linear)
                if hasattr(readout, "linear")
                else None,
                "linear_1": _linear(readout.linear_1)
                if hasattr(readout, "linear_1")
                else None,
                "linear_2": _linear(readout.linear_2)
                if hasattr(readout, "linear_2")
                else None,
                "activation": "silu" if hasattr(readout, "non_linearity") else None,
                "activation_constants": [
                    float(activation.cst)
                    for activation in readout.non_linearity.acts
                ] if hasattr(readout, "non_linearity") else None,
            }
            for readout in model.readouts
        ],
        "has_zbl": bool(getattr(model, "pair_repulsion", False)),
    }
    if output["has_zbl"]:
        zbl = model.pair_repulsion_fn
        output["zbl"] = {
            "a_exp": float(zbl.a_exp),
            "a_prefactor": float(zbl.a_prefactor),
            "c": _tensor(zbl.c),
            "covalent_radii": _tensor(zbl.covalent_radii),
            "p": _tensor(zbl.p),
        }
    return output
