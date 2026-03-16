from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from afrc import AnalyticalFRC
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as IP
from finches.frontend.calvados_frontend import CALVADOS_frontend
from finches.frontend.mpipi_frontend import Mpipi_frontend
from scipy import fftpack
from sparrow import Protein
from sparrow.predictors import batch_predict


NETWORKS = [
    "rg",
    "scaled_rg",
    "re",
    "scaled_re",
    "asphericity",
    "scaling_exponent",
    "prefactor",
]

MF = Mpipi_frontend(salt=0.5, dielectric=80)
CF = CALVADOS_frontend(salt=0.5, pH=8)


def _extract_prediction_value(value: object) -> float:
    if isinstance(value, (tuple, list, np.ndarray)):
        if len(value) > 1:
            return float(value[1])
        if len(value) == 1:
            return float(value[0])
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert prediction value: {value}") from exc


def grid_score(matrix: np.ndarray) -> float:
    fft2 = np.abs(fftpack.fft2(matrix))
    fft2[0, 0] = 0
    fft2_shifted = fftpack.fftshift(fft2)

    center_margin = max(3, fft2.shape[0] // 10)
    h, w = fft2_shifted.shape
    center_mask = np.ones_like(fft2_shifted, dtype=bool)
    center_mask[
        h // 2 - center_margin : h // 2 + center_margin,
        w // 2 - center_margin : w // 2 + center_margin,
    ] = False

    masked_fft = np.where(center_mask, fft2_shifted, 0)
    peak_values = np.sort(masked_fft.flatten())[-10:]

    non_zero = masked_fft[masked_fft > 0]
    mean_value = np.mean(non_zero) if len(non_zero) > 0 else 0
    peak_ratio = (
        np.sum(peak_values) / (mean_value * len(peak_values)) if mean_value > 0 else 0
    )
    peak_ratio /= matrix.shape[0]
    return float(peak_ratio)


def sticker_distance(seq: str, stickers: str = "FYW") -> float:
    sticker_positions = [i for i, aa in enumerate(seq) if aa in stickers]
    distances = [
        sticker_positions[i + 1] - sticker_positions[i]
        for i in range(len(sticker_positions) - 1)
    ]
    if not distances:
        return 0.0
    return float(sum(distances) / len(distances))


def rgre_cal(seq_dict: Mapping[str, str]) -> pd.DataFrame:
    all_predictions: dict[str, dict[str, float]] = {
        seq_id: {} for seq_id in seq_dict.keys()
    }

    for network in NETWORKS:
        return_dict = batch_predict.batch_predict(seq_dict, network=network)
        for seq_id, prediction in return_dict.items():
            all_predictions[seq_id][network] = _extract_prediction_value(prediction)

    rg_pred_res = pd.DataFrame.from_dict(all_predictions, orient="index")

    predictions: dict[str, tuple[float, float]] = {}
    for seq_name, seq in seq_dict.items():
        analytical = AnalyticalFRC(seq)
        mean_rg = analytical.get_mean_radius_of_gyration()
        mean_re = analytical.get_mean_end_to_end_distance()
        predictions[seq_name] = (float(mean_rg), float(mean_re))

    afrc_pred = pd.DataFrame.from_dict(
        predictions, orient="index", columns=["mean_rg", "mean_re"]
    )

    data_merge = pd.merge(rg_pred_res, afrc_pred, left_index=True, right_index=True)
    data_merge["norm_rg"] = data_merge["rg"] / data_merge["mean_rg"]
    data_merge["norm_re"] = data_merge["re"] / data_merge["mean_re"]
    return data_merge


def epsilon_cal(seq_dict: Mapping[str, str]) -> pd.DataFrame:
    predictions: dict[str, tuple[float, float, float]] = {}
    for seq_name, seq in seq_dict.items():
        epsilon_mf = MF.epsilon(seq, seq)
        epsilon_cf = CF.epsilon(seq, seq)
        fourier_peakratio = grid_score(MF.intermolecular_idr_matrix(seq, seq, window_size=1)[0][0])
        predictions[seq_name] = (float(epsilon_mf), float(epsilon_cf), float(fourier_peakratio))

    epsilon_pred = pd.DataFrame.from_dict(
        predictions,
        orient="index",
        columns=["epsilon_mf", "epsilon_cf", "fourier_peakratio"],
    )
    return epsilon_pred


def property_cal(seq_dict: Mapping[str, str]) -> pd.DataFrame:
    predictions: dict[str, dict[str, float]] = {}
    for seq_name, seq in seq_dict.items():
        protein = Protein(seq)
        pi_p = IP(seq)
        predictions[seq_name] = {
            "FCR": float(protein.FCR),
            "NCPR": float(protein.NCPR),
            "fraction_aliphatic": float(protein.fraction_aliphatic),
            "fraction_aromatic": float(protein.fraction_aromatic),
            "fraction_positive": float(protein.fraction_positive),
            "fraction_negative": float(protein.fraction_negative),
            "fraction_polar": float(protein.fraction_polar),
            "fraction_proline": float(protein.fraction_proline),
            "hydrophobicity": float(protein.hydrophobicity),
            "pI": float(pi_p.pi()),
        }

    return pd.DataFrame.from_dict(predictions, orient="index")


def _normalize_sequence(seq: str) -> str:
    return str(seq).strip().replace(" ", "")


def _ensure_name_column(data: pd.DataFrame, name_col: str) -> pd.DataFrame:
    if name_col not in data.columns:
        data[name_col] = [f"seq_{i+1}" for i in range(len(data))]
    else:
        data[name_col] = data[name_col].astype(str)

    if data[name_col].duplicated().any():
        counts = data.groupby(name_col).cumcount()
        data[name_col] = np.where(counts > 0, data[name_col] + "_" + counts.astype(str), data[name_col])
    return data


def feature_generation(
    data: pd.DataFrame,
    seq_col: str = "Seq",
    name_col: str = "name",
) -> pd.DataFrame:
    if seq_col not in data.columns:
        raise ValueError(f"Missing sequence column: {seq_col}")

    merged = data.copy()
    merged = _ensure_name_column(merged, name_col)

    merged["sequence"] = merged[seq_col].apply(_normalize_sequence)
    merged["length"] = merged["sequence"].apply(len)

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for aa in amino_acids:
        merged[aa] = merged["sequence"].apply(lambda x: x.count(aa))
        merged[f"frac_{aa}"] = merged[aa] / merged["length"]

    merged["sticker_distance"] = merged["sequence"].apply(sticker_distance)

    input_seqs = dict(zip(merged[name_col], merged["sequence"], strict=True))
    data_rgre = rgre_cal(input_seqs)
    data_epsilon = epsilon_cal(input_seqs)
    data_property = property_cal(input_seqs)

    merged = (
        merged.merge(data_rgre, left_on=name_col, right_index=True)
        .merge(data_epsilon, left_on=name_col, right_index=True)
        .merge(data_property, left_on=name_col, right_index=True)
    )
    return merged
