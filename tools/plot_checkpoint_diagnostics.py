"""Plot training diagnostics directly from PyTorch ZIP checkpoints.

This intentionally avoids importing torch so it can run in lightweight analysis
environments. It supports the tensor/storage subset used by the MAICC checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import pickle
import re
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("i1"),
    "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("?"),
}


class StorageType:
    def __init__(self, name: str):
        self.__name__ = name


class Storage:
    def __init__(self, array: np.ndarray):
        self.array = array


class TorchZipUnpickler(pickle.Unpickler):
    def __init__(self, data: bytes, archive: zipfile.ZipFile, prefix: str):
        super().__init__(io.BytesIO(data))
        self.archive = archive
        self.prefix = prefix

    def find_class(self, module: str, name: str):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return StorageType(name)
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return self._rebuild_tensor
        return super().find_class(module, name)

    def persistent_load(self, pid):
        kind, storage_type, key, _location, numel = pid[:5]
        if kind != "storage":
            raise ValueError(f"Unsupported persistent object: {kind}")
        dtype = DTYPES[storage_type.__name__]
        raw = self.archive.read(f"{self.prefix}/data/{key}")
        return Storage(np.frombuffer(raw, dtype=dtype, count=numel))

    @staticmethod
    def _rebuild_tensor(storage, offset, size, stride, *unused):
        base = storage.array[offset:]
        byte_strides = tuple(int(s) * base.dtype.itemsize for s in stride)
        return np.lib.stride_tricks.as_strided(base, shape=tuple(size), strides=byte_strides).copy()


def torch_zip_load(path: Path):
    with zipfile.ZipFile(path) as archive:
        pkl_name = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        prefix = pkl_name.split("/", 1)[0]
        return TorchZipUnpickler(archive.read(pkl_name), archive, prefix).load()


def numeric_arrays(obj):
    if isinstance(obj, np.ndarray) and obj.dtype.kind in "fiu":
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from numeric_arrays(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from numeric_arrays(value)


def optimizer_moments(opt):
    exp_avg, exp_avg_sq = [], []
    state = opt.get("state", {}) if isinstance(opt, dict) else {}
    for entry in state.values():
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("exp_avg"), np.ndarray):
            exp_avg.append(entry["exp_avg"])
        if isinstance(entry.get("exp_avg_sq"), np.ndarray):
            exp_avg_sq.append(entry["exp_avg_sq"])
    return exp_avg, exp_avg_sq


def sqsum(arrays):
    return sum(float(np.dot(a.ravel().astype(np.float64), a.ravel().astype(np.float64))) for a in arrays)


def font(size=22, bold=False):
    names = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


COLORS = {"blue": "#2563EB", "cyan": "#0891B2", "orange": "#EA580C", "gray": "#64748B", "grid": "#DDE4EE"}


def draw_chart(draw, box, title, xs, series, y_log=False, subtitle=None):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=16, fill="#FFFFFF", outline="#D9E1EA", width=2)
    draw.text((x0 + 24, y0 + 18), title, fill="#172033", font=font(24, True))
    if subtitle:
        draw.text((x0 + 24, y0 + 50), subtitle, fill="#64748B", font=font(15))
    left, top, right, bottom = x0 + 86, y0 + 86, x1 - 28, y1 - 58
    values = [v for _, ys, _ in series for v in ys if math.isfinite(v) and (v > 0 or not y_log)]
    if not values:
        draw.text((left, top + 60), "No data", fill="#64748B", font=font(22))
        return
    transform = (lambda v: math.log10(max(v, 1e-30))) if y_log else (lambda v: v)
    tv = [transform(v) for v in values]
    lo, hi = min(tv), max(tv)
    pad = (hi - lo) * 0.12 if hi > lo else max(abs(hi), 1) * 0.12
    lo, hi = lo - pad, hi + pad
    for i in range(5):
        yy = top + (bottom - top) * i / 4
        draw.line((left, yy, right, yy), fill=COLORS["grid"], width=1)
        raw = 10 ** (hi - (hi - lo) * i / 4) if y_log else hi - (hi - lo) * i / 4
        label = f"{raw:.1e}" if y_log or abs(raw) >= 1000 or (abs(raw) < .01 and raw != 0) else f"{raw:.3g}"
        draw.text((left - 12, yy), label, anchor="rm", fill="#64748B", font=font(14))
    xmin, xmax = min(xs), max(xs)
    for xv in np.linspace(xmin, xmax, 6):
        xx = left + (right - left) * (xv - xmin) / (xmax - xmin)
        draw.text((xx, bottom + 12), f"{int(xv/1000)}k", anchor="ma", fill="#64748B", font=font(14))
    for label, ys, color in series:
        pts = []
        for xv, v in zip(xs, ys):
            if not math.isfinite(v) or (y_log and v <= 0):
                continue
            xx = left + (right - left) * (xv - xmin) / (xmax - xmin)
            yy = bottom - (bottom - top) * (transform(v) - lo) / (hi - lo)
            pts.append((xx, yy))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=4, joint="curve")
        for p in pts:
            draw.ellipse((p[0]-5, p[1]-5, p[0]+5, p[1]+5), fill=color, outline="white", width=2)
    lx = left
    for label, _, color in series:
        draw.line((lx, y1 - 25, lx + 26, y1 - 25), fill=color, width=4)
        draw.text((lx + 34, y1 - 25), label, anchor="lm", fill="#334155", font=font(15))
        lx += 44 + draw.textlength(label, font=font(15))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(args.checkpoint_dir.glob("step_*"))
    rows, previous = [], None
    for cp in checkpoints:
        match = re.search(r"(\d+)$", cp.name)
        if not match or not (cp / "agent.th").exists():
            continue
        step = int(match.group(1))
        agent = torch_zip_load(cp / "agent.th")
        opt = torch_zip_load(cp / "opt.th")
        params = [a for a in numeric_arrays(agent) if a.dtype.kind == "f"]
        count = sum(a.size for a in params)
        norm = math.sqrt(sqsum(params))
        pmax = max(float(np.max(np.abs(a))) for a in params)
        drift = float("nan")
        rel_drift = float("nan")
        if previous is not None and len(previous) == len(params):
            diff_sq = sum(float(np.dot((a-b).ravel().astype(np.float64), (a-b).ravel().astype(np.float64))) for a, b in zip(params, previous))
            drift = math.sqrt(diff_sq)
            rel_drift = drift / norm if norm else float("nan")
        m1, m2 = optimizer_moments(opt)
        m1_rms = math.sqrt(sqsum(m1) / sum(a.size for a in m1)) if m1 else float("nan")
        m2_mean = sqsum([np.sqrt(np.maximum(a, 0)) for a in m2]) / sum(a.size for a in m2) if m2 else float("nan")
        update_proxy = math.sqrt(sum(float(np.sum((a / (np.sqrt(b) + 1e-8)) ** 2)) for a, b in zip(m1, m2)) / sum(a.size for a in m1)) if m1 and len(m1) == len(m2) else float("nan")
        rows.append({"step": step, "parameter_count": count, "parameter_l2_norm": norm, "parameter_rms": norm/math.sqrt(count), "max_abs_parameter": pmax, "checkpoint_drift_l2": drift, "relative_checkpoint_drift": rel_drift, "adam_first_moment_rms": m1_rms, "adam_second_moment_mean": m2_mean, "adam_update_proxy_rms": update_proxy})
        previous = params
        print(step, count, norm, drift, m1_rms, m2_mean, update_proxy)

    csv_path = args.output_dir / "checkpoint_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    xs = [r["step"] for r in rows]
    img = Image.new("RGB", (1800, 1220), "#F4F7FB")
    draw = ImageDraw.Draw(img)
    draw.text((70, 42), "MAICC Stage-0 Training Diagnostics", fill="#101828", font=font(38, True))
    draw.text((70, 94), "Derived from 10 saved checkpoints · historical batch loss was not stored", fill="#526174", font=font(20))
    l2_0, max_0 = rows[0]["parameter_l2_norm"], rows[0]["max_abs_parameter"]
    draw_chart(draw, (55, 145, 890, 630), "Parameter scale (indexed)", xs, [("L2 norm", [r["parameter_l2_norm"]/l2_0 for r in rows], COLORS["blue"]), ("Max |parameter|", [r["max_abs_parameter"]/max_0 for r in rows], COLORS["orange"])], y_log=False, subtitle="10k checkpoint = 1.0; absolute values are available in the CSV")
    draw_chart(draw, (910, 145, 1745, 630), "Checkpoint-to-checkpoint movement", xs, [("Relative drift", [r["relative_checkpoint_drift"] for r in rows], COLORS["cyan"])], y_log=False, subtitle="L2(current - previous) / L2(current)")
    draw_chart(draw, (55, 655, 890, 1140), "Adam optimizer moments", xs, [("1st moment RMS", [r["adam_first_moment_rms"] for r in rows], COLORS["blue"]), ("2nd moment mean", [r["adam_second_moment_mean"] for r in rows], COLORS["orange"])], y_log=True)
    draw_chart(draw, (910, 655, 1745, 1140), "Adam update proxy", xs, [("RMS m/(sqrt(v)+ε)", [r["adam_update_proxy_rms"] for r in rows], COLORS["cyan"])], y_log=False, subtitle="Direction/strength proxy before LR and bias correction")
    png_path = args.output_dir / "training_diagnostics.png"
    img.save(png_path, dpi=(180, 180))

    summary = {"checkpoints": len(rows), "first_step": xs[0], "last_step": xs[-1], "parameter_count": rows[-1]["parameter_count"], "relative_drift_first_available": rows[1]["relative_checkpoint_drift"], "relative_drift_final": rows[-1]["relative_checkpoint_drift"], "update_proxy_first": rows[0]["adam_update_proxy_rms"], "update_proxy_final": rows[-1]["adam_update_proxy_rms"]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
