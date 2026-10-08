"""
Sinh các hình cho bản thảo Bài báo #1 (bản EN và VI) từ JSON kết quả
benchmark trong outputs/phase5/plan_v3/ - không chạy lại FE/GPU, chỉ đọc
số đã đo để hình luôn khớp đúng bảng số trong bài.

Cách chạy (từ repo root, env simp):
    python3 docs/paper1/scripts/make_figures.py

Output: docs/paper1/figures/fig_{parity,gap,ood}_{en,vi}.pdf
"""

import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = os.path.dirname(ROOT)  # repo root
RES = os.path.join(ROOT, "outputs", "phase5", "plan_v3")
OUT = os.path.join(ROOT, "docs", "paper1", "figures")

# Bảng màu categorical cố định theo thứ tự (dataviz skill): xám cho baseline
# tham chiếu, cam = refine liên tục, xanh dương = refine nhận thức nhị phân
# hóa (phương pháp đề xuất), xanh ngọc = chế độ guarded.
C_BASE = "#8a8a85"
C_CONT = "#eb6834"
C_AWARE = "#2a78d6"
C_GUARD = "#1baf7a"
# Biên dưới của v12 trong tập train (min tập train, plan.md P1.1c).
TRAIN_MIN_V12 = -1.95

LABELS = {
    "en": {
        "target": r"Target $\nu_{12}^{*}$",
        "fe": r"FE-verified $\nu_{12}$",
        "single": "Single-shot (no refinement)",
        "cont": "Refinement, continuous density",
        "aware": "Refinement, binarization-aware",
        "obj": "Final optimization objective (MSE)",
        "ver": "FE-verified MSE after binarization",
        "ideal": "objective = verified",
        "ood_x": r"Target $\nu_{12}^{*}$ (with $\nu_{21}^{*}=-0.05$)",
        "ood_y": r"Median relative error of $\nu_{12}$",
        "bo30": "Best-of-30 (no refinement)",
        "guard": "Binarization-aware, guarded",
        "outside": "outside training range",
        "thr": "8% criterion",
        "sf_x": r"Target $\nu_{12}^{*}=\nu_{21}^{*}$ (training data: $\nu_{12}<0$)",
        "sf_y": r"FE-verified $\nu_{12}$",
        "retr": "Nearest-neighbour retrieval",
        "cvae": "cVAE, best-of-10",
        "sf_bo30": "Generative, best-of-30",
        "sf_ref": "Generative + guarded refinement",
        "sf_simp": "Topology optimization from scratch",
        "sp_x": "Spearman: composite score vs. Pareto rank",
        "sp_y": "Targets",
        "sp_thr": "pre-specified mean criterion 0.7",
    },
    "vi": {
        "target": r"$\nu_{12}^{*}$ mục tiêu",
        "fe": r"$\nu_{12}$ kiểm chứng bằng FE",
        "single": "Sinh một lần (không tinh chỉnh)",
        "cont": "Tinh chỉnh trên mật độ liên tục",
        "aware": "Tinh chỉnh nhận biết nhị phân hóa",
        "obj": "Hàm mục tiêu cuối (MSE)",
        "ver": "MSE kiểm chứng FE sau nhị phân hóa",
        "ideal": "mục tiêu = kiểm chứng",
        "ood_x": r"$\nu_{12}^{*}$ mục tiêu (với $\nu_{21}^{*}=-0{,}05$)",
        "ood_y": r"Trung vị sai số tương đối của $\nu_{12}$",
        "bo30": "Best-of-30 (không tinh chỉnh)",
        "guard": "Nhận biết nhị phân hóa, có kiểm soát",
        "outside": "ngoài dải huấn luyện",
        "thr": "ngưỡng 8%",
        "sf_x": r"$\nu_{12}^{*}=\nu_{21}^{*}$ mục tiêu (dữ liệu huấn luyện: $\nu_{12}<0$)",
        "sf_y": r"$\nu_{12}$ kiểm chứng bằng FE",
        "retr": "Truy xuất láng giềng gần nhất",
        "cvae": "cVAE, tốt nhất trong 10",
        "sf_bo30": "Sinh, tốt nhất trong 30",
        "sf_ref": "Sinh + tinh chỉnh có kiểm soát",
        "sf_simp": "Tối ưu tô-pô từ đầu",
        "sp_x": "Spearman: điểm tổng hợp và tầng Pareto",
        "sp_y": "Số mục tiêu",
        "sp_thr": "ngưỡng trung bình đặt trước 0,7",
    },
}


def load(name: str) -> list:
    """Đọc danh sách per_condition của 1 file kết quả benchmark.

    Args:
        name: tên file JSON trong outputs/phase5/plan_v3/.

    Returns:
        list các dict per-condition (target/baseline/refined v12, v21...).
    """
    with open(os.path.join(RES, name)) as f:
        return json.load(f)["per_condition"]


def style() -> None:
    """Style chung kiểu tạp chí: font sans hỗ trợ tiếng Việt, grid mờ."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e3e3e0",
            "grid.linewidth": 0.6,
            "axes.edgecolor": "#6b6b66",
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def fig_parity(lang: str) -> None:
    """Hình parity target-vs-FE ν12 trên IN100: sinh một lần, refine liên
    tục, refine nhận thức nhị phân hóa (cùng 100 condition, cùng z khởi đầu).

    Args:
        lang: "en" hoặc "vi".
    """
    lb = LABELS[lang]
    a = load("p1_1a_v2_finetuned_in100.json")
    e = load("p1_9_in100_single_nofp_images.json")
    panels = [
        (lb["single"], a, "baseline_v12", C_BASE),
        (lb["cont"], a, "refined_v12", C_CONT),
        (lb["aware"], e, "refined_v12", C_AWARE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for i, (ax, (title, rows, key, col)) in enumerate(zip(axes, panels)):
        t = np.array([r["target_v12"] for r in rows])
        y = np.array([r[key] for r in rows])
        r2 = 1 - np.sum((y - t) ** 2) / np.sum((t - t.mean()) ** 2)
        lo, hi = min(t.min(), y.min()) - 0.05, max(t.max(), y.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], color="#6b6b66", lw=0.8, ls="--")
        ax.scatter(t, y, s=10, color=col, edgecolor="white", lw=0.4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(
            textwrap.fill(f"({'abc'[i]}) {title}", 26),
            fontsize=7.5,
            loc="left",
        )
        r2s = f"{r2:.3f}" if lang == "en" else f"{r2:.3f}".replace(".", ",")
        ax.text(0.05, 0.92, f"$R^2$ = {r2s}", transform=ax.transAxes)
        ax.set_xlabel(lb["target"])
    axes[0].set_ylabel(lb["fe"])
    if lang == "vi":
        # Dấu thập phân kiểu Việt trên trục.
        fmt = plt.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
        for ax in axes:
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
    fig.savefig(os.path.join(OUT, f"fig_parity_{lang}.pdf"))
    plt.close(fig)


def fig_gap(lang: str) -> None:
    """Hình khoảng cách giữa objective tối ưu cuối và sai số FE thật sau
    nhị phân hóa - bằng chứng cơ chế cho việc refine liên tục khai thác vật
    liệu xám (điểm nằm xa đường chéo).

    Args:
        lang: "en" hoặc "vi".
    """
    lb = LABELS[lang]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for name, col, lab, mk in [
        ("p1_1a_v2_finetuned_in100.json", C_CONT, lb["cont"], "o"),
        ("p1_9_in100_single_nofp_images.json", C_AWARE, lb["aware"], "s"),
    ]:
        rows = load(name)
        obj = np.array([r["refinement_loss_history"][-1] for r in rows])
        ver = np.array(
            [
                0.5
                * (
                    (r["refined_v12"] - r["target_v12"]) ** 2
                    + (r["refined_v21"] - r["target_v21"]) ** 2
                )
                for r in rows
            ]
        )
        # Kẹp dưới 1e-14 để log-scale không mất điểm objective = 0.
        ax.scatter(
            np.maximum(obj, 1e-14),
            np.maximum(ver, 1e-14),
            s=12,
            color=col,
            marker=mk,
            edgecolor="white",
            lw=0.4,
            label=lab,
        )
    lim = (1e-13, 1e-1)
    ax.plot(lim, lim, color="#6b6b66", lw=0.8, ls="--", label=lb["ideal"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(1e-9, 1e-1)
    ax.set_xlabel(lb["obj"])
    ax.set_ylabel(lb["ver"])
    ax.legend(
        loc="upper center", fontsize=7, bbox_to_anchor=(0.5, -0.18), ncol=1
    )
    fig.savefig(os.path.join(OUT, f"fig_gap_{lang}.pdf"))
    plt.close(fig)


def fig_ood(lang: str) -> None:
    """Hình quét OOD dị hướng: trung vị (± IQR, 10 lặp/target) sai số
    tương đối ν12 theo ν*, cho best-of-30, refine liên tục, refine nhận
    thức nhị phân hóa và chế độ guarded.

    Args:
        lang: "en" hoặc "vi".
    """
    lb = LABELS[lang]
    d = load("p1_9_r3_ood_cont_nofp.json")
    e = load("p1_9_r4_ood_aware_nofp.json")
    targets = sorted({r["target_v12"] for r in d}, reverse=True)

    def rel(rows, t, which):
        out = []
        for r in rows:
            if r["target_v12"] != t:
                continue
            if which == "guarded":
                v = (
                    r["refined_v12"]
                    if r["guarded_accept"]
                    else r["baseline_v12"]
                )
            else:
                v = r[f"{which}_v12"]
            out.append(abs(v - t) / abs(t))
        return np.array(out)

    series = [
        (lb["bo30"], d, "baseline", C_BASE, "o", -0.03),
        (lb["cont"], d, "refined", C_CONT, "^", -0.01),
        (lb["aware"], e, "refined", C_AWARE, "s", 0.01),
        (lb["guard"], e, "guarded", C_GUARD, "D", 0.03),
    ]
    # Panel trên: mật độ dữ liệu train theo ν12 (review mục 6) - bão hòa
    # xảy ra nơi dữ liệu thưa dần về 0, không phải tại 1 "biên" sắc.
    fig, (axh, ax) = plt.subplots(
        2,
        1,
        figsize=(4.2, 3.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3]},
    )
    v12_train = np.load(os.path.join(ROOT, "outputs", "phase3", "train.npz"))[
        "v12"
    ]
    axh.hist(
        v12_train,
        bins=np.arange(-2.6, -0.85, 0.05),
        color=C_BASE,
        edgecolor="white",
        lw=0.5,
    )
    axh.set_yscale("log")
    axh.set_ylabel(DLB[lang]["dens"], fontsize=7)
    axh.axvspan(-2.6, TRAIN_MIN_V12, color="#f1f0ec", zorder=0)
    ax.axvspan(-2.6, TRAIN_MIN_V12, color="#f1f0ec", zorder=0)
    ax.text(-2.55, 0.52, lb["outside"], fontsize=7, color="#6b6b66")
    ax.axhline(0.08, color="#6b6b66", lw=0.8, ls=":")
    ax.text(-1.02, 0.09, lb["thr"], fontsize=7, color="#6b6b66", ha="right")
    for lab, rows, which, col, mk, dx in series:
        med, lo, hi = [], [], []
        for t in targets:
            v = rel(rows, t, which)
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            med.append(q2)
            lo.append(q2 - q1)
            hi.append(q3 - q2)
        x = np.array(targets) + dx
        ax.errorbar(
            x,
            med,
            yerr=[lo, hi],
            color=col,
            marker=mk,
            ms=4,
            lw=1.2,
            capsize=2,
            label=lab,
        )
    ax.set_xlim(-2.6, -0.9)
    ax.set_ylim(0, 0.58)
    ax.set_xlabel(lb["ood_x"])
    ax.set_ylabel(lb["ood_y"])
    ax.legend(
        loc="upper center", fontsize=7, bbox_to_anchor=(0.5, -0.2), ncol=2
    )
    if lang == "vi":
        # Dấu thập phân kiểu Việt trên trục.
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
        )
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
        )
    fig.savefig(os.path.join(OUT, f"fig_ood_{lang}.pdf"))
    plt.close(fig)


def _vi_decimal(ax, lang: str) -> None:
    """Đổi dấu thập phân trục sang dấu phẩy cho bản tiếng Việt.

    Args:
        ax: trục matplotlib.
        lang: "en" (không đổi) hoặc "vi".
    """
    if lang != "vi":
        return
    fmt = plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


def fig_signflip(lang: str) -> None:
    """Hình OOD đổi dấu (P1.6b, pipeline không fp): 12 mục tiêu đối xứng
    ν12*=ν21*>0 (dữ liệu huấn luyện toàn auxetic). So cVAE best-of-30, cVAE
    + refine nhận thức nhị phân hóa có kiểm soát và SIMP từ đầu (best-of-4,
    ngân sách full); truy xuất bị chặn ở ν12 ≤ −0,002 (đường xám). Điểm rỗng
    = thiết kế suy biến (min(E_x, E_y) < 1e-3·E0) theo
    p1_10_degenerate_check.json - SIMP trả về ô rỗng có ν = ν0 = 0,3.

    Args:
        lang: "en" hoặc "vi".
    """
    lb = LABELS[lang]
    cv = load("p1_9_r7_signflip_sym_nofp.json")
    sp = load("p1_6b_simp_sym.json")
    with open(os.path.join(RES, "p1_10_degenerate_check.json")) as f:
        deg = json.load(f)["results"]["SF_sym"]
    t = np.array([c["target_v12"] for c in cv])
    series = [
        (
            lb["sf_bo30"],
            np.array([c["baseline_v12"] for c in cv]),
            None,
            C_BASE,
            "o",
        ),
        (
            lb["sf_ref"],
            np.array([_guarded(c)[1] for c in cv]),
            deg["cvae"]["degenerate_idx"],
            C_AWARE,
            "s",
        ),
        (
            lb["sf_simp"],
            np.array([s["full"]["best"]["v12"] for s in sp]),
            deg["simp"]["degenerate_idx"],
            C_GUARD,
            "D",
        ),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.plot([0, 0.65], [0, 0.65], color="#6b6b66", lw=0.8, ls="--")
    ax.axhline(-0.002, color=C_BASE, lw=1.0, label=lb["retr"])
    for lab, y, bad, col, mk in series:
        bad = set(bad or [])
        ok = np.array([k not in bad for k in range(len(t))])
        ax.plot(t, y, "-", color=col, lw=0.8, alpha=0.6)
        ax.scatter(
            t[ok], y[ok], color=col, marker=mk, s=18, label=lab, zorder=3
        )
        if (~ok).any():
            # Thiết kế suy biến: vẽ rỗng để không bị đọc như nghiệm hợp lệ.
            ax.scatter(
                t[~ok],
                y[~ok],
                facecolor="white",
                edgecolor=col,
                marker=mk,
                s=18,
                zorder=3,
            )
    ax.set_xlim(0, 0.65)
    ax.set_ylim(-0.1, 0.7)
    ax.set_xlabel(lb["sf_x"])
    ax.set_ylabel(lb["sf_y"])
    ax.legend(loc="upper left", fontsize=6.5)
    _vi_decimal(ax, lang)
    fig.savefig(os.path.join(OUT, f"fig_signflip_{lang}.pdf"))
    plt.close(fig)


def fig_spearman(lang: str) -> None:
    """Hình phân bố tương quan Spearman (24 mục tiêu) giữa điểm tổng hợp
    0,6/0,3/0,1 và tầng Pareto độc lập trên 3 trục (sai số, chế tạo, thẩm
    mỹ) của 30 ứng viên mỗi mục tiêu.

    Args:
        lang: "en" hoặc "vi".
    """
    lb = LABELS[lang]
    with open(os.path.join(RES, "p1_9_pareto_nofp.json")) as f:
        rho = np.array(json.load(f)["spearman_per_condition"])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.hist(
        rho,
        bins=np.arange(0.4, 0.95, 0.05),
        color=C_AWARE,
        edgecolor="white",
        lw=1.0,
    )
    ax.axvline(0.7, color="#6b6b66", lw=0.8, ls=":", label=lb["sp_thr"])
    ax.axvline(
        rho.mean(),
        color=C_CONT,
        lw=1.2,
        label=(
            f"mean {rho.mean():.3f}"
            if lang == "en"
            else f"trung bình {rho.mean():.3f}".replace(".", ",")
        ),
    )
    ax.set_xlabel(lb["sp_x"])
    ax.set_ylabel(lb["sp_y"])
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7, ncol=2
    )
    _vi_decimal(ax, lang)
    fig.savefig(os.path.join(OUT, f"fig_spearman_{lang}.pdf"))
    plt.close(fig)


# Nhãn cho các hình thiết kế (P1.6c, 2026-10-06) - tách khỏi LABELS để
# không đụng các hình đã có.
DLB = {
    "en": {
        "base": "Single sample",
        "cont_raw": "Continuous refinement\n(decoder output)",
        "cont_bin": "Continuous refinement\n(binarized)",
        "aware": "Binarization-aware\nrefinement (binarized)",
        "obj": "objective",
        "cvae": "cVAE + guarded\nrefinement",
        "simp": "SIMP from scratch\n(best of 4 seeds)",
        "tgt": "target",
        "ver": "verified",
        "t_in": "cVAE, in-distribution",
        "t_pos": r"cVAE, $\nu^{*}>0$ (unseen sign)",
        "t_simp": "SIMP from scratch",
        "ex": r"$E_x/E_0$ (FE, $200\times200$)",
        "nu": r"$\nu_{12}$ (FE, $200\times200$)",
        "degen": "disconnected cell\n(no load path in x)",
        "dens": "Training\nsamples",
        "deform_cap": "Uniaxial tension along x (arrows); dashed: undeformed "
        "2×2 cells; displacements magnified",
    },
    "vi": {
        "base": "Mẫu sinh một lần",
        "cont_raw": "Tinh chỉnh liên tục\n(đầu ra decoder)",
        "cont_bin": "Tinh chỉnh liên tục\n(sau nhị phân hóa)",
        "aware": "Tinh chỉnh nhận biết\nnhị phân hóa",
        "obj": "hàm mục tiêu",
        "cvae": "cVAE + tinh chỉnh\ncó kiểm soát",
        "simp": "SIMP từ đầu\n(tốt nhất 4 seed)",
        "tgt": "mục tiêu",
        "ver": "kiểm chứng",
        "t_in": "cVAE, trong phân phối",
        "t_pos": r"cVAE, $\nu^{*}>0$ (dấu chưa thấy)",
        "t_simp": "SIMP từ đầu",
        "ex": r"$E_x/E_0$ (FE, lưới $200\times200$)",
        "nu": r"$\nu_{12}$ (FE, lưới $200\times200$)",
        "degen": "ô bị đứt\n(không truyền lực theo x)",
        "dens": "Số mẫu\nhuấn luyện",
        "deform_cap": "Kéo đơn trục theo x (mũi tên); nét đứt: 2×2 ô chưa "
        "biến dạng; chuyển vị được phóng đại",
    },
}


def _num(v: float, lang: str, fmt: str = "{:.3f}") -> str:
    """Định dạng số theo ngôn ngữ (dấu phẩy thập phân cho tiếng Việt)."""
    s = fmt.format(v)
    return s.replace(".", ",") if lang == "vi" else s


def _pair(a: float, b: float, lang: str) -> str:
    """Cặp (ν12, ν21) 2 chữ số; tiếng Việt dùng "; " vì dấu phẩy đã là dấu
    thập phân."""
    sep = "; " if lang == "vi" else ", "
    return f"({_num(a, lang, '{:.2f}')}{sep}{_num(b, lang, '{:.2f}')})"


def _show(ax, img: np.ndarray) -> None:
    """Vẽ 1 ảnh mật độ (đen = vật liệu), bỏ trục/lưới."""
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#c3c3be")


def _guarded(row: dict):
    """(ảnh nhị phân, ν12, ν21) của thiết kế cuối chế độ guarded."""
    key = "refined" if row["guarded_accept"] else "baseline"
    return (
        np.asarray(row[f"{key}_image"], dtype=float),
        row[f"{key}_v12"],
        row[f"{key}_v21"],
    )


def fig_refine_examples(lang: str) -> None:
    """Trước/sau tinh chỉnh cho 3 condition IN100: mẫu gốc, refine liên tục
    (ảnh decoder xám + sau nhị phân hóa), refine nhận biết nhị phân hóa.
    Số trên ảnh: ν12, ν21 kiểm chứng FE và objective cuối (refine liên tục).
    Nguồn: p1_6c_refine_examples.{npz,json} (refine_examples.py).

    Args:
        lang: "en" hoặc "vi".
    """
    lb = DLB[lang]
    arr = np.load(os.path.join(RES, "p1_6c_refine_examples.npz"))
    with open(os.path.join(RES, "p1_6c_refine_examples.json")) as f:
        rows = json.load(f)
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.2, 2.0 * len(rows)))
    for r, row in enumerate(rows):
        k = row["index"]
        t = row["target"]
        panels = [
            (arr[f"c{k}_baseline"] > 0.5, row["baseline"], None),
            (
                arr[f"c{k}_continuous"],
                row["continuous"]["verified"],
                row["continuous"]["final_objective"],
            ),
            (
                arr[f"c{k}_continuous"] > 0.5,
                row["continuous"]["verified"],
                None,
            ),
            (arr[f"c{k}_aware"] > 0.5, row["aware"]["verified"], None),
        ]
        for c, (img, nu, obj) in enumerate(panels):
            ax = axes[r, c]
            _show(ax, img.astype(float))
            if r == 0:
                title = [
                    lb["base"],
                    lb["cont_raw"],
                    lb["cont_bin"],
                    lb["aware"],
                ][c]
                ax.set_title(title, fontsize=7)
            if obj is not None:
                txt = f"{lb['obj']} {obj:.0e}"
            else:
                err = abs(nu[0] - t[0]) + abs(nu[1] - t[1])
                txt = (
                    f"ν = {_pair(nu[0], nu[1], lang)}\n"
                    f"|Δν| = {_num(err, lang)}"
                )
            ax.set_xlabel(txt, fontsize=6.5)
        axes[r, 0].set_ylabel(
            f"{lb['tgt']}\n{_pair(t[0], t[1], lang)}",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_refine_examples_{lang}.pdf"))
    plt.close(fig)


def fig_designs(lang: str, n: int = 6) -> None:
    """Bộ sưu tập thiết kế cuối cho n mục tiêu IN100 trải đều dải ν12:
    hàng trên cVAE (best-of-30 + refine guarded, ảnh 64²), hàng dưới SIMP từ
    đầu (best-of-4 seed, ngân sách full, lưới 50²). Nguồn:
    p1_7_c5_in100_bo30_nofp.json, p1_6a_simp_baseline_in100.json.

    Args:
        lang: "en" hoặc "vi".
        n: số mục tiêu.
    """
    lb = DLB[lang]
    cv = load("p1_7_c5_in100_bo30_nofp.json")
    sp = load("p1_6a_simp_baseline_in100.json")
    order = np.argsort([c["target_v12"] for c in cv])
    picks = order[np.linspace(0, len(order) - 1, n).round().astype(int)]
    fig, axes = plt.subplots(2, n, figsize=(7.2, 3.0))
    for j, i in enumerate(picks):
        t = (cv[i]["target_v12"], cv[i]["target_v21"])
        img, v12, v21 = _guarded(cv[i])
        best = sp[i]["full"]["best"]
        for r, (im, a, b) in enumerate(
            [
                (img, v12, v21),
                (np.asarray(best["image"], float), best["v12"], best["v21"]),
            ]
        ):
            ax = axes[r, j]
            _show(ax, im)
            ax.set_xlabel(
                _pair(a, b, lang),
                fontsize=6.5,
            )
        axes[0, j].set_title(
            f"{lb['tgt']} {_pair(t[0], t[1], lang)}",
            fontsize=6.5,
        )
    axes[0, 0].set_ylabel(lb["cvae"], fontsize=7)
    axes[1, 0].set_ylabel(lb["simp"], fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_designs_{lang}.pdf"))
    plt.close(fig)


def fig_tiling(lang: str) -> None:
    """Lưới lát 3×3 của 3 ô cơ sở: cVAE trong phân phối (mục tiêu trung vị
    IN100), cVAE với mục tiêu ν*=0,5 (dấu chưa thấy khi train, P1.6b) và
    SIMP từ đầu cho cùng mục tiêu trong phân phối - kiểm tra trực quan tính
    tuần hoàn/liên tục khi lát.

    Args:
        lang: "en" hoặc "vi".
    """
    lb = DLB[lang]
    cv = load("p1_7_c5_in100_bo30_nofp.json")
    sp = load("p1_6a_simp_baseline_in100.json")
    sf = load("p1_9_r7_signflip_sym_nofp.json")
    i = int(np.argsort([c["target_v12"] for c in cv])[len(cv) // 2])
    j = int(np.argmin([abs(c["target_v12"] - 0.5) for c in sf]))
    cells = [
        (lb["t_in"], _guarded(cv[i])),
        (lb["t_pos"], _guarded(sf[j])),
        (
            lb["t_simp"],
            (
                np.asarray(sp[i]["full"]["best"]["image"], float),
                sp[i]["full"]["best"]["v12"],
                sp[i]["full"]["best"]["v21"],
            ),
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (title, (img, v12, v21)) in zip(axes, cells):
        _show(ax, np.tile(img, (3, 3)))
        ax.set_title(title, fontsize=7)
        ax.set_xlabel(
            f"ν = {_pair(v12, v21, lang)}",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_tiling_{lang}.pdf"))
    plt.close(fig)


def fig_stiffness(lang: str) -> None:
    """ν12 theo E_x/E0 của thiết kế cuối (lưới FE mịn 200², cùng hình học)
    - kiểm tra thiết kế auxetic có phải cơ cấu gần như không có độ cứng.
    Nguồn: p1_6x_final_design_eval.json (final_design_eval.py).

    Args:
        lang: "en" hoặc "vi".
    """
    lb = DLB[lang]
    with open(os.path.join(RES, "p1_6x_final_design_eval.json")) as f:
        rows = json.load(f)["rows"]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for method, col, mk, name in (
        ("SIMP full best-of-4", C_BASE, "o", lb["t_simp"]),
        ("cVAE guarded", C_AWARE, "s", "cVAE"),
    ):
        pts = np.array(
            [
                (r["eval"]["k4"]["E_x"], r["eval"]["k4"]["v12"])
                for r in rows
                if r["method"] == method and r["eval"]["k4"] is not None
            ]
        )
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=10,
            color=col,
            marker=mk,
            edgecolor="white",
            lw=0.3,
            label=name,
        )
        # Ô suy biến (không có đường truyền lực theo x, E_x ~ E_min): chú
        # thích rõ thay vì ẩn - là 1 lần thất bại thật (condition 46 IN100).
        for ex, nu in pts[pts[:, 0] < 1e-3]:
            ax.annotate(
                lb["degen"],
                (ex, nu),
                xytext=(ex * 30, nu - 0.3),
                fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.6),
            )
    ax.set_xscale("log")
    ax.set_xlabel(lb["ex"])
    ax.set_ylabel(lb["nu"])
    ax.legend(loc="lower right", fontsize=7)
    if lang == "vi":
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
        )
    fig.savefig(os.path.join(OUT, f"fig_stiffness_{lang}.pdf"))
    plt.close(fig)


def fig_deformation(lang: str, strain_vis: float = 0.12) -> None:
    """Lưới 2×2 ô đã biến dạng dưới kéo đơn trục σ_xx (cơ chế auxetic):
    thiết kế auxetic mạnh nhất IN100 và thiết kế ν*=0,5. Chuyển vị = dao
    động tuần hoàn + phần affine ε·X, phóng đại để biến dạng vĩ mô lớn
    nhất bằng strain_vis. Viền nét đứt: ô chưa biến dạng.
    Nguồn: p1_6c_deformation_examples.{npz,json} (deformation_examples.py).

    Args:
        lang: "en" hoặc "vi".
        strain_vis: biến dạng vĩ mô lớn nhất khi hiển thị (phóng đại).
    """
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle

    lb = DLB[lang]
    arr = np.load(os.path.join(RES, "p1_6c_deformation_examples.npz"))
    with open(os.path.join(RES, "p1_6c_deformation_examples.json")) as f:
        meta = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, m in zip(axes, meta):
        n = m["name"]
        design, fl = arr[f"{n}_design"], arr[f"{n}_fluct"]
        eps = arr[f"{n}_eps"]
        k = strain_vis / np.abs(eps).max()
        nely, nelx = design.shape
        jj, ii = np.meshgrid(
            np.arange(nely + 1), np.arange(nelx + 1), indexing="ij"
        )
        polys = []
        for a in range(2):
            for b in range(2):
                X = ii / nelx + a
                Y = jj / nely + b
                # Affine theo quy ước U0 của solver: u = εxx·x + γ/2·y,
                # v = εyy·y + γ/2·x.
                ux = k * (fl[..., 0] + eps[0] * X + 0.5 * eps[2] * Y)
                uy = k * (fl[..., 1] + eps[1] * Y + 0.5 * eps[2] * X)
                xd, yd = X + ux, Y + uy
                for r, c in zip(*np.nonzero(design > 0.5)):
                    polys.append(
                        [
                            (xd[r, c], yd[r, c]),
                            (xd[r, c + 1], yd[r, c + 1]),
                            (xd[r + 1, c + 1], yd[r + 1, c + 1]),
                            (xd[r + 1, c], yd[r + 1, c]),
                        ]
                    )
        ax.add_collection(
            PolyCollection(
                polys, facecolor=C_AWARE, edgecolor=C_AWARE, linewidths=0.3
            )
        )
        ax.add_patch(
            Rectangle((0, 0), 2, 2, fill=False, ls="--", lw=0.8, ec="#6b6b66")
        )
        ax.annotate(
            "",
            xy=(2.45, 1.0),
            xytext=(2.15, 1.0),
            arrowprops=dict(arrowstyle="->", lw=1.2),
        )
        ax.annotate(
            "",
            xy=(-0.45, 1.0),
            xytext=(-0.15, 1.0),
            arrowprops=dict(arrowstyle="->", lw=1.2),
        )
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-0.4, 2.4)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"{lb['tgt']} {_pair(*m['target'], lang)}\n"
            r"$\nu_{12}$ = " + _num(m["nu12"], lang, "{:.2f}"),
            fontsize=7.5,
        )
    fig.text(0.5, 0.1, lb["deform_cap"], ha="center", fontsize=7)
    fig.savefig(os.path.join(OUT, f"fig_deformation_{lang}.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    style()
    os.makedirs(OUT, exist_ok=True)
    for lang in ("en", "vi"):
        fig_parity(lang)
        fig_gap(lang)
        fig_ood(lang)
        fig_signflip(lang)
        fig_spearman(lang)
        fig_refine_examples(lang)
        fig_designs(lang)
        fig_tiling(lang)
        fig_stiffness(lang)
        fig_deformation(lang)
    print("Wrote figures to", OUT)
