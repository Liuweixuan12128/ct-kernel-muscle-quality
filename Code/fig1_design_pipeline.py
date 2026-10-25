"""Figure 1 | Study design and fully automated body-composition analysis of the staging chest CT.

a  Design strip: public cohorts (NLST resected lung cancer; TCIA NSCLC-Radiogenomics; scan-rescan pairs) ->
   chest CT -> deep-learning segmentation -> vertebral-body levels -> tissue compartments -> outcomes.
b  Noise harmonisation: fat-window voxels on the native thin sharp-kernel image vs the harmonised image,
   and the liver attenuation distribution before/after (image-noise SD).
c  Posterior 3D surface rendering of segmented trunk muscle and skeleton; levels and the T5-T12 span.
d  Tissue compartments at T4, T8 and T12, identical scale, radiological orientation.
e  Muscle cross-sectional area on every slice from T12 to T4; shaded area = T5-T12 volume.
Example patient: public TCIA NSCLC-Radiogenomics (CC BY 3.0).

Usage: python figures/fig1_design_pipeline.py <case_id for a,c-e> <case_id for b>
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))
import style  # noqa: E402
from i18n import T  # noqa: E402
from bodycomp_metrics import Case, vertebra_info, labels, in_range, HU_FAT, VERTEBRAE, CAGE, ORGANS, INV  # noqa: E402

INV_T = INV["total"]

FIG = "Figure1"
HILITE = style.SERIES[1]  # orange: "voxels in the fat window" in panel b only (not a tissue identity)


def rp(*p):
    return os.path.join(style.ROOT, *p)


def disp(a):
    """RAS axial slice -> radiological display (anterior up, patient right on image left)."""
    return np.fliplr(np.rot90(a))


def window(img, level=40, width=400):
    return np.clip((img - (level - width / 2)) / width, 0, 1)


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def paint(gray, layers, alpha=0.65):
    rgb = np.repeat(gray[..., None], 3, axis=2)
    for m, col in layers:
        rgb[m] = (1 - alpha) * rgb[m] + alpha * hex_rgb(col)
    return rgb


def centred_crop(mask, half_h, half_w):
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    return slice(max(cy - half_h, 0), cy + half_h), slice(max(cx - half_w, 0), cx + half_w)


def image_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scalebar(ax, px_mm, length_mm=50, color="white"):
    """Scale bar at the lower right of an image axes (data coords are pixels)."""
    x1 = ax.get_xlim()[1]; y0 = ax.get_ylim()[0]
    w = length_mm / px_mm
    xe, ys = x1 - 0.06 * x1, y0 - 0.07 * y0
    ax.plot([xe - w, xe], [ys, ys], color=color, lw=1.0, solid_capstyle="butt")
    ax.text(xe - w / 2, ys - 0.025 * y0, f"{length_mm // 10} cm", color=color, fontsize=5, ha="center",
            va="bottom")


# ---------------------------------------------------------------- panel a: design strip
def cohort_counts():
    """Real counts from the public cohorts (files written by the pipeline and the analysis scripts)."""
    out = {}
    sel = rp("data", "nlst", "nlst_selected_series_last.csv")
    if os.path.exists(sel):
        s = pd.read_csv(sel)
        out["nlst"], out["nlst65"] = s.pid.nunique(), s[s.age_at_dx >= 65].pid.nunique()
    bc = rp("results", "public_rg_bodycomp.csv")
    if os.path.exists(bc):
        out["rg"] = len(pd.read_csv(bc))
    s2 = rp("results", "public_rg_scan2_bodycomp.csv")
    if os.path.exists(s2):
        out["pairs"] = len(pd.read_csv(s2))
    return out


def text_card(ax, title, lines, fc=style.WASH):
    image_ax(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec="none"))
    ax.text(0.06, 0.92, title, fontsize=6.2, fontweight="bold", va="top", color=style.INK)
    lines = [t for t in lines if t.strip()]  # translations can merge two English lines into one
    for i, t in enumerate(lines):
        ax.text(0.06, 0.74 - i * 0.135, t, fontsize=5.6, va="top", color=style.INK2)


def arrow_between(fig, ax0, ax1):
    p0, p1 = ax0.get_position(), ax1.get_position()
    y = (p0.y0 + p0.y1) / 2
    fig.add_artist(plt.matplotlib.patches.FancyArrowPatch(
        (p0.x1 + 0.004, y), (p1.x0 - 0.004, y), transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=6, lw=0.6, color=style.INK2))


def panel_b_axes(b1, b2, bh, case, z):
    """Native vs harmonised crops (fat-window voxels highlighted) + liver attenuation distributions."""
    body = disp(case.body[:, :, z])
    ys, xs = np.nonzero(body)
    y1, x0 = ys.max() + 4, max(xs.min() - 4, 0)
    h = int((ys.max() - ys.min()) * 0.62)
    crop = (slice(y1 - h, y1), slice(x0, x0 + int(h * 1.08)))
    liver = labels(case.segs["total"], "total", ["liver"])
    near = np.zeros(liver.shape[2], bool); near[max(z - 8, 0):z + 9] = True
    core = liver & near[None, None, :]
    rows, hist = [], []
    for ax, img3, title in [(b1, case.raw, T("Native (1.25 mm, lung kernel)")), (b2, case.ct, T("Harmonised"))]:
        im = disp(img3[:, :, z])
        fat = in_range(im, HU_FAT) & body
        ax.imshow(paint(window(im), [(fat, HILITE)], alpha=0.8)[crop], interpolation="nearest")
        image_ax(ax)
        ax.set_title(title, fontsize=6, pad=2.5, color=style.INK)
        vals = img3[core]
        sd = float(vals.std())
        ax.text(0.04, 0.04, T("noise SD {sd} HU").format(sd=f"{sd:.0f}"), transform=ax.transAxes, fontsize=5.2,
                color="white", bbox=dict(boxstyle="square,pad=0.25", fc="black", ec="none", alpha=0.75))
        rows.append(dict(image=title, liver_noise_sd_hu=round(sd, 1), fat_window_pixels_in_crop=int(fat[crop].sum())))
        hist.append(vals)
    bins = np.linspace(-300, 450, 151)
    mids = (bins[:-1] + bins[1:]) / 2
    dens = [np.histogram(v, bins=bins, density=True)[0] * 100 for v in hist]
    bh.axvspan(HU_FAT[0], HU_FAT[1], color=HILITE, alpha=0.14, lw=0)
    bh.plot(mids, dens[0], color=style.MUTED, lw=0.8)
    bh.plot(mids, dens[1], color=style.INK, lw=1.0)
    top = max(d.max() for d in dens) * 1.08
    bh.set_ylim(0, top)
    bh.text((HU_FAT[0] + HU_FAT[1]) / 2, top * 0.97, T("fat\nwindow"), fontsize=5, ha="center", va="top",
            color=style.INK2)
    pk = mids[np.argmax(dens[1])]
    bh.text(pk + 18, top * 0.8, T("harmonised"), fontsize=5.2, color=style.INK, va="center")
    bh.text(pk + 120, dens[0].max() * 1.6, T("native"), fontsize=5.2, color=style.INK2, va="bottom")
    frac = [float(((v >= HU_FAT[0]) & (v <= HU_FAT[1])).mean() * 100) for v in hist]
    bh.set_xlabel(T("Liver attenuation (HU)"), fontsize=6)
    bh.set_ylabel(T("Voxels (% per HU)"), fontsize=6)
    bh.set_xlim(bins[0], bins[-1])
    for r, f in zip(rows, frac):
        r["liver_voxels_in_fat_window_pct"] = round(f, 2)
    return pd.DataFrame(rows), pd.DataFrame({"hu_bin_mid": mids, "native_pct_per_hu": dens[0],
                                             "harmonised_pct_per_hu": dens[1]})


def panel_c_ax(ax, case, info):
    """Posterior 3D surface rendering of the segmented trunk muscles and skeleton (depth-map shading)."""
    from scipy import ndimage as ndi
    tot = case.segs["total"]
    bone = labels(tot, "total", CAGE + ["scapula_left", "scapula_right", "clavicula_left", "clavicula_right"])
    nz = tot.shape[2]
    z_lo, z_hi = 0, nz - 1
    mus, bon = case.muscle, bone
    solid = mus | bon
    hit = solid.any(axis=1)                               # (x, z)
    first = np.argmax(solid, axis=1)                      # first voxel seen from posterior (+axis1 = anterior)
    is_bone = np.take_along_axis(bon, first[:, None, :], axis=1)[:, 0, :] & hit
    depth = np.where(hit, first * case.zooms[1], np.nan)
    fill = np.nan_to_num(depth, nan=np.nanmax(depth))
    d = ndi.gaussian_filter(fill, sigma=(1.2, 1.2 * case.zooms[0] / case.zooms[2]))
    gx = np.gradient(d, case.zooms[0], axis=0)
    gz = np.gradient(d, case.zooms[2], axis=1)
    nrm = np.stack([-gx, -np.ones_like(d), -gz], axis=-1)   # surface normal toward the posterior viewer
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    light = np.array([-0.45, -1.0, 0.6]); light /= np.linalg.norm(light)
    lam = np.clip((nrm * light).sum(-1), 0, 1)
    view = np.array([0.0, -1.0, 0.0])
    half = (light + view) / np.linalg.norm(light + view)
    spec = np.clip((nrm * half).sum(-1), 0, 1) ** 24
    shade = 0.16 + 0.84 * lam
    zz = np.arange(nz)[None, :].repeat(hit.shape[0], 0)
    in_span = (zz >= info["T12"]) & (zz <= info["T5"])
    muscle_col = np.where(in_span[..., None], hex_rgb(style.TISSUE["muscle"]), np.array([0.86, 0.78, 0.77]))
    base = np.where(is_bone[..., None], np.array([0.90, 0.89, 0.86]), muscle_col)
    rgb = np.where(hit[..., None], np.clip(base * shade[..., None] + 0.18 * spec[..., None], 0, 1), 1.0)
    img = np.flipud(np.transpose(rgb, (1, 0, 2)))          # rows = z (superior up); posterior view: patient R on right
    cols = np.nonzero(hit.any(axis=1))[0]
    img = img[:, max(cols.min() - 3, 0):cols.max() + 4]
    ax.imshow(img, aspect=case.zooms[2] / case.zooms[0], interpolation="bilinear")
    ax.set_anchor("C")
    image_ax(ax)
    H_, W_ = img.shape[:2]
    Y = lambda lv: z_hi - info[lv]
    for lv in ["T4", "T8", "T12"]:
        ax.plot([-0.06 * W_, -0.01 * W_], [Y(lv)] * 2, color=style.INK, lw=0.6, clip_on=False)
        ax.text(-0.09 * W_, Y(lv), lv, fontsize=5.5, ha="right", va="center", color=style.INK)
    xb = 1.04 * W_
    ax.plot([xb, xb], [Y("T5"), Y("T12")], color=style.INK, lw=0.7, clip_on=False)
    for lv in ("T5", "T12"):
        ax.plot([xb - 0.03 * W_, xb], [Y(lv)] * 2, color=style.INK, lw=0.7, clip_on=False)
    ax.text(xb + 0.04 * W_, (Y("T5") + Y("T12")) / 2, T("T5–T12 volume (red)"), fontsize=5.5, va="center",
            ha="left", rotation=90, color=style.INK, clip_on=False)
    ax.text(0.02, -0.02, T("L"), transform=ax.transAxes, fontsize=5.5, va="top", color=style.INK2)
    ax.text(0.98, -0.02, T("R"), transform=ax.transAxes, fontsize=5.5, va="top", ha="right", color=style.INK2)
    ax.set_title(T("Posterior 3D view"), fontsize=6, pad=2.5)
    return pd.DataFrame([dict(level=k, z_index=round(v, 1)) for k, v in info.items()])


def panel_d_axes(fig, axes, case, levels, legend=True):
    zs, masks = [], []
    for lv in levels:
        zc, _ = vertebra_info(case.segs["total"], lv)
        zs.append(int(round(zc)))
        masks.append({k: disp(v) for k, v in case.slice_masks(zs[-1]).items()})
    ext = [np.nonzero(m["body"]) for m in masks]
    hh = int(max(e[0].max() - e[0].min() for e in ext) / 2) + 6
    hw = int(max(e[1].max() - e[1].min() for e in ext) / 2) + 6
    rows = []
    for j, (ax, lv, z, md) in enumerate(zip(axes, levels, zs, masks)):
        cr = centred_crop(md["body"], hh, hw)
        img = paint(window(disp(case.ct[:, :, z])),
                    [(md["sat"], style.TISSUE["sat"]), (md["ift"], style.TISSUE["ift"]),
                     (md["muscle"], style.TISSUE["muscle"]), (md["imat"], style.TISSUE["imat"])])
        ax.imshow(img[cr], interpolation="nearest", aspect="auto")
        image_ax(ax)
        ax.text(0.04, 0.95, lv, transform=ax.transAxes, fontsize=6.5, fontweight="bold", color="white", va="top")
        m = case.slice_masks(z)
        area = {k: float(m[k].sum() * case.px_cm2) for k in ["muscle", "imat", "sat", "ift"]}
        smd = float(case.raw[:, :, z][m["muscle"]].mean())
        ax.text(0.5, -0.06, T("Muscle {a} cm², {b} HU").format(a=f"{area['muscle']:.0f}", b=f"{smd:.0f}"), transform=ax.transAxes,
                fontsize=5.5, ha="center", va="top", color=style.INK)
        if j == 0:
            scalebar(ax, case.zooms[0], 50)
        rows.append(dict(level=lv, z_index=z, **{f"{k}_cm2": round(v, 2) for k, v in area.items()},
                         muscle_hu=round(smd, 1)))
    if not legend:
        return pd.DataFrame(rows)
    handles = [Line2D([0], [0], marker="s", ls="", ms=4.2, mfc=style.TISSUE[k], mec="none",
                      label=T(style.TISSUE_LABEL[k])) for k in ["muscle", "imat", "sat", "ift"]]
    p0, p1 = axes[0].get_position(), axes[-1].get_position()
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=5.6, handletextpad=0.3, columnspacing=1.0,
               labelspacing=0.35, bbox_to_anchor=((p0.x0 + p1.x1) / 2, p0.y0 - 0.045), bbox_transform=fig.transFigure)
    return pd.DataFrame(rows)


def mm_ax(fig, x, y, w, h, W=183.0, H=118.0):
    """Axes placed in millimetres from the lower-left corner of a W x H mm figure."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def muscle_profile(case, info):
    """Muscle cross-sectional area on every slice from 1 cm below T12 to 1 cm above T4 (harmonised)."""
    z0 = max(int(info["T12"] - 10 / case.zooms[2]), 0)
    z1 = min(int(info["T4"] + 10 / case.zooms[2]), case.ct.shape[2] - 1)
    rows = []
    for z in range(z0, z1 + 1):
        m = case.slice_masks(z)["muscle"]
        rows.append(dict(z_index=z, z_mm=(z - info["T12"]) * case.zooms[2], muscle_cm2=m.sum() * case.px_cm2))
    return pd.DataFrame(rows)


def panel_e(ax, case, info, prof):
    col = style.TISSUE["muscle"]
    x = prof["z_mm"] / 10.0
    ax.plot(x, prof["muscle_cm2"], color=col, lw=1.0)
    span = (prof["z_index"] >= info["T12"]) & (prof["z_index"] <= info["T5"])
    ax.fill_between(x[span], 0, prof["muscle_cm2"][span], color=col, alpha=0.12, lw=0)
    vol = prof["muscle_cm2"][span].sum() * case.zooms[2] / 10.0
    ymax = prof["muscle_cm2"].max() * 1.18
    for lv in ["T4", "T8", "T12"]:
        xv = (info[lv] - info["T12"]) * case.zooms[2] / 10.0
        yv = float(np.interp(xv, x, prof["muscle_cm2"]))
        ax.plot([xv], [yv], "o", ms=3.2, mfc=col, mec="white", mew=0.8, zorder=3)
        ax.text(xv, yv + ymax * 0.05, lv, fontsize=5.5, ha="center", va="bottom", color=style.INK)
    xm = (info["T5"] + info["T12"]) / 2
    xm = (xm - info["T12"]) * case.zooms[2] / 10.0
    ax.text(xm, ymax * 0.12, T("T5–T12 volume\n{v} cm³").format(v=f"{vol:,.0f}"), fontsize=5.5, ha="center", va="bottom",
            color=style.INK)
    ax.set_ylim(0, ymax)
    ax.set_xlim(x.min(), x.max())
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.set_xlabel(T("Distance above T12 centre (cm)"), fontsize=6)
    ax.set_ylabel(T("Muscle area (cm²)"), fontsize=6)
    return vol


def crop_alpha(img, pad=6):
    a = img[..., 3] > 0.02
    ys, xs = np.nonzero(a)
    y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad
    return img[y0:y1, x0:x1], (x0, y0)


def w_data(ax, t, rend):
    """Width of a text artist in the axes' data units."""
    bb = t.get_window_extent(rend)
    inv = ax.transData.inverted()
    return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]


def panel_overview(fig, A):
    """a: the three public cohorts, what was measured and what each one is used for."""
    nl = pd.read_csv(rp("results", "nlst", "flow.csv")).set_index("step").n
    pc = pd.read_csv(rp("results", "public_pooled", "cohort_description.csv")).set_index("cohort")
    n_rg = len(pd.read_csv(rp("results", "public_rg_bodycomp.csv")))
    n_pairs = len(pd.read_csv(rp("results", "public_rg_scan2_bodycomp.csv")))
    tc = pd.read_csv(rp("results", "tcga_cptac_bodycomp.csv"))
    qc = pd.read_csv(rp("results", "tcga_cptac_qc_review.csv"))
    n_tc = len(tc) - int((qc.qc_pass == 0).sum())  # biopsy-guidance CTs excluded at visual QC
    n_tc_an = int(pc.loc[["CPTAC", "TCGA-LUSC", "TCGA-LUAD"], "n"].sum())
    cards = [
        ("NLST · CT arm", "Last screening low-dose CT before diagnosis",
         f"{int(nl.iloc[2]):,}", "resected, pathologically staged cancers",
         [f"{int(nl.iloc[3])} aged ≥65 y", f"{int(nl.iloc[0])} CTs retrieved"],
         [("Age and sex gradients", True), ("Field-of-view audit", True)]),
        ("TCIA NSCLC-Radiogenomics", "Preoperative staging CT",
         f"{int(pc.loc['TCIA-RG', 'n'])}", "resected NSCLC with follow-up",
         [f"{n_rg} CTs measured", f"{n_pairs} scan–rescan pairs"],
         [("Technical validation", True), ("Overall survival", True)]),
        ("TCGA-LUAD/LUSC · CPTAC", "Preoperative diagnostic CT",
         f"{n_tc_an}", "resected NSCLC with follow-up",
         [f"{n_tc} CTs measured", "3 cohorts"],
         [("Overall survival", True), ("Leave-one-cohort-out", True)]),
    ]
    rows = []
    rend = fig.canvas.get_renderer()
    CH = 25.0
    for i, (name, ct, big, what, sub, tags) in enumerate(cards):
        ax = A(3 + i * 60.3, 115.5, 57.5, CH)
        image_ax(ax); ax.set_xlim(0, 57.5); ax.set_ylim(0, CH)
        ax.add_patch(FancyBboxPatch((0, 0), 57.5, CH, boxstyle="round,pad=0,rounding_size=1.6", fc="#f4f3ee",
                                    ec="none", zorder=0))
        ax.text(2.4, CH - 2.2, name, fontsize=6.3, fontweight="bold", va="top", color=style.INK)
        ax.text(2.4, CH - 5.6, ct, fontsize=5.2, va="top", color=style.INK2)
        tb = ax.text(2.4, 9.6, big, fontsize=15, fontweight="bold", va="baseline", color=style.INK)
        wbig = w_data(ax, tb, rend)
        ax.text(2.4 + wbig + 1.6, 13.2, what, fontsize=5.2, va="baseline", color=style.INK)
        ax.text(2.4 + wbig + 1.6, 9.8, "  ·  ".join(sub), fontsize=5.2, va="baseline", color=style.INK2)
        x = 2.4
        for t, done in tags:
            tt = ax.text(x + 1.5, 3.9, t, fontsize=5.0, ha="left", va="center", color=style.INK, zorder=3)
            w = w_data(ax, tt, rend) + 3.0
            ax.add_patch(FancyBboxPatch((x, 1.6), w, 4.6, boxstyle="round,pad=0,rounding_size=2.2",
                                        fc="#e6e5dd" if done else "white", ec="none" if done else style.MUTED,
                                        lw=0.5, ls=(0, (2, 1.2)), zorder=2))
            x += w + 1.2
        rows.append(dict(cohort=name, ct=ct, analysed=big, details="; ".join(sub), uses="; ".join(t for t, _ in tags)))
    return ax, pd.DataFrame(rows)


def seg_thumbnail(case, z, cr):
    """Deep-learning masks on one axial slice: muscle and skeleton filled, organs outlined."""
    from skimage.segmentation import find_boundaries
    tot = disp(case.segs["total"][:, :, z])
    g = window(disp(case.raw[:, :, z]))
    org = np.where(np.isin(tot, [INV_T[n] for n in ORGANS if n in INV_T]), tot, 0)
    edge = find_boundaries(org, mode="inner")
    rgb = paint(g, [(disp(labels(case.segs["total"], "total", CAGE)[:, :, z]), "#f2ecdc"),
                    (disp(case.muscle[:, :, z]), style.TISSUE["muscle"])], alpha=0.75)
    rgb[edge] = 0.35 * rgb[edge] + 0.65
    return rgb[cr]


def panel_pipeline(fig, A, case, info):
    """b: numbered processing steps on the example CT."""
    tot = case.segs["total"]
    z8, z12 = int(round(info["T8"])), int(round(info["T12"]))
    cr = centred_crop(disp(case.body[:, :, z8]), 150, 190)
    y0, w, h, gap = 80, 29.0, 23.0, 7.0  # b row
    xs = [3 + i * (w + gap) for i in range(5)]
    axs = [A(x, y0, w, h) for x in xs]
    axs[0].imshow(window(disp(case.raw[:, :, z8]))[cr], cmap="gray", interpolation="lanczos")
    axs[1].imshow(seg_thumbnail(case, z8, cr), interpolation="lanczos")
    xmid = int(np.median(np.nonzero(case.cord.any(axis=(1, 2)))[0]))
    nz, ny = tot.shape[2], tot.shape[1]
    zlo, zhi = max(int(info["T12"] - 40 / case.zooms[2]), 0), min(int(info["T4"] + 40 / case.zooms[2]), nz - 1)
    sag = np.flipud(window(case.raw[xmid, :, zlo:zhi + 1], 300, 1500).T)[:, ::-1]
    axs[2].imshow(sag, cmap="gray", aspect="auto", interpolation="lanczos")
    for lv in ["T4", "T8", "T12"]:
        yy = zhi - info[lv]
        axs[2].axhline(yy, color=style.TISSUE["muscle"], lw=0.6)
        axs[2].text(ny * 0.035, yy - 3, lv, color="white", fontsize=5, va="bottom", fontweight="bold")
    m12 = {k: disp(v) for k, v in case.slice_masks(z12).items()}
    cr12 = centred_crop(m12["body"], 150, 190)
    axs[3].imshow(paint(window(disp(case.ct[:, :, z12])),
                        [(m12["sat"], style.TISSUE["sat"]), (m12["ift"], style.TISSUE["ift"]),
                         (m12["muscle"], style.TISSUE["muscle"]), (m12["imat"], style.TISSUE["imat"])])[cr12],
                  interpolation="lanczos")
    ax = axs[4]
    image_ax(ax); ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.add_patch(FancyBboxPatch((0, 0), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc="#f4f3ee", ec="none"))
    items = [("muscle", "Muscle area, volume, density"), ("imat", "Intramuscular fat %"),
             ("sat", "Subcutaneous fat"), ("ift", "Visceral/mediastinal fat"), (None, "Trabecular bone density")]
    for i, (k, t) in enumerate(items):
        yy = h - 4.0 - i * 4.1
        ax.add_patch(Rectangle((2.0, yy - 1.1), 2.2, 2.2, fc=style.TISSUE[k] if k else "#cfc8b4", ec="none"))
        ax.text(5.4, yy, t, fontsize=5.0, va="center", color=style.INK)
    titles = ["Routine chest CT", "Deep-learning segmentation", "Vertebral-body levels", "Tissue compartments",
              "Per-patient measures"]
    subs = ["staging or screening, any kernel", "TotalSegmentator, open tasks", "T4–T12 bodies, not the L3 slice",
            "fat −190 to −30 HU; muscle −29 to 150 HU", "single level (T12) and T5–T12 volume"]
    for i, (a, t, sb) in enumerate(zip(axs, titles, subs)):
        if i < 4:
            image_ax(a)
        a.text(0, 1.0 + 3.2 / h, t, transform=a.transAxes, fontsize=6, fontweight="bold", va="bottom",
               color=style.INK)
        a.text(0, -1.6 / h, sb, transform=a.transAxes, fontsize=5, va="top", color=style.INK2)
    for a0, a1 in zip(axs[:-1], axs[1:]):
        p0, p1 = a0.get_position(), a1.get_position()
        yv = (p0.y0 + p0.y1) / 2
        fig.add_artist(plt.matplotlib.patches.FancyArrowPatch(
            (p0.x1 + 1.3 / 183, yv), (p1.x0 - 1.3 / 183, yv), transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=6, lw=0.7, color=style.INK2))
    return axs


def panel_3d(fig, A, case_id):
    """c: VTK surface renderings (figures/render3d.py) with level ticks at the true vertebral-body centroids."""
    import json
    meta = json.load(open(rp("figures", "assets", f"render3d_{case_id}.json")))
    out = []
    # posterior-oblique musculoskeletal view
    v = meta["views"]["msk"]
    img = plt.imread(rp(v["file"]))
    img, (ox, oy) = crop_alpha(img)
    wm = 47.0
    hm = wm * img.shape[0] / img.shape[1]
    ax = A(8, 16.5, wm, hm)
    ax.imshow(img, interpolation="lanczos")
    image_ax(ax)
    lv = {k: (x - ox, y - oy) for k, (x, y) in v["levels_px"].items()}
    W_ = img.shape[1]
    for k in ["T4", "T8", "T12"]:
        ax.plot([-0.07 * W_, -0.02 * W_], [lv[k][1]] * 2, color=style.INK, lw=0.6, clip_on=False)
        ax.text(-0.09 * W_, lv[k][1], k, fontsize=5.5, ha="right", va="center", color=style.INK)
    xb = 1.03 * W_
    ax.plot([xb, xb], [lv["T5"][1], lv["T12"][1]], color=style.INK, lw=0.6, clip_on=False)
    for k in ("T5", "T12"):
        ax.plot([xb - 0.025 * W_, xb], [lv[k][1]] * 2, color=style.INK, lw=0.6, clip_on=False)
    ax.text(xb + 0.035 * W_, (lv["T5"][1] + lv["T12"][1]) / 2, "T5–T12", fontsize=5.5, rotation=90, ha="left",
            va="center", color=style.INK)
    ax.text(0.5, 1.02, "Posterior oblique", transform=ax.transAxes, fontsize=6, ha="center", va="bottom",
            color=style.INK)
    out.append(dict(view="posterior oblique", **{f"{k}_y_px": round(p[1], 1) for k, p in lv.items()}))
    # T5-T12 slab
    v = meta["views"]["slab"]
    img, _ = crop_alpha(plt.imread(rp(v["file"])))
    ws = 38.0
    hs = ws * img.shape[0] / img.shape[1]
    ax2 = A(60, 16.5 + hm / 2 - hs / 2, ws, hs)
    ax2.imshow(img, interpolation="lanczos")
    image_ax(ax2)
    ax2.text(0.5, 1.01, "T5–T12 slab, anterior superior", transform=ax2.transAxes, fontsize=6, ha="center",
             va="bottom", color=style.INK)
    # shared tissue key
    ax3 = A(4, 3, 94, 11)
    image_ax(ax3); ax3.set_xlim(0, 94); ax3.set_ylim(0, 11)
    items = [(style.TISSUE["muscle"], "Skeletal muscle, T5–T12"), ("#cdbfbb", "Muscle outside T5–T12"),
             ("#e9e2cf", "Bone"), (style.TISSUE["sat"], "Subcutaneous fat"),
             (style.TISSUE["ift"], "Visceral/mediastinal fat"), (style.TISSUE["imat"], "Intramuscular fat")]
    for i, (c, t) in enumerate(items):
        xx, yy = 4 + (i % 3) * 31, 8 - (i // 3) * 4.6
        ax3.add_patch(Rectangle((xx, yy - 1.2), 2.4, 2.4, fc=c, ec="none"))
        ax3.text(xx + 3.4, yy, t, fontsize=5.2, va="center", color=style.INK)
    return ax, pd.DataFrame(out)


def main(case_id, noise_case_id=None):
    style.setup()
    case = Case(rp("data", "nifti", "public_rg", f"{case_id}.nii.gz"), rp("data", "seg", "public_rg", case_id))
    info = {}
    for lv in ["T4", "T5", "T8", "T12"]:
        zc, ok = vertebra_info(case.segs["total"], lv)
        assert zc is not None and ok, f"{lv} not fully imaged in {case_id}"
        info[lv] = zc
    W, H = 183.0, 143.5
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    A = lambda *a: mm_ax(fig, *a, W=W, H=H)

    ax_a, sa = panel_overview(fig, A)
    axs_b = panel_pipeline(fig, A, case, info)
    ax_c, sc = panel_3d(fig, A, case_id)
    dax = [A(101 + i * 27.3, 42, 26, 21) for i in range(3)]
    sd = panel_d_axes(fig, dax, case, ["T4", "T8", "T12"], legend=False)
    ax_e = A(111, 8.5, 69, 22)
    prof = muscle_profile(case, info)
    panel_e(ax_e, case, info, prof)

    for t, x, y in [("a", 1, 143), ("b", 1, 112), ("c", 1, 70.5), ("d", 99, 70.5), ("e", 99, 35)]:
        style.panel_label(None, t, x / W, y / H, fig=fig)
    style.source_data(sa, FIG, "a_cohorts")
    style.source_data(sc, FIG, "c_levels"); style.source_data(sd, FIG, "d_compartments")
    style.source_data(prof, FIG, "e_muscle_profile")
    print(style.save(fig, FIG))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "R01-049")
