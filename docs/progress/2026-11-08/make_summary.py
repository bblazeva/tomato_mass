"""
Build the November 2026 progress summary for the supervisor (summary_2026-11-08.pdf) and its figures.

Every number in the text was copied from these result files (see investigation/investigation_6_11.md):
  results/eval/cv5_20k_summary/report.txt          known sessions (5-fold cross-validation by cluster)
  results/eval/loso_20k_summary/report.txt         new sessions (leave one session out)
  results/eval/real_area_predicted_masks/report.txt
  results/eval/depth_geometry_test/report.txt (+ ..._report.txt for model 4.2)
  results/eval/depth_structure_test/report.txt
  results/eval/cv5_20k_background/report.txt
  results/eval/dataset_audit.txt
Figures are drawn from the images, analysis/dataset_by_mass.csv and analysis/cluster_inventory.csv.

Needs reportlab (not in the thesis env). Run from the repo root:
  <python with reportlab, matplotlib, opencv> docs/progress/2026-11-08/make_summary.py
"""
import csv
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

OUT = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(OUT, "summary_2026-11-08.pdf")
SESSION = {"2025-07-13": "Home", "2023-12-01": "Lab 1", "2025-03-28": "Lab 2"}
SESSION_COLOR = {"Home": "#2a78d6", "Lab 1": "#1baf7a", "Lab 2": "#eb6834"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"

pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "figure.facecolor": "white"})


# ====================================================================================
# FIGURES
# ====================================================================================

def coco_records():
    with open("data_min120/gt_json/all/all.json") as fh:
        coco = json.load(fh)
    boxes = {a["image_id"]: a["bbox"] for a in coco["annotations"]}
    return [(i["file_name"], boxes[i["id"]]) for i in coco["images"]]


def middle_frame(records, mass, session):
    frames = sorted(r for r in records if r[0].startswith(f"{mass}_{session}"))
    return frames[len(frames) // 2]


def fig_sessions(records):
    picks = [(940, "2025-07-13", "Home: normal tomatoes, 940 g"), (208, "2025-07-13", "Home: cherry tomatoes, 208 g"),
             (150, "2023-12-01", "Lab 1: cherry tomatoes, 150 g"), (697, "2025-03-28", "Lab 2: normal tomatoes, 697 g")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.2))
    for ax, (mass, session, title) in zip(axes, picks):
        name, _ = middle_frame(records, mass, session)
        ax.imshow(cv2.cvtColor(cv2.imread(f"data_min120/images/all/{name}"), cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_sessions.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_four_tomatoes(records):
    picks = [(940, "2025-07-13", "Home, 940 g"), (364, "2023-12-01", "Lab 1, 364 g"), (368, "2025-03-28", "Lab 2, 368 g")]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    for ax, (mass, session, title) in zip(axes, picks):
        name, (x, y, w, h) = middle_frame(records, mass, session)
        image = cv2.imread(f"data_min120/images/all/{name}")
        pad = 25
        crop = image[max(0, int(y) - pad):int(y + h) + pad, max(0, int(x) - pad):int(x + w) + pad]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"4 tomatoes: {title}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_four_tomatoes.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_dataset():
    with open("analysis/dataset_by_mass.csv") as fh:
        rows = sorted((int(r["mass_g"]), SESSION[r["session"]], int(r["n_images"])) for r in csv.DictReader(fh)
                      if int(r["mass_g"]) >= 120)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    x = np.arange(len(rows))
    ax.bar(x, [n for _, _, n in rows], width=0.8, color=[SESSION_COLOR[s] for _, s, _ in rows])
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m, _, _ in rows], rotation=90, fontsize=7)
    ax.set_xlim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("cluster mass (g), in order of mass")
    ax.set_ylabel("images")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SESSION_COLOR[s]) for s in ("Home", "Lab 1", "Lab 2")]
    ax.legend(handles, ["Home", "Lab 1", "Lab 2"], frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_dataset.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_real_area():
    sessions = ["Home", "Lab 1", "Lab 2"]
    series = [("Current model (mass head)", [30.9, 249.0, 37.8], "#6b6a64"),
              ("Real area, tomato type unknown", [47.0, 57.9, 18.2], "#eb6834"),
              ("Real area, tomato type known", [19.5, 21.0, 22.4], "#2a78d6")]
    cap = 80
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    x = np.arange(len(sessions))
    width = 0.26
    for j, (label, values, color) in enumerate(series):
        shown = [min(v, cap) for v in values]
        bars = ax.bar(x + (j - 1) * width, shown, width - 0.03, color=color, label=label)
        for b, v in zip(bars, values):
            text = f"{v:.0f}%" + (" (off scale)" if v > cap else "")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, text, ha="center", va="bottom",
                    fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(never seen in training)" for s in sessions], fontsize=8)
    ax.set_ylim(0, cap + 12)
    ax.set_ylabel("average error (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_real_area.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ====================================================================================
# DOCUMENT
# ====================================================================================

BASE = ParagraphStyle("base", fontName="DejaVu", fontSize=9.5, leading=13.5, textColor=colors.HexColor(INK), alignment=TA_LEFT)
SMALL = ParagraphStyle("small", parent=BASE, fontSize=8, leading=11, textColor=colors.HexColor("#4a4944"))
H1 = ParagraphStyle("h1", parent=BASE, fontName="DejaVu-Bold", fontSize=16, leading=20, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=BASE, fontName="DejaVu-Bold", fontSize=12, leading=16, spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("h3", parent=BASE, fontName="DejaVu-Bold", fontSize=10, leading=14, spaceBefore=6, spaceAfter=2)
BULLET = ParagraphStyle("bullet", parent=BASE, leftIndent=12, bulletIndent=2)
CELL = ParagraphStyle("cell", parent=BASE, fontSize=8.5, leading=11)
CELL_BOLD = ParagraphStyle("cellb", parent=CELL, fontName="DejaVu-Bold")


def p(text, style=BASE):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BULLET, bulletText="•") for t in items]


def table(rows, widths, bold_first_col=False, highlight_rows=()):
    data = []
    for i, row in enumerate(rows):
        data.append([Paragraph(str(c), CELL_BOLD if i == 0 or (bold_first_col and j == 0) else CELL) for j, c in enumerate(row)])
    t = Table(data, colWidths=[w * cm for w in widths], hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c8c2")),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efeee9")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for r in highlight_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e8f0fb")))
    t.setStyle(TableStyle(style))
    return t


def box(flowables):
    t = Table([[flowables]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#2a78d6")),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f8fd")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def image(path, width_cm):
    h, w = cv2.imread(path).shape[:2]
    return Image(path, width=width_cm * cm, height=width_cm * cm * h / w)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(2 * cm, 1.2 * cm, "Tomato cluster mass from RGB-D: progress summary, November 2026")
    canvas.drawRightString(19 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


def build():
    records = coco_records()
    f_sessions, f_four, f_dataset, f_area = fig_sessions(records), fig_four_tomatoes(records), fig_dataset(), fig_real_area()

    s = []
    s.append(p("Estimating tomato cluster mass from RGB-D images", H1))
    s.append(p("What the model learns, and what transfers to new capture conditions. Progress summary, November 2026.", SMALL))
    s.append(Spacer(1, 8))
    s.append(box([p("<b>Main findings</b>", BASE)] + bullets([
        "On <b>new clusters from capture sessions the model has seen</b>, the average error is about <b>19%</b>.",
        "On a <b>capture session the model has never seen</b>, the error rises to <b>31%, 38% and 249%</b>.",
        "The model <b>does not use depth as a measurement of size</b>. With colour present, changing the depth values does not change "
        "the prediction, and colour + depth is not more accurate than colour alone.",
        "A simple physical measurement, the cluster's <b>real visible area</b> (from the model's own mask and the depth map), "
        "transfers to unseen sessions with about <b>20% error</b>, when the tomato type (cherry or normal) is known.",
    ]) + [Spacer(1, 4), p("<b>Proposal:</b> one final training experiment in which the network uses this real area directly "
                          "(section 5), then writing up.", BASE)]))
    s.append(Spacer(1, 8))
    s.append(p("<b>Terms used</b>", H3))
    s.extend(bullets([
        "<b>Cluster:</b> one physical tomato bunch, weighed once and photographed 5 to 64 times.",
        "<b>Error:</b> the average, over images, of |predicted mass - true mass| / true mass, in % (mean absolute percentage error).",
        "<b>Colour + depth / colour only / depth only:</b> the same trained model, given both inputs, or with one input replaced by "
        "its average value. Models were trained with colour removed in half of the training images, so depth only works too.",
        "<b>Model:</b> Mask R-CNN (ResNet-50 with FPN) with an added mass head, colour + depth input. All models below use the same "
        "settings and 20,000 training iterations.",
    ]))

    s.append(p("1. Data", H2))
    s.append(p("1,122 images of 46 clusters from three capture sessions. Clusters lighter than 120 g (7 clusters, 68 images) were "
               "left out.", BASE))
    s.append(Spacer(1, 4))
    s.append(table([
        ["Session", "Type", "Clusters", "Images", "Mass", "Tomatoes per cluster", "Mass per tomato", "Camera distance"],
        ["Home", "normal", "11", "476", "375-1160 g", "2-5", "188-283 g", "337-1371 mm"],
        ["Home", "cherry", "3", "97", "164-239 g", "8-12", "20-21 g", "350-927 mm"],
        ["Lab 1", "normal", "4", "56", "183-430 g", "2-5", "86-93 g", "439-509 mm"],
        ["Lab 1", "cherry", "7", "77", "122-165 g", "6-10", "15-21 g", "441-519 mm"],
        ["Lab 2", "normal", "21", "416", "186-1029 g", "2-9", "92-140 g", "472-685 mm"],
    ], [1.8, 1.5, 1.9, 1.8, 2.2, 2.3, 2.3, 2.5]))
    s.append(Spacer(1, 6))
    s.append(KeepTogether([image(f_sessions, 17), p("One example image per session and tomato type.", SMALL)]))
    s.append(Spacer(1, 4))
    s.append(KeepTogether([image(f_dataset, 17),
                           p("Number of images per cluster, clusters in order of mass, coloured by session.", SMALL)]))

    s.append(p("2. How the models were tested", H2))
    s.extend(bullets([
        "<b>Known sessions (5-fold cross-validation by cluster):</b> the 46 clusters were split into 5 groups of about 9, each mixing "
        "all sessions. Five models were trained, each on 4 groups and tested on the fifth. Every image is tested once, by a model "
        "that never saw its cluster, but did see other clusters from the same session.",
        "<b>New sessions (leave one session out):</b> three models, each trained on two sessions and tested on the third.",
    ]))

    s.append(p("3. Results", H2))
    s.append(p("3.1 The model works on known sessions, and fails on new ones", H3))
    s.append(table([
        ["Session tested", "Error, session in training", "Error, session never seen", "Note"],
        ["Home", "19.6%", "30.9%", "9 of 573 images not detected (session never seen)"],
        ["Lab 1", "15.7%", "249%", "63 of 133 images are lighter than any training cluster"],
        ["Lab 2", "19.0%", "37.8%", "An earlier model trained on Home only: 66.8%"],
        ["All images", "18.9%", "", "Per test group: 13.0% to 23.3%"],
    ], [2.6, 3.6, 3.6, 7.2]))
    s.append(Spacer(1, 4))
    s.append(p("The Lab 1 result is driven by its cherry clusters: when Lab 1 is never seen, they are predicted on average about "
               "4 times too heavy. Their colour differs strongly from the Home cherries (measured colour saturation 116-139 in Lab 1, "
               "222-237 in the other sessions, on a 0-255 scale).", BASE))

    s.append(p("3.2 Depth does not make the model more accurate", H3))
    s.append(table([
        ["Test", "Colour + depth", "Colour only", "Depth only"],
        ["Known sessions (all 1,122 images)", "18.9%", "19.2%", "20.9%"],
        ["Home never seen", "30.9%", "32.8%", "39.2% (312 of 573 images not detected)"],
        ["Lab 1 never seen", "249%", "243%", "155%"],
        ["Lab 2 never seen", "37.8%", "36.4%", "53.3%"],
    ], [5.2, 3, 3, 5.8]))
    s.append(Spacer(1, 4))
    s.append(p("On the known sessions, the difference between colour + depth and colour only is -0.4 percentage points "
               "(95% interval over clusters: -1.3 to +0.5), so no measurable benefit.", BASE))

    s.append(CondPageBreak(7 * cm))
    s.append(p("3.3 The model does not measure size", H3))
    s.append(p("If the model computed the cluster's real size from depth, two simple changes to the input should raise the predicted "
               "mass by about <b>1.8 times</b> (1.7 to 2.0 depending on how mass scales with area): multiplying every depth value by "
               "1.25 (\"everything is 25% farther away\"), or enlarging the cluster in the image by 25% while keeping depth unchanged.", BASE))
    s.append(Spacer(1, 4))
    s.append(table([
        ["Change to the input", "If it measured size", "Colour + depth", "Depth only", "Colour only"],
        ["All depth values x1.25", "about x1.8", "x1.00", "x1.08", "(no depth)"],
        ["Cluster enlarged 25%, depth unchanged", "about x1.8", "x1.05", "x1.15", "x1.06"],
    ], [5.6, 3, 2.8, 2.8, 2.8]))
    s.append(p("Median factor over 46 clusters (275 images, the 5 cross-validation models). A model trained without zoom "
               "augmentation behaves the same with colour + depth (depth x1.25: x1.00 to x1.01; cluster enlarged: x1.01 to x1.05). These changed images do not occur in "
               "real data, so they show how the model reacts, not how accurate it is.", SMALL))
    s.append(Spacer(1, 6))
    s.append(p("What the model uses instead:", BASE))
    s.extend(bullets([
        "<b>Colour + depth:</b> replacing only the cluster's depth with another cluster's depth changes the prediction by "
        "1.3% (median), almost the same as replacing it with another photo of the same cluster (1.0%).",
        "<b>Colour only:</b> blanking the background raises the error from 19.2% to 24.6%, so most of the information comes from "
        "the cluster itself, not the scene around it.",
        "<b>Depth only:</b> a flat silhouette of the cluster (no surface detail, background removed) still gives 29.5% error, "
        "against 21.4% with the real depth map and 78.7% for always guessing the average mass. Flattening the cluster's surface "
        "adds only +0.5 percentage points once the background is removed.",
    ]))
    s.append(p("So the model judges mass from the appearance of the cluster (in colour, or from its outline in depth), not "
               "from measured size.", BASE))

    s.append(p("3.4 A physical measurement transfers to new sessions", H3))
    s.append(p("<b>Real visible area:</b> each pixel of the cluster's mask covers a real area that grows with the square of its "
               "distance, which the depth map gives. Adding this up over the mask gives the cluster's visible area in cm<super>2</super>, "
               "independent of how far away it is. A simple relation between this area and mass was fitted on two sessions and applied "
               "to the third, using masks from the model that never saw that session (median overlap with the hand-drawn masks: "
               "0.94 to 0.95). Cherry and normal tomatoes need separate relations: about 2.3-2.4 g per cm<super>2</super> of "
               "visible area for cherries, 5.3-6.4 g for normal tomatoes.", BASE))
    s.append(Spacer(1, 4))
    s.append(KeepTogether([image(f_area, 15)]))
    s.append(table([
        ["Session never seen", "Current model", "Real area, type unknown", "Real area, type known"],
        ["Home", "30.9%", "47.0%", "19.5%"],
        ["Lab 1", "249%", "57.9%", "21.0%"],
        ["Lab 2", "37.8%", "18.2%", "22.4%"],
    ], [4.2, 3.6, 4.4, 4.4]))
    s.append(p("The tomato type was taken from hand labels. Lab 2 has only normal tomatoes, so knowing the type does not help there. "
               "The current model is better on the 3 Home cherry clusters (17.5% vs 21.1%).", SMALL))

    s.append(CondPageBreak(9 * cm))
    s.append(p("4. Weaknesses of the dataset", H2))
    s.append(p("The sessions differ in more than lighting and background: they also contain different tomatoes.", BASE))
    s.append(Spacer(1, 4))
    s.append(KeepTogether([image(f_four, 13), p("Three clusters of 4 normal tomatoes. Crops are scaled to the same height, "
                                                  "so sizes in the image are not comparable.", SMALL)]))
    s.append(Spacer(1, 4))
    s.append(table([
        ["Normal tomatoes per cluster", "Home", "Lab 1", "Lab 2"],
        ["2", "375-566 g", "183 g", "186-220 g"],
        ["3", "600-790 g", "280 g", "296-421 g"],
        ["4", "820-1010 g", "364 g", "368-560 g"],
        ["5", "1160 g", "430 g", "507-608 g"],
        ["6 to 9", "none", "none", "655-1029 g"],
    ], [5.2, 3.6, 3.6, 3.6]))
    s.append(Spacer(1, 4))
    s.extend(bullets([
        "Normal tomatoes at Home (188-283 g each) are larger than in both labs (86-140 g), with no overlap.",
        "Only Lab 2 has clusters of 6 to 9 tomatoes.",
        "Only Home photographs each cluster from clearly different distances: the farthest image of a cluster is typically "
        "2.4 times farther than the nearest, against 1.1 (Lab 1) and 1.2 times (Lab 2).",
        "Cherry clusters: 10 in total, none in Lab 2. Home provides 51% of all images.",
        "Within one session and tomato type, more tomatoes always means more mass. But because tomato type, tomato size and capture "
        "conditions all change together between sessions, this data cannot tell which of them causes the failure on new sessions.",
    ]))

    s.append(p("5. Proposed next experiment", H2))
    s.append(p("Train the model to predict <b>grams per cm<super>2</super> of visible area</b>, and compute "
               "<b>mass = real visible area x predicted grams per cm<super>2</super></b>. The size calculation is then done "
               "physically instead of being learned; the network only has to judge how heavy a cluster is for its visible area, "
               "which may let it tell cherry and normal tomatoes apart by itself.", BASE))
    s.append(Spacer(1, 4))
    s.append(table([
        ["", "Plan"],
        ["Test", "The same three new-session tests as above"],
        ["Success", "Error on new sessions clearly below the current model (30.9%, 249%, 37.8%), and close to the simple "
                    "real-area result with known type (19.5%, 21.0%, 22.4%)"],
        ["If no gain", "Report it as a result and stop experimenting"],
        ["Cost", "Coding, plus about 5 hours of training (three models)"],
    ], [3, 14]))
    s.append(Spacer(1, 6))
    s.append(p("Questions", H3))
    s.extend(bullets([
        "Do you agree with this experiment as the last one before writing (deadline January)?",
        "Would new captures be worth it, for example the same clusters photographed in both the home and the lab setup? That is the "
        "only way to separate tomato type from capture conditions.",
    ]))

    s.append(KeepTogether([p("6. Limits of these results", H2)] + bullets([
        "One training run per setting. Repeating the same training can change the error by several percentage points.",
        "Only three sessions, so conclusions about new sessions rest on very few examples.",
        "The tests in section 3.3 use images that never occur in real data.",
        "Tomato type and tomato counts were labelled by hand. The real-area result needs the tomato type.",
        "Full record of all tests, numbers and corrections: investigation/investigation_6_11.md in the repository.",
    ])))

    doc = SimpleDocTemplate(PDF, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Tomato cluster mass from RGB-D: progress summary", author="")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("->", PDF)


if __name__ == "__main__":
    build()
