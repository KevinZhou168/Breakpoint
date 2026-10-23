"""Plain-language condition labels and discrete object-status plots."""
import plotly.graph_objects as go

from .core import physical_value


CONDITION_HELP = {
    "blur": "Blur spread is measured in pixels (Gaussian sigma). Larger values blend a wider neighborhood of pixels; this is not a hard blur radius.",
    "darkness": "Percentage of the original stored pixel values remaining. Smaller values mean a darker image. This is not a measurement of real-world light.",
    "noise": "Spread of the random changes added independently to each color channel, on the image's 0–255 scale (Gaussian standard deviation). Larger values mean stronger noise.",
}


def condition_display(kind, severity, shape):
    value, _ = physical_value(kind, severity, shape)
    if kind == "blur":
        return f"{value:.2f} px", "Blur spread"
    if kind == "darkness":
        return f"{100 * value:.2f}%", "Original pixel values remaining"
    return f"{value:.1f} levels", "Noise spread"


def boundary_display(kind, bracket, shape, has_reference):
    _, label = condition_display(kind, 0, shape)
    if not has_reference:
        return {"value": "Undefined", "label": label, "detail": "No original detections at this confidence setting. Lower it to measure a failure point."}
    if bracket is None:
        maximum, _ = condition_display(kind, 1, shape)
        return {"value": "Not observed", "label": label,
                "detail": f"No failure found through severity 1.00 ({maximum}). This only covers the tested range."}
    lower, upper = bracket
    lo, _ = condition_display(kind, lower, shape)
    hi, _ = condition_display(kind, upper, shape)
    return {"value": f"{lo} → {hi}", "label": label,
            "detail": f"Severity {lower:.2f}: still met the cutoff. Severity {upper:.2f}: first fell below it. The exact failure point between these tests is unknown."}


def status_matrix(rows, levels, labels, kind, shape):
    """Square cells with explicit states; no continuous colors or interpolation."""
    trials = max(sum(r["severity"] == level for r in rows) for level in levels)
    noise = kind == "noise"
    categories = (["Original control"] + [f"{n}/{trials} trials" for n in range(trials + 1)]
                  if noise else ["Retained", "Lost"])
    cells = {category: [] for category in categories}
    for col, level in enumerate(levels):
        subset = [r for r in rows if r["severity"] == level]
        counts = [0] * len(labels)
        for row in subset:
            for a, _, _ in row["pairs"]:
                counts[a] += 1
        amount, unit = condition_display(kind, level, shape)
        for obj, label in enumerate(labels):
            kept = counts[obj]
            control = noise and level == 0
            if control:
                category, symbol, status = "Original control", "✓", "Original result reused; not three separate trials"
            elif noise:
                category = f"{kept}/{len(subset)} trials"
                symbol, status = str(kept), f"Retained in {kept}/{len(subset)} trials"
            else:
                category = "Retained" if kept else "Lost"
                symbol, status = ("✓" if kept else "×"), category
            cells[category].append((col, obj, symbol, [label, f"{level:.2f}", f"{unit}: {amount}", status]))
    palette = ["#FF775E", "#FFB56B", "#EAD079", "#72B5FF", "#80CF99", "#B7F36B"]
    fig = go.Figure()
    for category in categories:
        points = cells[category]
        if category in ("Retained", "Original control"):
            color = "#B7F36B" if category == "Retained" else "#72B5FF"
        elif category == "Lost":
            color = "#FF775E"
        else:
            kept = int(category.split("/")[0])
            color = palette[round(kept / trials * (len(palette) - 1))]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in points] or [None], y=[p[1] for p in points] or [None],
            mode="markers+text", name=category, text=[p[2] for p in points],
            customdata=[p[3] for p in points],
            marker=dict(symbol="square", size=23, color=color),
            textfont=dict(color="#10171D", size=13),
            hovertemplate="%{customdata[0]}<br>Severity %{customdata[1]}<br>%{customdata[2]}<br>%{customdata[3]}<extra></extra>",
        ))
    fig.update_layout(
        height=max(340, 28 * len(labels) + 170), margin=dict(l=5, r=10, t=110, b=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.10, x=0),
        xaxis=dict(title="Severity →", side="top", tickmode="array", tickvals=list(range(len(levels))),
                   ticktext=[f"{level:.2f}" for level in levels], range=[-0.6, len(levels) - 0.4],
                   showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels,
                   range=[len(labels) - 0.4, -0.6], showgrid=False, zeroline=False, fixedrange=True),
    )
    return fig
