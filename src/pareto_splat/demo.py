"""Static demo generation for Pareto-Splat experiment summaries."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SUMMARY_PATTERN = "results/**/summary/summary.json"
DEMO_SCHEMA_VERSION = 1


class DemoError(ValueError):
    """Raised when demo inputs are invalid."""


def discover_summary_paths(
    project_root: Path,
    *,
    pattern: str = SUMMARY_PATTERN,
) -> tuple[Path, ...]:
    """Return generated experiment summary JSON paths in stable order."""

    root = project_root.resolve()
    return tuple(sorted(path for path in root.glob(pattern) if path.is_file()))


def read_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    """Read one summary JSON file and return copied row dictionaries."""

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DemoError(f"invalid JSON summary: {summary_path}") from error
    if not isinstance(payload, list):
        raise DemoError(f"summary must contain a list of rows: {summary_path}")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise DemoError(
                f"summary row {index} must be an object: {summary_path}"
            )
        rows.append(dict(row))
    return rows


def build_demo_payload(
    project_root: Path,
    *,
    summary_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Build the browser payload for all available experiment summaries."""

    root = project_root.resolve()
    paths = (
        tuple(summary_paths)
        if summary_paths is not None
        else discover_summary_paths(root)
    )
    studies: list[dict[str, Any]] = []
    for summary_path in sorted(paths):
        rows = read_summary_rows(summary_path)
        if not rows:
            continue
        metadata = _study_metadata(summary_path, root, rows)
        normalized_rows = [
            _normalize_row(row, index)
            for index, row in enumerate(rows)
        ]
        studies.append(
            {
                **metadata,
                "row_count": len(normalized_rows),
                "pareto_row_count": sum(
                    1
                    for row in normalized_rows
                    if _is_number(row.get("pareto_rank"))
                    and int(row["pareto_rank"]) == 0
                ),
                "rows": normalized_rows,
            }
        )

    return {
        "schema_version": DEMO_SCHEMA_VERSION,
        "objective": {
            "description": "maximize PSNR and FPS, minimize serialized model size",
            "vector": ["psnr", "fps", "-serialized_mib"],
        },
        "studies": studies,
    }


def write_demo_html(payload: Mapping[str, Any], output_path: Path) -> Path:
    """Write a self-contained static HTML demo."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_demo_html(payload), encoding="utf-8")
    return output_path


def build_demo_site(
    project_root: Path,
    output_path: Path,
    *,
    summary_paths: Sequence[Path] | None = None,
) -> Path:
    """Build and write the complete static demo site."""

    payload = build_demo_payload(project_root, summary_paths=summary_paths)
    return write_demo_html(payload, output_path)


def build_demo_html(payload: Mapping[str, Any]) -> str:
    """Return the complete static HTML document for a demo payload."""

    payload_json = json.dumps(payload, indent=2, sort_keys=True)
    safe_payload_json = payload_json.replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pareto-Splat Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #162033;
      --muted: #667085;
      --line: #d8dee9;
      --accent: #2558d8;
      --accent-soft: #e8efff;
      --danger: #c7352f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 2rem clamp(1rem, 5vw, 4rem);
      background: linear-gradient(135deg, #111827, #2546a0);
      color: white;
    }}
    header p {{
      max-width: 70rem;
      color: #dbe6ff;
      margin-bottom: 0;
    }}
    main {{
      padding: 1.5rem clamp(1rem, 5vw, 4rem) 3rem;
      display: grid;
      gap: 1rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
      padding: 1rem;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
      gap: 1rem;
    }}
    label {{
      display: grid;
      gap: 0.35rem;
      font-weight: 650;
    }}
    select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      color: var(--ink);
      padding: 0.7rem 0.8rem;
      font: inherit;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
      gap: 0.8rem;
    }}
    .card {{
      background: var(--accent-soft);
      border-radius: 12px;
      padding: 0.85rem;
    }}
    .card span {{
      display: block;
      color: var(--muted);
      font-size: 0.82rem;
    }}
    .card strong {{
      display: block;
      margin-top: 0.25rem;
      font-size: 1.2rem;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(18rem, 1fr) minmax(20rem, 1fr);
      gap: 1rem;
    }}
    @media (max-width: 850px) {{
      .layout {{ grid-template-columns: 1fr; }}
    }}
    svg {{
      width: 100%;
      height: auto;
      min-height: 24rem;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fbfcff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 0.55rem 0.5rem;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    tr {{
      cursor: pointer;
    }}
    tr:hover, tr.selected {{
      background: var(--accent-soft);
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 0.1rem 0.5rem;
      background: #edf2f7;
      color: var(--muted);
      font-weight: 700;
      font-size: 0.78rem;
    }}
    .badge.front {{
      background: #e6f4ea;
      color: #137333;
    }}
    .muted {{
      color: var(--muted);
    }}
    .empty {{
      border: 1px dashed var(--line);
      border-radius: 12px;
      padding: 1rem;
      color: var(--muted);
    }}
    .selected-point {{
      color: var(--danger);
      font-weight: 750;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Pareto-Splat Demo</h1>
    <p>
      Select an experiment summary, inspect operating points, and compare the
      default Pareto objectives: maximize PSNR, maximize FPS, and minimize
      serialized model size. Objective vector: f(x) = [PSNR, FPS, -Size].
    </p>
  </header>
  <main>
    <section class="panel controls">
      <label>
        Study
        <select id="study-select"></select>
      </label>
      <label>
        Operating point
        <select id="variant-select"></select>
      </label>
    </section>
    <section class="panel">
      <h2 id="study-title">Study</h2>
      <p id="study-meta" class="muted"></p>
      <div id="metric-cards" class="cards"></div>
    </section>
    <section class="layout">
      <div class="panel">
        <h2>PSNR vs FPS</h2>
        <p class="muted">
          Rank-0 points are outlined. The selected operating point is shown in
          red. Robustness-only studies may not include FPS.
        </p>
        <div id="plot"></div>
      </div>
      <div class="panel">
        <h2>Operating Points</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Variant</th>
                <th>Strategy</th>
                <th>Rank</th>
                <th>PSNR</th>
                <th>FPS</th>
                <th>Size</th>
                <th>Gaussians</th>
                <th>PSNR drop</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
    </section>
  </main>
  <script type="application/json" id="demo-data">{safe_payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("demo-data").textContent);
    const studySelect = document.getElementById("study-select");
    const variantSelect = document.getElementById("variant-select");
    const rowsBody = document.getElementById("rows");
    const cards = document.getElementById("metric-cards");
    const plot = document.getElementById("plot");
    const title = document.getElementById("study-title");
    const meta = document.getElementById("study-meta");

    let selectedStudyIndex = 0;
    let selectedRowIndex = 0;

    function isNumber(value) {{
      return value !== null && value !== undefined && Number.isFinite(Number(value));
    }}

    function fmt(value, digits = 3) {{
      if (!isNumber(value)) return "n/a";
      return Number(value).toLocaleString(undefined, {{
        maximumFractionDigits: digits
      }});
    }}

    function pct(value) {{
      if (!isNumber(value)) return "n/a";
      return `${{fmt(Number(value) * 100, 1)}}%`;
    }}

    function pointLabel(row) {{
      return row.variant_id || row.id || "row";
    }}

    function rankBadge(row) {{
      if (!isNumber(row.pareto_rank)) return '<span class="badge">n/a</span>';
      const rank = Number(row.pareto_rank);
      const className = rank === 0 ? "badge front" : "badge";
      return `<span class="${{className}}">rank ${{rank}}</span>`;
    }}

    function optionLabel(study) {{
      return `${{study.scene}} / ${{study.study_type}} / ${{study.run}}`;
    }}

    function init() {{
      if (!payload.studies || payload.studies.length === 0) {{
        document.querySelector("main").innerHTML =
          '<section class="panel empty">No summary files were found. Run a study summary stage, then rebuild the demo.</section>';
        return;
      }}
      payload.studies.forEach((study, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = optionLabel(study);
        studySelect.appendChild(option);
      }});
      studySelect.addEventListener("change", () => {{
        selectedStudyIndex = Number(studySelect.value);
        selectedRowIndex = 0;
        render();
      }});
      variantSelect.addEventListener("change", () => {{
        selectedRowIndex = Number(variantSelect.value);
        render();
      }});
      render();
    }}

    function render() {{
      const study = payload.studies[selectedStudyIndex];
      const rows = study.rows || [];
      const selected = rows[selectedRowIndex] || rows[0] || {{}};
      title.textContent = optionLabel(study);
      meta.textContent = `${{study.row_count}} operating points from ${{study.summary_path}}. ` +
        `${{study.pareto_row_count}} rank-0 points under f(x) = [PSNR, FPS, -Size].`;
      renderVariantOptions(rows);
      renderCards(selected);
      renderRows(rows);
      renderPlot(study, selectedRowIndex);
    }}

    function renderVariantOptions(rows) {{
      variantSelect.innerHTML = "";
      rows.forEach((row, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = pointLabel(row);
        if (index === selectedRowIndex) option.selected = true;
        variantSelect.appendChild(option);
      }});
    }}

    function renderCards(row) {{
      const specs = [
        ["PSNR", fmt(row.psnr), "dB"],
        ["FPS", fmt(row.fps, 2), ""],
        ["Model size", fmt(row.serialized_mib, 2), "MiB"],
        ["Gaussians", fmt(row.gaussian_count, 0), ""],
        ["Keep fraction", pct(row.keep_fraction), ""],
        ["PSNR drop", fmt(row.psnr_drop, 3), "dB"]
      ];
      cards.innerHTML = specs.map(([label, value, unit]) => `
        <div class="card">
          <span>${{label}}</span>
          <strong>${{value}} ${{unit}}</strong>
        </div>
      `).join("");
    }}

    function renderRows(rows) {{
      rowsBody.innerHTML = "";
      rows.forEach((row, index) => {{
        const tr = document.createElement("tr");
        if (index === selectedRowIndex) tr.classList.add("selected");
        tr.innerHTML = `
          <td>${{pointLabel(row)}}</td>
          <td>${{row.strategy || row.degradation || "n/a"}}</td>
          <td>${{rankBadge(row)}}</td>
          <td>${{fmt(row.psnr)}}</td>
          <td>${{fmt(row.fps, 2)}}</td>
          <td>${{fmt(row.serialized_mib, 2)}}</td>
          <td>${{fmt(row.gaussian_count, 0)}}</td>
          <td>${{fmt(row.psnr_drop)}}</td>
        `;
        tr.addEventListener("click", () => {{
          selectedRowIndex = index;
          render();
        }});
        rowsBody.appendChild(tr);
      }});
    }}

    function renderPlot(study, selectedIndex) {{
      const rows = (study.rows || []).filter(row => isNumber(row.psnr) && isNumber(row.fps));
      if (rows.length < 2) {{
        plot.innerHTML = '<div class="empty">This study does not include enough FPS data for a PSNR-vs-FPS scatter plot.</div>';
        return;
      }}
      const width = 680;
      const height = 430;
      const margin = {{left: 58, right: 24, top: 24, bottom: 54}};
      const fpsValues = rows.map(row => Number(row.fps));
      const psnrValues = rows.map(row => Number(row.psnr));
      const minFps = Math.min(...fpsValues);
      const maxFps = Math.max(...fpsValues);
      const minPsnr = Math.min(...psnrValues);
      const maxPsnr = Math.max(...psnrValues);
      const x = value => margin.left + ((value - minFps) / Math.max(maxFps - minFps, 1e-9)) * (width - margin.left - margin.right);
      const y = value => height - margin.bottom - ((value - minPsnr) / Math.max(maxPsnr - minPsnr, 1e-9)) * (height - margin.top - margin.bottom);

      const circles = (study.rows || []).map((row, originalIndex) => {{
        if (!isNumber(row.psnr) || !isNumber(row.fps)) return "";
        const cx = x(Number(row.fps));
        const cy = y(Number(row.psnr));
        const rankZero = Number(row.pareto_rank) === 0;
        const selected = originalIndex === selectedIndex;
        const fill = selected ? "#c7352f" : "#2558d8";
        const stroke = selected ? "#c7352f" : (rankZero ? "#111827" : "white");
        const radius = selected ? 7 : 5;
        return `
          <g>
            <circle cx="${{cx}}" cy="${{cy}}" r="${{radius}}" fill="${{fill}}" stroke="${{stroke}}" stroke-width="${{rankZero ? 3 : 1.5}}"></circle>
            <title>${{pointLabel(row)}}: PSNR ${{fmt(row.psnr)}}, FPS ${{fmt(row.fps, 2)}}</title>
          </g>
        `;
      }}).join("");

      plot.innerHTML = `
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="PSNR versus FPS scatter plot">
          <line x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}" stroke="#98a2b3"></line>
          <line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}" stroke="#98a2b3"></line>
          <text x="${{width / 2}}" y="${{height - 14}}" text-anchor="middle" fill="#667085">FPS</text>
          <text transform="translate(18 ${{height / 2}}) rotate(-90)" text-anchor="middle" fill="#667085">PSNR</text>
          <text x="${{margin.left}}" y="${{height - margin.bottom + 24}}" text-anchor="middle" fill="#667085">${{fmt(minFps, 1)}}</text>
          <text x="${{width - margin.right}}" y="${{height - margin.bottom + 24}}" text-anchor="middle" fill="#667085">${{fmt(maxFps, 1)}}</text>
          <text x="${{margin.left - 10}}" y="${{y(minPsnr) + 4}}" text-anchor="end" fill="#667085">${{fmt(minPsnr, 1)}}</text>
          <text x="${{margin.left - 10}}" y="${{y(maxPsnr) + 4}}" text-anchor="end" fill="#667085">${{fmt(maxPsnr, 1)}}</text>
          ${{circles}}
        </svg>
      `;
    }}

    init();
  </script>
</body>
</html>
"""


def _study_metadata(
    summary_path: Path,
    project_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    relative_path = _relative_path(summary_path, project_root)
    parts = Path(relative_path).parts
    scene = _first_text(rows, "scene") or "unknown"

    if len(parts) >= 5 and parts[-2:] == ("summary", "summary.json"):
        study_type = parts[-5]
        scene = _first_text(rows, "scene") or parts[-4]
        run = parts[-3]
        study_id = "/".join(parts[-5:-2])
    else:
        study_type = "unknown"
        run = summary_path.parent.parent.name or "summary"
        study_id = relative_path

    return {
        "study_id": study_id,
        "study_type": study_type,
        "scene": scene,
        "run": run,
        "summary_path": relative_path,
    }


def _normalize_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    normalized = {
        key: _json_scalar(value)
        for key, value in row.items()
        if _json_scalar(value) is not None
    }
    normalized.setdefault("variant_id", f"row_{index:03d}")
    normalized.setdefault("strategy", normalized.get("degradation", "unknown"))
    normalized["is_pareto_rank_zero"] = (
        _is_number(normalized.get("pareto_rank"))
        and int(normalized["pareto_rank"]) == 0
    )
    return normalized


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        return value
    return str(value)


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _first_text(rows: Sequence[Mapping[str, Any]], key: str) -> str | None:
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
