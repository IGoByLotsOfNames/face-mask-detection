"""Render an offline report for the synthetic software demonstration.

The renderer does not run inference or load remote resources. Report text is
escaped, and embedded pictures accept only bounded PNG data URIs.
"""

from __future__ import annotations

import base64
import binascii
import html
import math

_LABELS = {
    "mask_weared_incorrect": "Mask worn incorrectly",
    "with_mask": "Mask worn",
    "without_mask": "No mask",
}

_STYLE = """
:root{color-scheme:light;--ink:#172a41;--muted:#52657b;--teal:#087d78;
--line:#dce5ed;--paper:#fff;--wash:#f2f6fa;--accent:#ccece7}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);
font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:#086f79;text-underline-offset:3px}a:hover{color:#124452}
a:focus-visible,select:focus-visible{outline:3px solid #b06c05;outline-offset:4px}
.skip{position:absolute;left:1rem;top:-5rem;background:#fff;padding:12px;z-index:9}
.skip:focus{top:1rem}.wrap{width:min(1180px,calc(100% - 64px));margin:auto}
header{background:#132b43;color:#fff;padding:52px 0 42px;border-bottom:5px solid #51bbae}
.eyebrow{font-size:.75rem;font-weight:750;letter-spacing:.16em;text-transform:uppercase;
color:#a8dbd6;margin:0 0 14px}h1{font-size:clamp(2rem,4vw,3.05rem);line-height:1.13;
letter-spacing:-.04em;max-width:940px;margin:0 0 20px}h2{font-size:1.4rem;
letter-spacing:-.025em;line-height:1.25;margin:0 0 10px}h3{font-size:1rem;margin:0 0 7px}
p{margin:0 0 12px}.lede{max-width:830px;color:#d3e1ee;font-size:1.02rem;margin:0}
main{padding:30px 0 46px}.notice{border:1px solid #d5b478;border-left:5px solid #9b661e;
border-radius:10px;background:#fff7e9;padding:18px 22px;margin-bottom:30px;color:#624411}
.notice strong{display:block;font-size:.76rem;text-transform:uppercase;letter-spacing:.1em;
margin-bottom:4px}.notice p{margin:0}.flow{list-style:none;padding:0;display:grid;
grid-template-columns:repeat(4,1fr);gap:14px;margin:0 0 32px}.flow li{border-top:3px solid
var(--teal);padding:12px 6px 0 0}.flow small{font-weight:750;color:var(--teal)}
.flow strong{display:block;margin:3px 0}.flow span{display:block;font-size:.85rem;color:var(--muted)}
section{margin:0 0 30px}.section-head{display:flex;justify-content:space-between;align-items:end;
gap:20px;margin-bottom:17px}.section-head p,.subtext{color:var(--muted);font-size:.9rem;margin:0}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.stat,.panel,.sample{
background:var(--paper);border:1px solid var(--line);border-radius:12px}
.stat{padding:20px 22px}.stat .value{font-size:2rem;font-weight:730;line-height:1.2;
font-variant-numeric:tabular-nums;letter-spacing:-.04em}.stat .label{color:var(--muted);
font-size:.85rem;margin-top:6px}.split-line{margin-top:14px;display:flex;gap:9px;flex-wrap:wrap}
.chip{display:inline-block;background:#e5eef4;color:#264c66;border:1px solid #d1dfe9;
border-radius:100px;font-size:.76rem;font-weight:650;padding:4px 10px}
.panel{padding:24px}.checks{list-style:none;padding:0;margin:0;display:grid;
grid-template-columns:repeat(2,1fr);gap:20px}.checks li{position:relative;padding-left:31px}
.check-icon{position:absolute;left:0;top:1px;background:var(--accent);color:#075b55;
width:22px;height:22px;text-align:center;border-radius:50%;font-size:.85rem;font-weight:750}
.checks strong{font-size:.9rem}.checks p{font-size:.84rem;color:var(--muted);margin:4px 0 0}
.results{display:grid;grid-template-columns:1fr 1fr;gap:18px}.scroll{overflow-x:auto;
max-width:100%;overscroll-behavior-x:contain}table{border-collapse:collapse;width:100%;
font-size:.8rem;font-variant-numeric:tabular-nums}caption{text-align:left;font-size:.85rem;
font-weight:700;margin:0 0 14px;color:var(--muted)}th,td{padding:11px 9px;
border-bottom:1px solid var(--line);text-align:right}th{font-weight:650}thead th{
color:var(--muted);font-size:.73rem;vertical-align:bottom}th:first-child{text-align:left}
.matrix td{background:#f2f7f9}.matrix td.diagonal{background:#d9eee8;color:#125b50;
font-weight:750}.matrix tbody th{min-width:114px}.table-note{font-size:.78rem;color:var(--muted);
margin:12px 0 0}.fixture-tag{font-size:.72rem;color:#79551d;background:#fcf0d9;
padding:4px 9px;border-radius:5px;display:inline-block;margin:0 0 14px;font-weight:650}
.filters{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin:0 0 18px}
.filters[hidden]{display:none}.filters label{font-size:.78rem;font-weight:650;display:block}
select{display:block;font:inherit;font-size:.85rem;margin-top:4px;border:1px solid #a5b8c9;
border-radius:7px;padding:9px 32px 9px 10px;background:#fff;color:var(--ink)}
.match-count{font-size:.82rem;color:var(--muted);padding-bottom:10px}
.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:17px}
.sample[hidden]{display:none}.sample{overflow:hidden}.thumb{height:156px;background:#e8eff4;
display:flex;align-items:center;justify-content:center;border-bottom:1px solid var(--line)}
.thumb img{max-height:136px;max-width:90%;image-rendering:auto;object-fit:contain}
.sample-body{padding:17px}.sample .filename{font: .73rem/1.5 ui-monospace,SFMono-Regular,
Consolas,monospace;color:var(--muted);overflow-wrap:anywhere;margin:7px 0 11px}
.sample dl{margin:0;font-size:.8rem;display:grid;grid-template-columns:1fr auto;gap:7px 12px}
.sample dt{color:var(--muted)}.sample dd{margin:0;text-align:right;font-weight:650}
.probabilities{font-size:.74rem;color:var(--muted);margin:13px 0 0;padding-top:10px;
border-top:1px solid var(--line)}.probabilities span{display:block;margin-top:2px}
footer{border-top:1px solid var(--line);padding:22px 0 0;color:var(--muted);font-size:.81rem}
.empty{padding:20px;color:var(--muted);background:white;border:1px solid var(--line);
border-radius:10px}.empty[hidden]{display:none}
@media(max-width:850px){.results{grid-template-columns:1fr}.gallery{grid-template-columns:
repeat(2,1fr)}.flow{grid-template-columns:repeat(2,1fr)}.checks{grid-template-columns:1fr}}
@media(max-width:560px){.wrap{width:calc(100% - 32px)}header{padding:35px 0 30px}
main{padding-top:20px}.stats{grid-template-columns:repeat(2,1fr);gap:10px}
.stat{padding:16px}.stat .value{font-size:1.7rem}.gallery{grid-template-columns:1fr}
.section-head{display:block}.section-head p{margin-top:8px}.panel{padding:17px}
.notice{padding:15px}th,td{padding:9px 6px}.filters{gap:10px}}
@media print{body{background:white;font-size:10pt}.wrap{width:100%}header{background:white;
color:#172a41;padding:0 0 15px;border-bottom:2px solid #172a41}.eyebrow,.lede{color:#52657b}
main{padding:18px 0}h1{font-size:26pt}.skip,.filters{display:none!important}
.sample[hidden]{display:block}.notice,.panel,.stat,.sample{break-inside:avoid}
.results{display:block}.results .panel{margin-bottom:18px}.gallery{grid-template-columns:
repeat(3,1fr)}.thumb{height:100px}.thumb img{max-height:88px}.flow{margin-bottom:20px}
section{margin-bottom:22px}a{color:inherit}.empty{display:none}.sample-body{padding:10px}}
"""

_SCRIPT = """
(() => {
  const filters = document.getElementById('gallery-filters');
  const split = document.getElementById('split-filter');
  const category = document.getElementById('class-filter');
  const cards = [...document.querySelectorAll('.sample')];
  const update = () => {
    let visible = 0;
    cards.forEach(card => {
      card.hidden = (split.value !== '' && card.dataset.split !== split.value) ||
                    (category.value !== '' && card.dataset.class !== category.value);
      if (!card.hidden) visible += 1;
    });
    document.getElementById('match-count').textContent =
      `${visible} of ${cards.length} illustrated crops`;
    document.getElementById('no-matches').hidden = visible !== 0;
  };
  split.addEventListener('change', update);
  category.addEventListener('change', update);
  filters.hidden = false;
  update();
})();
"""


def _text(value: object) -> str:
    return html.escape(str(value), quote=True)


def _label(value: object) -> str:
    return _text(_LABELS.get(str(value), str(value)))


def _integer(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Report counts must be non-negative integers")
    return f"{value:,}"


def _rate(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Report rates must be finite numbers between zero and one")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Report rates must be finite numbers between zero and one")
    return f"{value:.3f}"


def _image(value: object) -> str:
    prefix = "data:image/png;base64,"
    if not isinstance(value, str) or not value.startswith(prefix) or len(value) > 2_000_000:
        raise ValueError("Report images must be bounded PNG data URIs")
    try:
        decoded = base64.b64decode(value[len(prefix) :], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Report image has invalid base64 data") from error
    if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Report image is not PNG data")
    return value


def _sample(sample: dict, classes: list[str]) -> str:
    probabilities = sample["probabilities"]
    if len(probabilities) != len(classes):
        raise ValueError("Probability count does not match report classes")
    probability_rows = "".join(
        f"<span>{_label(name)}: {_rate(probability)}</span>"
        for name, probability in zip(classes, probabilities)
    )
    return (
        f'<article class="sample" data-split="{_text(sample["split"])}" '
        f'data-class="{_text(sample["class_name"])}">'
        f'<div class="thumb"><img src="{_image(sample["image_data_uri"])}" '
        f'alt="Synthetic illustration: {_label(sample["class_name"])}" loading="lazy"></div>'
        '<div class="sample-body">'
        f'<span class="chip">{_text(sample["split"])}</span>'
        f'<p class="filename">{_text(sample["source_image"])} · '
        f"object {_integer(sample['object_index'])}</p><dl>"
        f"<dt>Fixture label</dt><dd>{_label(sample['class_name'])}</dd>"
        f"<dt>Scripted output</dt><dd>{_label(sample['predicted_class'])}</dd></dl>"
        f'<p class="probabilities">Scripted probabilities{probability_rows}</p>'
        "</div></article>"
    )


def render_report(summary: dict) -> str:
    """Return a self-contained HTML report; never describe fixtures as inference."""
    if summary.get("schema_version") != 1 or summary.get("kind") != "synthetic-software-demo":
        raise ValueError("Unsupported demo summary")
    classes = summary["class_names"]
    if not isinstance(classes, list) or len(classes) != 3 or len(set(classes)) != 3:
        raise ValueError("The report requires three distinct class names")
    matrix = summary["confusion_matrix"]
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("The confusion matrix must be three by three")
    if len(summary["per_class"]) != 3:
        raise ValueError("The report requires three per-class metric rows")
    if summary["run_summary_relative"] != "summary.json":
        raise ValueError("The report summary link must be summary.json")
    count_labels = (
        ("source_images", "Source illustrations"),
        ("source_objects", "Annotated objects"),
        ("crops", "Retained face crops"),
        ("quarantined_source_images", "Quarantined source images"),
        ("excluded_objects", "Excluded objects"),
        ("duplicate_groups", "Duplicate image groups"),
    )
    stats = "".join(
        f'<div class="stat"><div class="value">{_integer(summary["counts"][key])}</div>'
        f'<div class="label">{label}</div></div>'
        for key, label in count_labels
    )
    split_counts = "".join(
        f'<span class="chip">{name.title()}: {_integer(summary["split_counts"][name])}</span>'
        for name in ("train", "validation", "test")
    )
    checks = []
    for check in summary["checks"]:
        if check["status"] != "passed":
            raise ValueError("A successful demo report must contain only passed checks")
        checks.append(
            '<li><span class="check-icon" aria-hidden="true">✓</span>'
            f"<strong>{_text(check['name'])} — passed</strong>"
            f"<p>{_text(check['detail'])}</p></li>"
        )
    matrix_head = "".join(f'<th scope="col">{_label(name)}</th>' for name in classes)
    matrix_body = "".join(
        f'<tr><th scope="row">{_label(classes[index])}</th>'
        + "".join(
            f'<td class="{"diagonal" if index == column else "off-diagonal"}">'
            f"{_integer(value)}</td>"
            for column, value in enumerate(row)
        )
        + "</tr>"
        for index, row in enumerate(matrix)
    )
    metrics = "".join(
        f'<tr><th scope="row">{_label(row["name"])}</th>'
        + "".join(f"<td>{_rate(row[key])}</td>" for key in ("precision", "recall", "f1"))
        + f"<td>{_integer(row['support'])}</td></tr>"
        for row in summary["per_class"]
    )
    class_options = "".join(
        f'<option value="{_text(name)}">{_label(name)}</option>' for name in classes
    )
    gallery = "".join(_sample(sample, classes) for sample in summary["samples"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>{_text(summary["title"])}</title><style>{_STYLE}</style></head><body>
<a class="skip" href="#main">Skip to report</a>
<header><div class="wrap"><p class="eyebrow">Face Mask Detection / Local demonstration</p>
<h1>{_text(summary["title"])}</h1>
<p class="lede">Follow illustrated inputs through annotation checks, face cropping,
grouped splitting and a reproducible report.</p></div></header>
<main class="wrap" id="main">
<aside class="notice" aria-label="Synthetic demonstration notice">
<strong>Software demonstration · synthetic data</strong><p>{_text(summary["notice"])}</p></aside>
<ol class="flow" aria-label="Demonstration sequence">
<li><small>01 / INPUT</small><strong>Illustrated fixtures</strong>
<span>Three documented mask categories</span></li>
<li><small>02 / CROP</small><strong>Validate and prepare</strong>
<span>Annotations, exclusions and duplicate groups</span></li>
<li><small>03 / SPLIT</small><strong>Keep groups together</strong>
<span>Train, validation and test partitions</span></li>
<li><small>04 / REPORT</small><strong>Inspect the evidence</strong>
<span>Checks, scripted outputs and source traces</span></li></ol>
<section aria-labelledby="inventory-title"><div class="section-head">
<div><h2 id="inventory-title">From source images to usable crops</h2>
<p>Counts from this demonstration run</p></div></div><div class="stats">{stats}</div>
<div class="split-line" aria-label="Retained crop counts by split">{split_counts}</div></section>
<section aria-labelledby="checks-title"><div class="section-head"><div>
<h2 id="checks-title">Pipeline checks</h2><p>Assertions exercised by this run</p></div></div>
<div class="panel"><ul class="checks">{"".join(checks)}</ul></div></section>
<section aria-labelledby="outputs-title"><div class="section-head"><div>
<h2 id="outputs-title">Scripted fixture output</h2>
<p>These numbers verify the reporting path. They do not measure a trained model.</p>
</div></div><div class="results">
<div class="panel"><span class="fixture-tag">Software fixture only</span>
<h3>Confusion matrix</h3><div class="scroll"><table class="matrix">
<caption>Rows: fixture label · columns: scripted output</caption><thead><tr>
<th scope="col">Fixture label</th>{matrix_head}</tr></thead><tbody>{matrix_body}</tbody>
</table></div><p class="table-note">Diagonal cells show matching labels;
off-diagonal cells show scripted mismatches. Values are crop counts.</p></div>
<div class="panel"><span class="fixture-tag">Software fixture only</span>
<h3>Per-class metrics</h3><div class="scroll"><table>
<caption>Computed from the scripted fixture predictions</caption><thead><tr>
<th scope="col">Category</th><th scope="col">Precision</th><th scope="col">Recall</th>
<th scope="col">F1</th><th scope="col">Support</th></tr></thead><tbody>{metrics}</tbody>
</table></div><p class="table-note">Rates use a 0–1 scale. Support is the number of
fixture crops for that class. No real-world performance is established.</p></div></div></section>
<section aria-labelledby="gallery-title"><div class="section-head"><div>
<h2 id="gallery-title">Inspect the retained crops</h2><p>Embedded illustrations with their
source, split and scripted output</p></div></div>
<div class="filters" id="gallery-filters" hidden><label for="split-filter">Partition
<select id="split-filter"><option value="">All partitions</option>
<option value="train">Train</option><option value="validation">Validation</option>
<option value="test">Test</option></select></label><label for="class-filter">Fixture category
<select id="class-filter"><option value="">All categories</option>{class_options}</select></label>
<span id="match-count" class="match-count" role="status" aria-live="polite"></span></div>
<div class="gallery">{gallery}</div><p id="no-matches" class="empty" hidden>
No illustrated crops match these filters.</p></section>
<footer><p><a href="summary.json">Open the machine-readable run summary</a> ·
This report embeds its styles, script and illustrations; it uses no external assets.</p>
<p>Research and software demonstration. Synthetic illustrations and scripted predictions
do not establish mask detection accuracy or suitability for deployment.</p></footer>
</main><script>{_SCRIPT}</script></body></html>
"""
