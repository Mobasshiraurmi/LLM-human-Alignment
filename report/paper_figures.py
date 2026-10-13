"""paper_figures.py — figures that follow the ACL camera-ready rules.

  * Drawn at their printed size, in points: one column = 219 pt (\\columnwidth),
    two columns = 455 pt (\\textwidth). Text is 7-8 pt at that size, so it stays legible
    on A4 paper and when zoomed.
  * Exported as vector PDF (for \\includegraphics) and SVG, next to the HTML report.
  * Captions go BELOW the figure as "Figure N: ...", 10 pt roman; centred when they fit
    on one line, left-aligned when longer (CSS below; acl.sty does the same in LaTeX).
  * Nothing relies on colour alone: the figure code pairs every colour with a fill
    pattern, marker shape or line style that survives grayscale printing.

PDF export prints a one-page HTML wrapper with headless Google Chrome (vector output,
fonts embedded). Without Chrome the SVG is still written and the PDF is skipped with a note.
"""
from __future__ import annotations

import html
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

COLUMN_PT, TEXT_PT = 219, 455          # ACL \columnwidth and \textwidth, in points

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "chromium", "chromium-browser",
]


def _chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if os.path.isabs(c) and os.path.exists(c):
            return c
        if not os.path.isabs(c) and shutil.which(c):
            return shutil.which(c)
    return None


def svg_root(inner: str, w: float, h: float, sid: str, label: str, screen_scale: float = 2.2) -> str:
    """Root element. The viewBox is in points; on screen the figure scales to the page width
    (up to screen_scale x its printed size); export() pins it to its printed size."""
    return (f'<svg id="{sid}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" width="100%" '
            f'style="max-width:{w * screen_scale:.0f}px" font-family="Helvetica, Arial, sans-serif" '
            f'role="img" aria-label="{html.escape(label, quote=True)}">{inner}</svg>')


def _print_size(svg: str) -> tuple[str, float, float]:
    import re
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    w, h = float(m.group(1)), float(m.group(2))
    svg = re.sub(r'width="100%"\s+style="max-width:[^"]*"', f'width="{w:g}pt" height="{h:g}pt"', svg, count=1)
    return svg, w, h


def _pdf(chrome: str, svg: str, w: float, h: float, pdf_path: str) -> bool:
    page = (f"<!doctype html><html><head><meta charset='utf-8'><style>@page {{ size: {w:g}pt {h:g}pt; margin: 0 }}"
            f" html, body {{ margin: 0; padding: 0 }} svg {{ display: block }}</style></head><body>{svg}</body></html>")
    import time
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    # Chrome's helper processes can still be writing to the profile when it is stopped,
    # so a leftover file must not abort the build
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "fig.html")
        open(src, "w").write(page)
        proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                                 "--no-first-run", "--no-default-browser-check",
                                 f"--user-data-dir={os.path.join(tmp, 'profile')}",
                                 f"--print-to-pdf={pdf_path}", "file://" + src],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Chrome writes the PDF within a second or two but may linger afterwards: wait for a
        # complete file (ends with %%EOF), then stop it.
        deadline, done = time.time() + 90, False
        while time.time() < deadline:
            if os.path.exists(pdf_path) and open(pdf_path, "rb").read()[-1024:].rstrip().endswith(b"%%EOF"):
                done = True
                break
            if proc.poll() is not None and not os.path.exists(pdf_path):
                break
            time.sleep(0.2)
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return done


def export(figs: dict[str, str], out_dir: str, pdf: bool = True) -> list[str]:
    """figs: {file stem: svg from svg_root}. Writes <stem>.svg and <stem>.pdf at printed size.
    Returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    chrome = _chrome() if pdf else None
    if pdf and chrome is None:
        print("  ! Google Chrome not found: SVG figures written, PDF export skipped")
    written, jobs = [], []
    for stem, svg in figs.items():
        psvg, w, h = _print_size(svg)
        path = os.path.join(out_dir, stem + ".svg")
        with open(path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + psvg)
        written.append(path)
        if chrome:
            jobs.append((psvg, w, h, os.path.join(out_dir, stem + ".pdf")))
    if jobs:
        with ThreadPoolExecutor(max_workers=6) as pool:
            ok = list(pool.map(lambda j: _pdf(chrome, *j), jobs))
        written += [j[3] for j, k in zip(jobs, ok) if k]
        if not all(ok):
            print(f"  ! PDF export failed for {ok.count(False)} figure(s)")
    return written


def figure_html(svg: str, number: int, caption: str, extra: str = "") -> str:
    """caption: HTML, without the 'Figure N:' prefix."""
    return (f"<figure class='paper'>{svg}{extra}"
            f"<figcaption>Figure {number}: {caption}</figcaption></figure>")


def tex_figure(pdf_name: str, caption: str, wide: bool, label: str) -> str:
    env = "figure*" if wide else "figure"
    width = r"\textwidth" if wide else r"\columnwidth"
    return "\n".join([rf"\begin{{{env}}}[t]", r"\centering",
                      rf"\includegraphics[width={width}]{{{pdf_name}}}",
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\end{{{env}}}"])


def tex_text(s: str) -> str:
    """Caption text (plain, may contain the report's unicode) -> LaTeX."""
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("Δ", r"$\Delta$").replace("−", "$-$").replace("–", "--")
            .replace("—", "---").replace("≥", r"$\geq$").replace("≤", r"$\leq$").replace("×", r"$\times$")
            .replace("ln 2", r"$\ln 2$"))


# Download buttons on the HTML pages: save a chart (by SVG id) as PNG or SVG.
DOWNLOAD_SCRIPT = """function downloadChart(id, name, fmt) {
  const svg = document.getElementById(id);
  const src = new XMLSerializer().serializeToString(svg);
  const save = (href, ext) => {
    const a = document.createElement('a');
    a.href = href; a.download = name + '.' + ext;
    document.body.appendChild(a); a.click(); a.remove();
  };
  const svgUrl = URL.createObjectURL(new Blob([src], {type: 'image/svg+xml;charset=utf-8'}));
  if (fmt === 'svg') { save(svgUrl, 'svg'); setTimeout(() => URL.revokeObjectURL(svgUrl), 1000); return; }
  const vb = svg.viewBox.baseVal, scale = 3;
  const img = new Image();
  img.onload = () => {
    const c = document.createElement('canvas');
    c.width = vb.width * scale; c.height = vb.height * scale;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(img, 0, 0, c.width, c.height);
    URL.revokeObjectURL(svgUrl);
    c.toBlob(b => { const u = URL.createObjectURL(b); save(u, 'png'); setTimeout(() => URL.revokeObjectURL(u), 1000); }, 'image/png');
  };
  img.src = svgUrl;
}
"""


# Caption and figure styling shared by the reports (ACL: 10 pt roman, below the figure,
# centred when one line, left-aligned when longer -- `width: fit-content` gives exactly that).
CSS = """
figure.paper { margin: 26px 0 30px; max-width: 1000px; }
figure.paper svg { display: block; margin: 0 auto; }
figure.paper figcaption, .acl-cap { font-family: 'Times New Roman', Times, serif; font-size: 10pt; line-height: 1.35;
  color: #000; width: fit-content; max-width: 100%; margin: 10px auto 0; text-align: left; }
"""
