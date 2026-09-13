(function () {
  "use strict";
  var D = window.TALLY;
  var NS = "http://www.w3.org/2000/svg";

  // ------------------------------------------------------------------ helpers
  function svg(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function html(tag, attrs, parent, text) {
    var n = document.createElement(tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function f3(v) { return v.toFixed(3); }
  function signed(v) { return (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(3); }
  function intc(v) { return Math.round(v).toLocaleString("en-US"); }
  function billions(v) { return (v / 1e9).toFixed(2) + "B"; }
  function millions(v) { return (v / 1e6).toFixed(1) + "M"; }

  // ------------------------------------------------------------------ tooltip
  var tip = document.getElementById("tip");
  function fillTip(lines) {
    clear(tip);
    lines.forEach(function (l, i) {
      html(i === 0 ? "b" : "div", l.muted ? { class: "m" } : {}, tip, l.text);
    });
    tip.hidden = false;
  }
  function place(x, y) {
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.min(x + 14, window.innerWidth - w - 8);
    var top = y + 14 + h > window.innerHeight ? y - h - 12 : y + 14;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }
  function bindTip(node, lines, label) {
    node.setAttribute("tabindex", "0");
    node.setAttribute("role", "img");
    node.setAttribute("aria-label", label);
    node.addEventListener("pointermove", function (e) { fillTip(lines); place(e.clientX, e.clientY); });
    node.addEventListener("pointerleave", function () { tip.hidden = true; });
    node.addEventListener("focus", function () {
      var r = node.getBoundingClientRect();
      fillTip(lines); place(r.left + r.width / 2, r.top + r.height / 2);
    });
    node.addEventListener("blur", function () { tip.hidden = true; });
  }
  function icon(kind) {
    var s = svg("svg", { viewBox: "0 0 16 16", "aria-hidden": "true" });
    if (kind === "ok") {
      svg("circle", { cx: 8, cy: 8, r: 7, fill: "none", stroke: "currentColor", "stroke-width": 1.5 }, s);
      svg("path", { d: "M4.8 8.2l2.1 2.1 4.3-4.6", fill: "none", stroke: "currentColor", "stroke-width": 1.6, "stroke-linecap": "round", "stroke-linejoin": "round" }, s);
    } else {
      svg("path", { d: "M8 1.8l6.6 11.6H1.4z", fill: "none", stroke: "currentColor", "stroke-width": 1.4, "stroke-linejoin": "round" }, s);
      svg("path", { d: "M8 6.2v3.4M8 11.4v.2", fill: "none", stroke: "currentColor", "stroke-width": 1.6, "stroke-linecap": "round" }, s);
    }
    return s;
  }

  // ------------------------------------------------------------------ planner
  var state = { bench: "terminalbench", mode: "both", certainty: 90, attempts: "3" };
  var certainty = document.getElementById("certainty");
  var certaintyOut = document.getElementById("certainty-out");

  function cell() {
    var t = ((100 - state.certainty) / 100).toFixed(2);
    return D.planner[state.bench].grid[state.mode + "|" + t + "|" + state.attempts];
  }

  function renderTiles(c, P) {
    var names = Object.keys(P.full);
    var topFull = names.reduce(function (a, b) { return P.full[a] >= P.full[b] ? a : b; });
    document.getElementById("p-cost").textContent = (c.cost * 100).toFixed(1) + "%";
    document.getElementById("p-meter").style.width = Math.min(100, c.cost * 100).toFixed(1) + "%";
    document.getElementById("p-tok-plan").textContent = billions(c.cost * P.tokens_full) + " tokens";
    document.getElementById("p-tok-full").textContent = billions(P.tokens_full) + " full run";
    document.getElementById("p-sp").textContent = f3(c.spearman);
    document.getElementById("p-mae").textContent = f3(c.mae);
    document.getElementById("p-top").textContent = c.top1[0] + " of " + c.top1[1];
    document.getElementById("p-top-s").textContent = (c.top1[1] === 1 ? "run keeps " : "samples keep ") + topFull + " first";
    document.getElementById("p-run").textContent = Math.round(c.tasks_run * 100) + "%";
    document.getElementById("p-run-s").textContent = "of " + P.tasks + " shared tasks";

    var st = document.getElementById("p-status");
    clear(st);
    var intact = c.spearman >= 0.95 && c.top1[0] === c.top1[1];
    var pill = html("span", { class: "pill " + (intact ? "good" : "warn") }, st);
    pill.appendChild(icon(intact ? "ok" : "warn"));
    html("span", {}, pill, intact ? "Ranking intact" : c.spearman >= 0.85 ? "Ranking mostly holds" : "Ranking shifts");
    document.getElementById("p-tie").hidden = state.bench !== "swebenchpro";
  }

  function renderDumbbell(c, P) {
    var box = document.getElementById("dumbbell");
    clear(box);
    var W = Math.max(300, box.clientWidth || 600);
    var names = Object.keys(P.full).sort(function (a, b) { return P.full[b] - P.full[a]; });
    var rankFull = names.slice();
    var rankEst = names.slice().sort(function (a, b) { return c.est[b] - c.est[a]; });
    var left = 76, right = 70, top = 22, rowH = 38, bottom = 28;
    var H = top + names.length * rowH + bottom;
    var vals = names.map(function (n) { return P.full[n]; }).concat(names.map(function (n) { return c.est[n]; }));
    var lo = Math.max(0, Math.floor((Math.min.apply(null, vals) - 0.03) * 10) / 10);
    var hi = Math.min(1, Math.ceil((Math.max.apply(null, vals) + 0.03) * 10) / 10);
    if (hi - lo < 0.3) { hi = Math.min(1, lo + 0.3); }
    var x = function (v) { return left + (v - lo) / (hi - lo) * (W - left - right); };
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, box);

    var step = hi - lo > 0.6 ? 0.2 : 0.1;
    for (var tv = lo; tv <= hi + 1e-9; tv += step) {
      var tx = x(tv);
      svg("line", { x1: tx, x2: tx, y1: top - 6, y2: H - bottom, stroke: "var(--grid)", "stroke-width": 1 }, s);
      svg("text", { x: tx, y: H - 8, "text-anchor": "middle", "font-size": 11, fill: "var(--muted)" }, s).textContent = tv.toFixed(1);
    }
    svg("text", { x: W - 2, y: 12, "text-anchor": "end", "font-size": 11, fill: "var(--muted)" }, s).textContent = "est. − full";

    names.forEach(function (n, i) {
      var cy = top + i * rowH + rowH / 2;
      var xf = x(P.full[n]), xe = x(c.est[n]);
      var g = svg("g", {}, s);
      svg("text", { x: 0, y: cy + 4, "font-size": 13, fill: "var(--ink-2)" }, g).textContent = n;
      svg("line", { x1: Math.min(xf, xe), x2: Math.max(xf, xe), y1: cy, y2: cy, stroke: "var(--axis)", "stroke-width": 2, "stroke-linecap": "round" }, g);
      svg("circle", { cx: xf, cy: cy, r: 5, fill: "var(--accent-soft)", stroke: "var(--surface)", "stroke-width": 2 }, g);
      svg("circle", { cx: xe, cy: cy, r: 5, fill: "var(--accent-est)", stroke: "var(--surface)", "stroke-width": 2 }, g);
      svg("text", { x: W - 2, y: cy + 4, "text-anchor": "end", "font-size": 12, fill: "var(--muted)", style: "font-variant-numeric: tabular-nums" }, g).textContent = signed(c.est[n] - P.full[n]);
      var rf = rankFull.indexOf(n) + 1, re = rankEst.indexOf(n) + 1;
      var hit = svg("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: "transparent" }, g);
      var lines = [
        { text: f3(c.est[n]) + " estimated" },
        { text: n },
        { text: "full evaluation " + f3(P.full[n]) + " · error " + signed(c.est[n] - P.full[n]), muted: true },
        { text: rf === re ? "rank #" + rf + " kept" : "rank #" + rf + " → #" + re, muted: true }
      ];
      bindTip(hit, lines, n + ": full " + f3(P.full[n]) + ", estimate " + f3(c.est[n]));
    });

    var t = document.getElementById("dumbbell-table");
    clear(t);
    var hr = html("tr", {}, html("thead", {}, t));
    ["Model", "Full evaluation", "Tally estimate", "Error", "Rank"].forEach(function (h, i) { html("th", i ? { class: "num" } : {}, hr, h); });
    var tb = html("tbody", {}, t);
    names.forEach(function (n) {
      var tr = html("tr", {}, tb);
      html("td", {}, tr, n);
      html("td", { class: "num" }, tr, f3(P.full[n]));
      html("td", { class: "num" }, tr, f3(c.est[n]));
      html("td", { class: "num" }, tr, signed(c.est[n] - P.full[n]));
      html("td", { class: "num" }, tr, "#" + (rankFull.indexOf(n) + 1) + " → #" + (rankEst.indexOf(n) + 1));
    });
  }

  function renderPlanner() {
    var P = D.planner[state.bench], c = cell();
    certaintyOut.textContent = state.certainty === 100 ? "never skip" : state.certainty + "% sure";
    renderTiles(c, P);
    renderDumbbell(c, P);
  }

  Array.prototype.forEach.call(document.querySelectorAll(".seg"), function (seg) {
    seg.addEventListener("click", function (e) {
      var b = e.target.closest("button");
      if (!b) return;
      state[seg.getAttribute("data-key")] = b.getAttribute("data-value");
      Array.prototype.forEach.call(seg.querySelectorAll("button"), function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
      renderPlanner();
    });
  });
  certainty.addEventListener("input", function () { state.certainty = +certainty.value; renderPlanner(); });

  // ------------------------------------------------------------------ real run: ranking
  function renderRanking() {
    var R = D.run, C = R.compare, box = document.getElementById("ranking");
    clear(box);
    var ex = Object.keys(C.excluded).map(function (n) { return n + " (" + C.excluded[n] + " tasks)"; });
    document.getElementById("compare-note").textContent =
      "Every model is scored on the same " + C.tasks + " tasks, using only frontier attempts where the agent was not told whether it was right; " +
      "half the study's attempts had that oracle feedback, which roughly doubles the weaker models. " +
      (ex.length ? ex.join(" and ") + " have too few runs without feedback to include. " : "") +
      "Budgets still differ: the frontier models ran with 10M-token trajectories, Nemotron Nano with 60 turns and 2K thinking tokens. " +
      "Across all " + R.tasks_all + " tasks Nano estimates " + f3(R.estimate) + ".";
    var W = Math.max(300, box.clientWidth || 600);
    var rows = Object.keys(C.frontier).map(function (n) { return { name: n, v: C.frontier[n], ours: false }; });
    rows.push({ name: R.model, v: C.nano, ours: true, ci: C.ci });
    rows.sort(function (a, b) { return b.v - a.v; });
    var left = W < 460 ? 118 : 150, right = W < 460 ? 36 : 56, top = 8, rowH = 32, bar = 16, bottom = 26;
    var H = top + rows.length * rowH + bottom;
    var x = function (v) { return left + v * (W - left - right); };
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, box);
    [0, 0.25, 0.5, 0.75, 1].forEach(function (tv) {
      svg("line", { x1: x(tv), x2: x(tv), y1: top, y2: H - bottom, stroke: tv === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }, s);
      svg("text", { x: x(tv), y: H - 8, "text-anchor": "middle", "font-size": 11, fill: "var(--muted)" }, s).textContent = tv === 0 ? "0" : tv.toFixed(2);
    });
    rows.forEach(function (r, i) {
      var cy = top + i * rowH + rowH / 2, x0 = x(0), x1 = x(r.v), y0 = cy - bar / 2;
      var g = svg("g", {}, s);
      svg("text", { x: 0, y: cy + 4, "font-size": 13, fill: r.ours ? "var(--ink)" : "var(--ink-2)", "font-weight": r.ours ? 600 : 400 }, g).textContent = r.name;
      var len = Math.max(0, x1 - x0), rad = Math.min(4, len, bar / 2);
      svg("path", {
        d: "M" + x0 + "," + y0 + "H" + (x1 - rad) + "Q" + x1 + "," + y0 + " " + x1 + "," + (y0 + rad) +
           "V" + (y0 + bar - rad) + "Q" + x1 + "," + (y0 + bar) + " " + (x1 - rad) + "," + (y0 + bar) + "H" + x0 + "Z",
        fill: r.ours ? "var(--accent)" : "var(--deemph)"
      }, g);
      var labelX = x1 + 8, label = f3(r.v);
      if (r.ours) {
        var a = x(r.ci[0]), b = x(r.ci[1]);
        svg("line", { x1: a, x2: b, y1: cy, y2: cy, stroke: "var(--ink-2)", "stroke-width": 2 }, g);
        svg("line", { x1: a, x2: a, y1: cy - 5, y2: cy + 5, stroke: "var(--ink-2)", "stroke-width": 2 }, g);
        svg("line", { x1: b, x2: b, y1: cy - 5, y2: cy + 5, stroke: "var(--ink-2)", "stroke-width": 2 }, g);
        labelX = b + 8;
        label = f3(r.v) + "  (" + f3(r.ci[0]) + "–" + f3(r.ci[1]) + ")";
      }
      svg("text", { x: labelX, y: cy + 4, "font-size": 12, fill: r.ours ? "var(--ink)" : "var(--ink-2)", style: "font-variant-numeric: tabular-nums" }, g).textContent = label;
      var hit = svg("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: "transparent" }, g);
      bindTip(hit, r.ours
        ? [{ text: f3(r.v) + " estimated" }, { text: r.name + " · this run, no feedback" }, { text: "95% interval " + f3(r.ci[0]) + "–" + f3(r.ci[1]) + " on the same " + C.tasks + " tasks", muted: true }, { text: f3(R.estimate) + " across all " + R.tasks_all + " tasks", muted: true }]
        : [{ text: f3(r.v) }, { text: r.name + " · study logs, no feedback" }, { text: "mean over the same " + C.tasks + " tasks", muted: true }],
        r.name + ": " + f3(r.v));
    });

    var t = document.getElementById("ranking-table");
    clear(t);
    var hr = html("tr", {}, html("thead", {}, t));
    ["Model", "Accuracy", "Source"].forEach(function (h, i) { html("th", i === 1 ? { class: "num" } : {}, hr, h); });
    var tb = html("tbody", {}, t);
    rows.forEach(function (r) {
      var tr = html("tr", {}, tb);
      html("td", {}, tr, r.name);
      html("td", { class: "num" }, tr, r.ours ? f3(r.v) + " (" + f3(r.ci[0]) + "–" + f3(r.ci[1]) + ")" : f3(r.v));
      html("td", {}, tr, r.ours ? "this run, 3 attempts per task, no feedback" : "study logs, attempts without feedback");
    });
  }

  // ------------------------------------------------------------------ real run: consistency grid
  function renderConsistency() {
    var R = D.run, box = document.getElementById("consistency");
    clear(box);
    var longest = R.solved.reduce(function (m, s) { return Math.max(m, s.task.length); }, 0);
    var left = Math.ceil(longest * 7.3) + 12, cw = 30, ch = 22, gap = 2, top = 20, right = 44;
    var W = left + 3 * (cw + gap) + right, H = top + R.solved.length * (ch + gap);
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, style: "max-width:" + W + "px" }, box);
    for (var a = 0; a < 3; a++) {
      svg("text", { x: left + a * (cw + gap) + cw / 2, y: 12, "text-anchor": "middle", "font-size": 11, fill: "var(--muted)" }, s).textContent = "#" + (a + 1);
    }
    svg("text", { x: W, y: 12, "text-anchor": "end", "font-size": 11, fill: "var(--muted)" }, s).textContent = "passed";
    R.solved.forEach(function (row, i) {
      var y = top + i * (ch + gap);
      svg("text", { x: 0, y: y + ch / 2 + 4, "font-size": 12, fill: "var(--ink-2)", "font-family": "var(--mono)" }, s).textContent = row.task;
      var passes = 0;
      row.attempts.forEach(function (at, j) {
        var pass = at.outcome === "pass";
        passes += pass ? 1 : 0;
        var r = svg("rect", { x: left + j * (cw + gap), y: y, width: cw, height: ch, rx: 3, fill: pass ? "var(--accent)" : "var(--cell-fail)" }, s);
        bindTip(r, [
          { text: pass ? "Pass" : at.outcome === "timeout" ? "Timed out" : "Fail" },
          { text: row.task + " · attempt " + (j + 1) },
          { text: at.steps + " agent steps · " + at.minutes + " min · " + Math.round(at.tokens / 1000) + "K tokens", muted: true }
        ], row.task + " attempt " + (j + 1) + ": " + at.outcome);
      });
      svg("text", { x: W, y: y + ch / 2 + 4, "text-anchor": "end", "font-size": 12, fill: "var(--ink-2)", style: "font-variant-numeric: tabular-nums" }, s).textContent = passes + "/" + row.attempts.length;
    });
    document.getElementById("grid-foot").textContent = "+ " + R.always_fail + " more tasks failed all three attempts";
    var flaky = R.solved.length - R.always_pass.length;
    document.getElementById("flaky").textContent = flaky + " of " + R.solved.length;

    var t = document.getElementById("consistency-table");
    clear(t);
    var hr = html("tr", {}, html("thead", {}, t));
    ["Task", "Attempt 1", "Attempt 2", "Attempt 3"].forEach(function (h) { html("th", {}, hr, h); });
    var tb = html("tbody", {}, t);
    R.solved.forEach(function (row) {
      var tr = html("tr", {}, tb);
      html("td", { class: "mono" }, tr, row.task);
      row.attempts.forEach(function (at) { html("td", {}, tr, at.outcome + " · " + at.steps + " steps"); });
    });
  }

  // ------------------------------------------------------------------ receipt
  function renderReceipt() {
    var R = D.run, rc = R.receipt, box = document.getElementById("receipt");
    clear(box);
    html("div", { class: "r-title" }, box, "Evaluation receipt");
    html("div", { class: "r-sub" }, box, R.model_id);
    html("div", { class: "r-sub" }, box, "terminal-bench@2.0 · terminus-2");
    html("hr", {}, box);
    function row(k, v, cls) {
      var r = html("div", { class: "row" + (cls ? " " + cls : "") }, box);
      html("span", {}, r, k); html("span", {}, r, v);
    }
    row("Tasks run", R.tasks_run + " × 3");
    row("Tasks skipped", String(R.tasks_all - R.tasks_run), "sub");
    row("Trials", String(R.trials));
    row("passed", String(R.passes), "sub");
    row("timed out", String(R.timeouts), "sub");
    row("Infra errors", "0");
    html("hr", {}, box);
    row("Input tokens", intc(rc.tokens_in));
    row("served from cache", intc(rc.tokens_cached), "sub");
    row("Output tokens", intc(rc.tokens_out));
    row("@ $" + rc.price_in_per_m.toFixed(2) + "/M in, $" + rc.price_out_per_m.toFixed(2) + "/M out", "", "sub");
    row("Inference", "$" + rc.inference_usd.toFixed(2));
    row("Sandbox VM " + rc.vm_hours.toFixed(1) + " h @ $" + rc.vm_per_hour.toFixed(2), "$" + rc.vm_usd.toFixed(2));
    html("hr", { class: "double" }, box);
    row("TOTAL", "$" + (rc.inference_usd + rc.vm_usd).toFixed(2), "total");
    html("hr", {}, box);
    row("Estimated accuracy", f3(R.estimate));
    row("95% interval", f3(R.ci[0]) + "–" + f3(R.ci[1]), "sub");
    row("Rank, like for like", (Object.keys(R.compare.frontier).filter(function (n) { return R.compare.frontier[n] > R.compare.nano; }).length + 1) + " of " + (Object.keys(R.compare.frontier).length + 1));
    html("div", { class: "r-foot" }, box,
      "Nebius Token Factory · Nebius AI Cloud cpu-d3 · trials " + rc.sittings.map(function (p) { return p[0] + "–" + p[1]; }).join(", ") + " UTC. VM hours count running trials only.");
  }

  // ------------------------------------------------------------------ tables
  function renderValidation() {
    var t = document.getElementById("validation-table");
    var hr = html("tr", {}, html("thead", {}, t));
    ["Task", "History", "Said", "Result"].forEach(function (h, i) { html("th", i === 1 ? { class: "num" } : {}, hr, h); });
    var tb = html("tbody", {}, t);
    D.validation.forEach(function (v) {
      var tr = html("tr", {}, tb);
      html("td", { class: "mono" }, tr, v.task);
      html("td", { class: "num" }, tr, v.prior.toFixed(2));
      html("td", {}, tr, "certain " + v.predicted);
      var ok = (v.predicted === "pass") === (v.actual === "pass");
      var td = html("td", {}, tr);
      var m = html("span", { class: ok ? "mark-ok" : "mark-miss" }, td);
      m.appendChild(icon(ok ? "ok" : "warn"));
      html("span", {}, m, ok ? "held" : "missed");
    });
  }
  function renderCold() {
    var t = document.getElementById("cold-table");
    var hr = html("tr", {}, html("thead", {}, t));
    ["Benchmark", "Nemotron read", "Issue length"].forEach(function (h, i) { html("th", i ? { class: "num" } : {}, hr, h); });
    var tb = html("tbody", {}, t);
    [["Terminal-Bench 2.0", D.coldstart.terminalbench], ["SWE-bench Pro", D.coldstart.swebenchpro]].forEach(function (p) {
      var tr = html("tr", {}, tb);
      html("td", {}, tr, p[0]);
      html("td", { class: "num" }, tr, p[1].model.toFixed(2));
      html("td", { class: "num" }, tr, p[1].length.toFixed(2));
    });
    var tf = html("tr", {}, html("tfoot", {}, t));
    html("td", { colspan: 3, class: "note", style: "white-space:normal;border:0;padding-top:8px" }, tf, "Spearman correlation with true task difficulty; 0 is no signal.");
  }

  // ------------------------------------------------------------------ boot
  renderPlanner();
  renderRanking();
  renderConsistency();
  renderReceipt();
  renderValidation();
  renderCold();

  var pending = false;
  function rerender() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(function () { pending = false; renderDumbbell(cell(), D.planner[state.bench]); renderRanking(); });
  }
  if ("ResizeObserver" in window) {
    new ResizeObserver(rerender).observe(document.getElementById("dumbbell"));
    new ResizeObserver(rerender).observe(document.getElementById("ranking"));
  } else {
    window.addEventListener("resize", rerender);
  }
})();
