/* NetPulse 输出探针 Canvas 渲染模块（从 index.html 拆出，零构建直接引入） */

const COLORMAPS = {
  viridis: [[68,1,84],[72,40,120],[62,73,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[110,206,88],[181,222,43],[253,231,37]],
  coolwarm: [[59,76,192],[98,130,234],[141,176,254],[184,208,249],[221,221,221],[245,196,173],[244,154,123],[222,96,77],[180,4,38]],
  magma: [[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],[229,80,100],[251,135,97],[254,194,135],[252,253,191]],
  grayscale: [[0,0,0],[255,255,255]],
};

function sampleColormap(name, v255) {
  const stops = COLORMAPS[name] || COLORMAPS.viridis;
  const t = Math.max(0, Math.min(1, v255 / 255));
  const scaled = t * (stops.length - 1);
  const idx = Math.min(stops.length - 2, Math.floor(scaled));
  const frac = scaled - idx;
  const c0 = stops[idx], c1 = stops[idx + 1];
  return [
    Math.round(c0[0] + (c1[0] - c0[0]) * frac),
    Math.round(c0[1] + (c1[1] - c0[1]) * frac),
    Math.round(c0[2] + (c1[2] - c0[2]) * frac),
  ];
}

function drawHeatmapCanvas(canvas, heatmap) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!heatmap || !heatmap.rows || !heatmap.cols) return;

  const { rows, cols, data, colormap, vmin, vmax } = heatmap;
  const off = document.createElement("canvas");
  off.width = cols;
  off.height = rows;
  const octx = off.getContext("2d");
  const img = octx.createImageData(cols, rows);
  for (let i = 0; i < data.length; i++) {
    const [r, g, b] = sampleColormap(colormap, data[i]);
    const p = i * 4;
    img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
  }
  octx.putImageData(img, 0, 0);

  const barH = 16;
  const plotH = H - barH - 4;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, W, plotH);

  // 底部色标条
  const grad = ctx.createLinearGradient(6, 0, W - 6, 0);
  const stops = COLORMAPS[colormap] || COLORMAPS.viridis;
  stops.forEach((rgb, idx) => grad.addColorStop(idx / (stops.length - 1), `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`));
  ctx.fillStyle = grad;
  ctx.fillRect(6, plotH + 3, W - 12, 5);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "9px ui-monospace, monospace";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText(vmin.toFixed(2), 6, plotH + 9);
  ctx.textAlign = "right";
  ctx.fillText(vmax.toFixed(2), W - 6, plotH + 9);
  canvas._heatMeta = { rows, cols, data, vmin, vmax, plotH };
}

// 反向梯度视图：上半部分梯度分布热力图，下半部分梯度范数历史曲线
function drawGradCanvas(canvas, grad) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  canvas._heatMeta = null;

  const heat = grad.heatmap;
  const histLen = grad.norm_history || [];
  const curveH = Math.max(38, Math.round(H * 0.36));
  const heatH = Math.max(20, H - curveH - 16);

  if (heat) {
    const off = document.createElement("canvas");
    off.width = heat.cols;
    off.height = heat.rows;
    const octx = off.getContext("2d");
    const img = octx.createImageData(heat.cols, heat.rows);
    for (let i = 0; i < heat.data.length; i++) {
      const [r, g, b] = sampleColormap("coolwarm", heat.data[i]);
      const p = i * 4;
      img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, W, heatH);
    canvas._heatMeta = { rows: heat.rows, cols: heat.cols, data: heat.data, vmin: heat.vmin, vmax: heat.vmax, plotH: heatH };
  }

  // 梯度范数曲线
  const mL = 8, mR = 8, mT = heatH + 6, mB = 12;
  const pw = W - mL - mR, ph = Math.max(10, H - mT - mB);
  const vals = histLen.length ? histLen : [0];
  let minY = Math.min(...vals), maxY = Math.max(...vals);
  if (minY === maxY) { minY = Math.max(0, minY - 0.5); maxY += 0.5; }
  const mapX = (i) => vals.length === 1 ? mL + pw / 2 : mL + (i / (vals.length - 1)) * pw;
  const mapY = (v) => mT + ph - ((v - minY) / Math.max(1e-8, maxY - minY)) * ph;

  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(mL, mT + ph); ctx.lineTo(W - mR, mT + ph); ctx.stroke();

  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  vals.forEach((v, i) => {
    const x = mapX(i), y = mapY(v);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#e2e8f0";
  ctx.font = "bold 12px ui-monospace, monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(`∇n=${Number(grad.stats.norm).toFixed(3)}`, 8, heatH - 16);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "9px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.fillText(`σ=${Number(grad.stats.std).toFixed(3)}`, W - 8, heatH - 15);
}

function drawScalarCanvas(canvas, probe, histObj) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  canvas._heatMeta = null;

  const vals = histObj?.values || [];
  const emas = histObj?.ema || [];
  if (!vals.length) return;

  let minY = Math.min(...vals), maxY = Math.max(...vals);
  if (minY === maxY) { minY -= 0.5; maxY += 0.5; }
  const pad = (maxY - minY) * 0.1;
  minY -= pad; maxY += pad;

  const mL = 8, mR = 8, mT = 26, mB = 14;
  const pw = W - mL - mR, ph = H - mT - mB;
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 2; g++) {
    const gy = mT + (ph * g) / 2;
    ctx.beginPath(); ctx.moveTo(mL, gy); ctx.lineTo(W - mR, gy); ctx.stroke();
  }

  const mapX = (i) => vals.length === 1 ? mL + pw / 2 : mL + (i / (vals.length - 1)) * pw;
  const mapY = (v) => mT + ph - ((v - minY) / Math.max(1e-8, maxY - minY)) * ph;

  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  vals.forEach((v, i) => {
    const x = mapX(i), y = mapY(v);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  if (emas.length >= 2 && (probe.ema_alpha || 0) > 0) {
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    emas.forEach((v, i) => {
      const x = mapX(i), y = mapY(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }

  ctx.fillStyle = "#e2e8f0";
  ctx.font = "bold 14px ui-monospace, monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(Number(probe.scalar).toFixed(4), 8, 5);

  ctx.fillStyle = "#94a3b8";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.fillText(`${probe.reduce || "val"} · n=${vals.length}`, W - 8, 7);
}

function drawImagesCanvas(canvas, images) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  canvas._heatMeta = null;
  if (!images || !images.samples || !images.samples.length) return;

  const { count, channels, height, width, samples } = images;
  const cols = count === 1 ? 1 : 2;
  const rows = Math.ceil(count / cols);
  const gap = 4;
  const cellW = Math.floor((W - gap * (cols + 1)) / cols);
  const cellH = Math.floor((H - gap * (rows + 1)) / rows);
  ctx.imageSmoothingEnabled = false;

  samples.forEach((flat, idx) => {
    const off = document.createElement("canvas");
    off.width = width;
    off.height = height;
    const octx = off.getContext("2d");
    const img = octx.createImageData(width, height);
    for (let i = 0; i < width * height; i++) {
      const p = i * 4;
      if (channels === 3) {
        img.data[p] = flat[i * 3]; img.data[p + 1] = flat[i * 3 + 1]; img.data[p + 2] = flat[i * 3 + 2];
      } else {
        const v = flat[i];
        img.data[p] = v; img.data[p + 1] = v; img.data[p + 2] = v;
      }
      img.data[p + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    const gx = idx % cols, gy = Math.floor(idx / cols);
    const scale = Math.min(cellW / width, cellH / height);
    const dw = Math.max(1, Math.floor(width * scale));
    const dh = Math.max(1, Math.floor(height * scale));
    const dx = gap + gx * (cellW + gap) + Math.floor((cellW - dw) / 2);
    const dy = gap + gy * (cellH + gap) + Math.floor((cellH - dh) / 2);
    ctx.drawImage(off, dx, dy, dw, dh);
  });
}

function drawHistogramCanvas(canvas, hist) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  canvas._heatMeta = null;
  if (!hist || !hist.counts || !hist.counts.length) return;
  const counts = hist.counts;
  const maxC = Math.max(1, ...counts);
  const mL = 8, mR = 8, mT = 10, mB = 18;
  const bw = (W - mL - mR) / counts.length;
  const ph = H - mT - mB;
  counts.forEach((c, idx) => {
    const bh = (c / maxC) * ph;
    ctx.fillStyle = "#06b6d4";
    ctx.fillRect(mL + idx * bw + 1, mT + ph - bh, Math.max(2, bw - 2), bh);
  });
  ctx.fillStyle = "#94a3b8";
  ctx.font = "9px ui-monospace, monospace";
  ctx.textBaseline = "bottom";
  ctx.textAlign = "left";
  ctx.fillText(hist.vmin.toFixed(3), 6, H - 2);
  ctx.textAlign = "right";
  ctx.fillText(hist.vmax.toFixed(3), W - 6, H - 2);
}

function renderRawTableHtml(raw) {
  if (!raw || !raw.values || !raw.values.length) return `<div style="color:var(--muted);font-size:11px">无数据</div>`;
  const rowsHtml = raw.values.map((row) =>
    `<tr>${row.map((v) => `<td>${v}</td>`).join("")}</tr>`
  ).join("");
  return `<div style="font-size:10px;color:var(--muted);margin-bottom:3px;font-family:ui-monospace,monospace">显示 ${raw.rows}×${raw.cols} / 原始 ${raw.total_rows}×${raw.total_cols}</div><table class="probe-raw-table">${rowsHtml}</table>`;
}

const GATE_COLORS = {
  i: "#38bdf8",
  f: "#fbbf24",
  o: "#f472b6",
  c: "#34d399",
};

function drawGateDynamicsCanvas(canvas, probe) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  canvas._heatMeta = null;
  canvas._trajMeta = null;
  canvas._energyMeta = null;

  const gd = probe.gate_dynamics;
  if (!gd || !gd.gates || !Object.keys(gd.gates).length) {
    canvas._gateMeta = null;
    return;
  }

  const gates = gd.gates;
  const keys = ["i", "f", "o", "c"].filter((k) => gates[k]);
  const steps = Math.max(1, gd.steps || 1);

  const mL = 16, mR = 8, mT = 18, mB = 14;
  const stripRows = gd.strip_rows || [];
  const hasStrip = stripRows.length > 0;
  const stripH = hasStrip ? Math.min(Math.floor(H * 0.34), Math.max(26, stripRows.length * 4 + 8)) : 0;
  const curveH = Math.max(30, H - mT - mB - stripH - (hasStrip ? 8 : 0));
  const pw = Math.max(20, W - mL - mR);

  // 背景参考线 0 / 0.5 / 1
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  [0, 0.5, 1].forEach((lv) => {
    const gy = mT + curveH * (1 - lv);
    ctx.beginPath();
    if (lv === 0.5) ctx.setLineDash([3, 3]);
    else ctx.setLineDash([]);
    ctx.moveTo(mL, gy);
    ctx.lineTo(W - mR, gy);
    ctx.stroke();
  });
  ctx.setLineDash([]);

  const mapX = (t) => (steps === 1 ? mL + pw / 2 : mL + (t / (steps - 1)) * pw);

  let cMin = -1, cMax = 1;
  if (gates.c) {
    const vals = gates.c.mean;
    cMin = Math.min(...vals);
    cMax = Math.max(...vals);
    if (Math.abs(cMax - cMin) < 1e-5) { cMin -= 0.5; cMax += 0.5; }
  }

  keys.forEach((k) => {
    const arr = gates[k].mean || [];
    if (!arr.length) return;
    ctx.strokeStyle = GATE_COLORS[k] || "#94a3b8";
    ctx.lineWidth = k === "f" ? 2.0 : 1.5;
    ctx.beginPath();
    arr.forEach((v, idx) => {
      const normY = k === "c" ? (v - cMin) / Math.max(1e-6, cMax - cMin) : Math.max(0, Math.min(1, v));
      const x = mapX(idx);
      const y = mT + curveH * (1 - normY);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  let lx = mL;
  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  keys.forEach((k) => {
    ctx.fillStyle = GATE_COLORS[k] || "#e2e8f0";
    const g = gates[k];
    const satPct = k === "c"
      ? `|c|=${Number(g.abs_mean).toFixed(2)}`
      : `${k}:${Math.round((g.sat_low + g.sat_high) * 100)}%sat`;
    ctx.fillText(satPct, lx, 3);
    lx += ctx.measureText(satPct).width + 8;
  });
  ctx.fillStyle = "#94a3b8";
  ctx.textAlign = "right";
  ctx.fillText(`L=${gd.valid_len}/${gd.total_len}`, W - mR, 3);

  if (hasStrip) {
    const sy = mT + curveH + 6;
    const rows = stripRows.length;
    const rowH = Math.max(1, stripH / rows);
    const colW = pw / steps;
    const labels = gd.strip_labels || [];
    for (let r = 0; r < rows; r++) {
      const rowData = stripRows[r];
      const ry = sy + r * rowH;
      if (labels[r]) {
        ctx.fillStyle = GATE_COLORS[labels[r]] || "#94a3b8";
        ctx.font = "9px ui-monospace, monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(labels[r], 3, ry);
      }
      for (let c = 0; c < rowData.length; c++) {
        const [cr, cg, cb] = sampleColormap("viridis", rowData[c]);
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
        ctx.fillRect(mL + c * colW, ry, Math.ceil(colW), Math.ceil(rowH));
      }
    }
  }

  canvas._gateMeta = { mL, pw, steps, gates, keys };
}

function drawTrajectoryCanvas(canvas, probe) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  canvas._heatMeta = null;
  canvas._gateMeta = null;
  canvas._energyMeta = null;

  const traj = probe.trajectory;
  if (!traj || !traj.points || !traj.points.length) {
    canvas._trajMeta = null;
    return;
  }

  const pts = traj.points;
  const mL = 12, mR = 12, mT = 20, mB = 14;
  const pw = W - mL - mR, ph = H - mT - mB;

  let xmin = traj.xmin, xmax = traj.xmax, ymin = traj.ymin, ymax = traj.ymax;
  if (Math.abs(xmax - xmin) < 1e-5) { xmin -= 0.5; xmax += 0.5; }
  if (Math.abs(ymax - ymin) < 1e-5) { ymin -= 0.5; ymax += 0.5; }
  const padX = (xmax - xmin) * 0.14;
  const padY = (ymax - ymin) * 0.14;
  xmin -= padX; xmax += padX; ymin -= padY; ymax += padY;

  const mapX = (x) => mL + ((x - xmin) / Math.max(1e-6, xmax - xmin)) * pw;
  const mapY = (y) => mT + ph - ((y - ymin) / Math.max(1e-6, ymax - ymin)) * ph;

  ctx.strokeStyle = "#172033";
  ctx.lineWidth = 1;
  ctx.strokeRect(mL, mT, pw, ph);
  if (xmin < 0 && xmax > 0) {
    const zx = mapX(0);
    ctx.beginPath(); ctx.setLineDash([3, 3]); ctx.moveTo(zx, mT); ctx.lineTo(zx, mT + ph); ctx.stroke();
  }
  if (ymin < 0 && ymax > 0) {
    const zy = mapY(0);
    ctx.beginPath(); ctx.setLineDash([3, 3]); ctx.moveTo(mL, zy); ctx.lineTo(W - mR, zy); ctx.stroke();
  }
  ctx.setLineDash([]);

  const mapped = pts.map(([px, py]) => [mapX(px), mapY(py)]);

  for (let i = 0; i < mapped.length - 1; i++) {
    const frac = i / Math.max(1, mapped.length - 2);
    const r = Math.round(56 + (244 - 56) * frac);
    const g = Math.round(189 + (114 - 189) * frac);
    const b = Math.round(248 + (182 - 248) * frac);
    ctx.strokeStyle = `rgb(${r},${g},${b})`;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.moveTo(mapped[i][0], mapped[i][1]);
    ctx.lineTo(mapped[i + 1][0], mapped[i + 1][1]);
    ctx.stroke();
  }

  mapped.forEach(([cx, cy], idx) => {
    const isStart = idx === 0;
    const isEnd = idx === mapped.length - 1;
    ctx.fillStyle = isStart ? "#34d399" : (isEnd ? "#f472b6" : "rgba(226,232,240,0.65)");
    ctx.beginPath();
    ctx.arc(cx, cy, isStart || isEnd ? 4.2 : 2.0, 0, Math.PI * 2);
    ctx.fill();
  });

  if (mapped.length >= 1) {
    ctx.font = "bold 9px ui-monospace, monospace";
    ctx.textBaseline = "bottom";
    ctx.textAlign = "left";
    ctx.fillStyle = "#34d399";
    ctx.fillText("S", mapped[0][0] + 5, mapped[0][1] - 2);
    ctx.fillStyle = "#f472b6";
    const last = mapped[mapped.length - 1];
    ctx.fillText("E", last[0] + 5, last[1] - 2);
  }

  ctx.fillStyle = "#94a3b8";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  const evrTxt = traj.method === "pca" ? `PCA ${(traj.explained_var * 100).toFixed(0)}%` : "first2";
  ctx.fillText(`${evrTxt} · len=${Number(traj.path_len).toFixed(2)}`, mL, 4);
  ctx.textAlign = "right";
  ctx.fillText(`n=${pts.length}`, W - mR, 4);

  canvas._trajMeta = { mapped, pts };
}

function drawStepEnergyCanvas(canvas, probe) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  canvas._heatMeta = null;
  canvas._gateMeta = null;
  canvas._trajMeta = null;

  const se = probe.step_energy;
  if (!se || !se.energy || !se.energy.length) {
    canvas._energyMeta = null;
    return;
  }

  const energy = se.energy;
  const delta = se.delta || [];
  const n = energy.length;
  const maxVal = Math.max(1e-5, ...energy, ...delta);

  const mL = 8, mR = 8, mT = 20, mB = 14;
  const pw = W - mL - mR, ph = H - mT - mB;
  const bw = pw / n;

  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(mL, mT + ph); ctx.lineTo(W - mR, mT + ph); ctx.stroke();

  energy.forEach((val, idx) => {
    const bh = (val / maxVal) * ph;
    const isPeak = idx === se.peak_step;
    ctx.fillStyle = isPeak ? "#fbbf24" : "rgba(56, 189, 248, 0.55)";
    ctx.fillRect(mL + idx * bw + 1, mT + ph - bh, Math.max(2, bw - 2), bh);
  });

  if (delta.length >= 2) {
    ctx.strokeStyle = "#f472b6";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    delta.forEach((dv, idx) => {
      const x = mL + (idx + 0.5) * bw;
      const y = mT + ph - (dv / maxVal) * ph;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillStyle = "#fbbf24";
  ctx.fillText(`t*=${se.peak_step} (${Number(se.peak_val).toFixed(2)})`, mL, 4);
  ctx.textAlign = "right";
  ctx.fillStyle = "#94a3b8";
  ctx.fillText(`decay=${Number(se.decay_ratio).toFixed(2)} · L=${se.valid_len}/${se.total_len}`, W - mR, 4);

  canvas._energyMeta = { mL, bw, n, energy, delta };
}

function renderProbeIntoTarget(canvas, rawWrap, probe, histObj, forcedView) {
  let view = forcedView && forcedView !== "auto" ? forcedView : probe.view;
  if (view === "image" && !probe.images) view = "heatmap";
  if (view === "gate_dynamics" && !probe.gate_dynamics) view = probe.view;
  if (view === "trajectory" && !probe.trajectory) view = probe.view;
  if (view === "step_energy" && !probe.step_energy) view = probe.view;
  if (view === "grad" && !probe.grad) view = probe.view;

  if (view === "raw") {
    canvas.style.display = "none";
    rawWrap.style.display = "block";
    rawWrap.innerHTML = renderRawTableHtml(probe.raw);
  } else {
    canvas.style.display = "block";
    rawWrap.style.display = "none";
    if (view === "grad") drawGradCanvas(canvas, probe.grad);
    else if (view === "gate_dynamics") drawGateDynamicsCanvas(canvas, probe);
    else if (view === "trajectory") drawTrajectoryCanvas(canvas, probe);
    else if (view === "step_energy") drawStepEnergyCanvas(canvas, probe);
    else if (view === "scalar") drawScalarCanvas(canvas, probe, histObj);
    else if (view === "image") drawImagesCanvas(canvas, probe.images);
    else if (view === "hist") drawHistogramCanvas(canvas, probe.hist);
    else drawHeatmapCanvas(canvas, probe.heatmap);
  }
}

function renderProbeIntoCard(card, nid, probe, histObj) {
  const probeBox = card.querySelector(".node-probe");
  if (!probeBox || !probe) return;
  const tag = probeBox.querySelector(".probe-tag");
  const shapeTxt = probeBox.querySelector(".probe-shape-txt");
  const meta = probeBox.querySelector(".probe-meta");
  const canvas = probeBox.querySelector(".probe-canvas");
  const rawWrap = probeBox.querySelector(".probe-raw-wrap");
  if (tag) tag.textContent = probe.view;
  if (shapeTxt) shapeTxt.textContent = `(${probe.shape.join(",")}) ${probe.dtype}`;
  if (meta && probe.stats) {
    let html = `<span>min ${probe.stats.min.toFixed(2)}</span><span>max ${probe.stats.max.toFixed(2)}</span><span>μ ${probe.stats.mean.toFixed(2)}</span>`;
    if (probe.grad) {
      html += `<span style="color:#22d3ee">∇ ${Number(probe.grad.stats.norm).toFixed(2)}</span>`;
    }
    meta.innerHTML = html;
  }
  const history = histObj !== undefined ? histObj : (typeof state !== "undefined" ? state.probeHistory[nid] : null);
  renderProbeIntoTarget(canvas, rawWrap, probe, history, "auto");
}

function handleProbeHover(ev, nid, canvas, hoverEl) {
  if (!hoverEl) return;
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * (canvas.width / Math.max(1, rect.width));
  const y = (ev.clientY - rect.top) * (canvas.height / Math.max(1, rect.height));

  if (canvas._gateMeta) {
    const gm = canvas._gateMeta;
    const t = Math.max(0, Math.min(gm.steps - 1, Math.round(((x - gm.mL) / gm.pw) * Math.max(1, gm.steps - 1))));
    const parts = gm.keys.map((k) => `${k}=${Number(gm.gates[k].mean[t] ?? 0).toFixed(3)}`);
    hoverEl.textContent = `t=${t}: ${parts.join("  ")}`;
    return;
  }
  if (canvas._trajMeta) {
    const tm = canvas._trajMeta;
    let bestIdx = 0, bestD = Infinity;
    tm.mapped.forEach(([mx, my], idx) => {
      const d = (mx - x) ** 2 + (my - y) ** 2;
      if (d < bestD) { bestD = d; bestIdx = idx; }
    });
    const pt = tm.pts[bestIdx];
    hoverEl.textContent = `t=${bestIdx}: (${pt[0].toFixed(3)}, ${pt[1].toFixed(3)})`;
    return;
  }
  if (canvas._energyMeta) {
    const em = canvas._energyMeta;
    const t = Math.max(0, Math.min(em.n - 1, Math.floor((x - em.mL) / Math.max(1e-5, em.bw))));
    hoverEl.textContent = `t=${t}: E=${Number(em.energy[t] ?? 0).toFixed(3)} · Δ=${Number(em.delta[t] ?? 0).toFixed(3)}`;
    return;
  }

  const hm = canvas._heatMeta;
  if (!hm) return;
  if (y < 0 || y >= hm.plotH) return;
  const c = Math.max(0, Math.min(hm.cols - 1, Math.floor((x / canvas.width) * hm.cols)));
  const r = Math.max(0, Math.min(hm.rows - 1, Math.floor((y / hm.plotH) * hm.rows)));
  const q = hm.data[r * hm.cols + c];
  const val = hm.vmin + (q / 255) * (hm.vmax - hm.vmin);
  hoverEl.textContent = `[${r}, ${c}] ≈ ${val.toFixed(4)}`;
}
