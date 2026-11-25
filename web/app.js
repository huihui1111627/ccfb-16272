/* 慢节点监控前端：纯原生 JS，无外部依赖。
 * 数据结构见 slownode/reporting.py 的 export_run。 */
"use strict";

let DATA = null;          // 完整 payload
let runMode = "full";
let metric = "total";
let cursor = 0;           // 当前时间轴步
let playing = false;
let playTimer = null;

const $ = (id) => document.getElementById(id);
const CAUSE_CN = {
  normal: "正常", sporadic: "偶发(观察)", data_driven: "数据倾斜",
  persistent_gpu: "持续慢(建议剔除)", degrading_gpu: "硬件退化(建议剔除)",
  thermal_cycle: "温度周期(不剔)", host_level: "机器级事件",
  cluster_window: "集群时间段(不剔)",
};
const TRUTH_CN = {
  transient: "偶发尖峰", data_skew: "数据倾斜", persistent: "持续慢",
  progressive: "硬件退化", thermal_periodic: "温度降频",
  host_nic_periodic: "网卡争抢", cluster_storage: "存储抖动", recovery: "维修恢复",
};

// 蓝→黄→红 的热力配色
function heatColor(v, vmin, vmax) {
  if (v <= 0) return [0, 0, 0];
  const t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)));
  if (t < 0.5) {
    const k = t / 0.5;
    return [Math.round(22 + 25 * k), Math.round(35 + 76 * k), Math.round(60 + 116 * k)];
  }
  const k = (t - 0.5) / 0.5;
  return [Math.round(47 + 203 * k), Math.round(111 + 69 * k), Math.round(176 - 140 * k)];
}

async function load() {
  const resp = await fetch("/data/experiment.json");
  DATA = await resp.json();
  cursor = DATA.runs.full.steps.length - 1;
  initControls();
  renderAll();
  renderEventLog();
  renderKpis();
}

function initControls() {
  $("runSelect").onchange = (e) => { runMode = e.target.value; renderAll(); };
  $("metricSelect").onchange = (e) => { metric = e.target.value; renderHeatmap(); };
  $("playBtn").onclick = togglePlay;
  $("heat").addEventListener("mousemove", onHeatHover);
  $("heat").addEventListener("mouseleave", () => ($("tooltip").style.display = "none"));
}

function curRun() { return DATA.runs[runMode]; }

function renderAll() {
  renderHeatmap();
  renderSlowPanel();
  renderActionTable();
  renderCharts();
  $("stepPill").textContent = `step ${cursor} / ${curRun().steps.length - 1}`;
}

// ---------- 热力图 ----------
function renderHeatmap() {
  const run = curRun();
  const matrix = run.heatmap[metric];      // [steps][gpus]
  const steps = run.steps.length;
  const ngpu = run.num_gpus;
  const cv = $("heat");
  const dpr = window.devicePixelRatio || 1;
  const W = cv.parentElement.clientWidth;
  const cw = W / steps;                     // 每步像素宽
  const rh = 8;                             // 每卡行高
  cv.width = W * dpr; cv.height = ngpu * rh * dpr;
  cv.style.height = ngpu * rh + "px";
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);

  // 颜色量程用 5%~95% 分位，避免极端慢卡压缩正常卡的色差
  const flat = [];
  for (const row of matrix) for (const x of row) if (x > 0) flat.push(x);
  flat.sort((a, b) => a - b);
  const vmin = flat[Math.floor(flat.length * 0.05)];
  const vmax = flat[Math.floor(flat.length * 0.95)];

  const statesAtCursor = run.gpu_state[cursor] || [];
  for (let s = 0; s < steps; s++) {
    const row = matrix[s];
    for (let g = 0; g < ngpu; g++) {
      const st = statesAtCursor[g];
      let color;
      if (st === "ejected") color = [12, 12, 16];
      else if (st === "watch") color = [167, 139, 250];
      else { const c = heatColor(row[g], vmin, vmax); color = c; }
      ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
      ctx.fillRect(s * cw, g * rh, Math.ceil(cw) + 0.5, rh);
    }
  }
  // 机器分组横线
  ctx.strokeStyle = "#2a3650";
  ctx.lineWidth = 1;
  for (let h = 0; h <= ngpu; h += run.gpus_per_host) {
    ctx.beginPath(); ctx.moveTo(0, h * rh); ctx.lineTo(W, h * rh); ctx.stroke();
  }
  // 当前步游标
  ctx.strokeStyle = "#4f9dff"; ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(cursor * cw, 0);
  ctx.lineTo(cursor * cw, ngpu * rh); ctx.stroke(); ctx.setLineDash([]);
  // 机器编号
  ctx.fillStyle = "#8595b0"; ctx.font = "10px monospace";
  for (let h = 0; h < ngpu / run.gpus_per_host; h++) {
    ctx.fillText(`host${h}`, 2, h * run.gpus_per_host * rh + rh - 1);
  }
}

function onHeatHover(e) {
  const cv = $("heat");
  const rect = cv.getBoundingClientRect();
  const run = curRun();
  const cw = rect.width / run.steps.length;
  const rh = 8;
  const s = Math.max(0, Math.min(run.steps.length - 1,
    Math.floor((e.clientX - rect.left) / cw)));
  const g = Math.max(0, Math.min(run.num_gpus - 1,
    Math.floor((e.clientY - rect.top) / rh)));
  cursor = s;
  renderAll();
  const tt = $("tooltip");
  const truth = run.truth[s][g] ? TRUTH_CN[run.truth[s][g]] || run.truth[s][g] : "—";
  const cause = CAUSE_CN[run.detected_cause[s][g]] || run.detected_cause[s][g];
  tt.innerHTML = `<b>GPU ${g}</b>（host ${run.hosts[g]}）· step ${s}<br>
    总耗时 ${run.heatmap.total[s][g]}ms ｜ 计算 ${run.heatmap.compute[s][g]} ｜
    通信 ${run.heatmap.comm[s][g]} ｜ 数据 ${run.heatmap.data[s][g]}<br>
    token ${run.heatmap.tokens[s][g]} ｜ 状态 ${run.gpu_state[s][g]}<br>
    真实原因：${truth}<br>
    系统判定：${cause}（置信度 ${run.confidence[s][g]}）`;
  tt.style.display = "block";
  tt.style.left = Math.min(e.clientX + 14, window.innerWidth - 300) + "px";
  tt.style.top = (e.clientY + 14) + "px";
}

// ---------- 慢卡面板 ----------
function renderSlowPanel() {
  const run = curRun();
  const items = [];
  for (let g = 0; g < run.num_gpus; g++) {
    const cause = run.detected_cause[cursor][g];
    if (cause && cause !== "normal") items.push({ g, cause, conf: run.confidence[cursor][g] });
  }
  items.sort((a, b) => b.conf - a.conf);
  if (!items.length) {
    $("slowPanel").innerHTML = `<div class="muted">step ${cursor} 无异常卡</div>`;
    return;
  }
  $("slowPanel").innerHTML = items.slice(0, 20).map((it) => {
    const truth = run.truth[cursor][it.g];
    const tagCls = it.cause.includes("persistent") || it.cause.includes("degrading")
      ? "t-eject" : it.cause.includes("thermal") || it.cause.includes("cluster")
      ? "t-watch" : "t-rebalance";
    return `<div class="gpuitem">
      <div class="top"><b>GPU ${it.gpu}</b>
        <span class="tag ${tagCls}">${CAUSE_CN[it.cause] || it.cause}</span></div>
      <div class="evidence">置信度 ${it.conf} ｜ 真实原因：${truth ? TRUTH_CN[truth] : "无事件"}</div>
    </div>`;
  }).join("");
}

// ---------- 动作记录表 ----------
function renderActionTable() {
  const run = curRun();
  const tb = $("actionTable").querySelector("tbody");
  const rows = run.actions.slice().reverse();
  tb.innerHTML = rows.map((a) => {
    const cls = a.kind.includes("reject") ? "t-rollback"
      : a.kind === "eject" ? "t-eject"
      : a.kind === "rejoin" ? "t-rejoin"
      : a.kind.includes("rollback") ? "t-rollback" : "t-rebalance";
    const status = a.status === "kept" ? "已生效"
      : a.status === "rolled_back" ? "已撤回"
      : a.status === "evaluating" ? "观察中" : a.status;
    return `<tr>
      <td>${a.step}</td>
      <td><span class="tag ${cls}">${a.kind}</span></td>
      <td>${(a.gpus || []).join(",")}</td>
      <td>${CAUSE_CN[a.cause] || a.cause}<br><span class="muted">${(a.confidence || 0).toFixed(2)}</span></td>
      <td>${status}</td></tr>`;
  }).join("");
}

// ---------- 折线图 ----------
function drawLine(canvasId, series, colors, labels, yFmt) {
  const cv = $(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const W = cv.parentElement.clientWidth, H = 180;
  cv.width = W * dpr; cv.height = H * dpr;
  cv.style.height = H + "px";
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
  const all = series.flatMap((s) => s.data);
  const lo = Math.min(...all), hi = Math.max(...all);
  const pad = { l: 46, r: 10, t: 12, b: 22 };
  const x = (i) => pad.l + (i / (series[0].data.length - 1)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo || 1)) * (H - pad.t - pad.b);
  ctx.strokeStyle = "#2a3650"; ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b);
  ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();
  ctx.fillStyle = "#8595b0"; ctx.font = "10px monospace";
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4;
    ctx.fillText(yFmt(v), 2, y(v) + 3);
  }
  series.forEach((s, si) => {
    ctx.strokeStyle = colors[si]; ctx.lineWidth = 1.4; ctx.beginPath();
    s.data.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)));
    ctx.stroke();
  });
  // 当前步游标
  const cx = x(cursor);
  ctx.strokeStyle = "#4f9dff"; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(cx, pad.t); ctx.lineTo(cx, H - pad.b);
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#dce4f2";
  labels.forEach((lb, i) => {
    ctx.fillStyle = colors[i]; ctx.fillRect(pad.l + i * 110, H - 8, 10, 3);
    ctx.fillStyle = "#8595b0"; ctx.fillText(lb, pad.l + i * 110 + 14, H - 5);
  });
}

function renderCharts() {
  const r = DATA.runs;
  drawLine("wallChart",
    [ {data: r.full.wall_ms}, {data: r.observe.wall_ms}, {data: r.clean.wall_ms} ],
    ["#4f9dff", "#f0b429", "#37c98c"], ["full", "observe", "clean"],
    (v) => v.toFixed(0));
  drawLine("waitChart",
    [ {data: r.full.wait_ratio_normalized},
      {data: r.observe.wait_ratio_normalized},
      {data: r.clean.wait_ratio_normalized} ],
    ["#4f9dff", "#f0b429", "#37c98c"], ["full", "observe", "clean"],
    (v) => v.toFixed(2));
  drawLine("lossChart",
    [ {data: r.full.loss}, {data: r.observe.loss}, {data: r.clean.loss} ],
    ["#4f9dff", "#f0b429", "#37c98c"], ["full", "observe", "clean"],
    (v) => v.toFixed(2));
}

// ---------- KPI ----------
function renderKpis() {
  const rep = DATA.report;
  const k = [
    ["墙钟节省", `${rep.wall_total.full_vs_observe_saving_pct}%`,
     "full 相对 observe", rep.wall_total.full_vs_observe_saving_pct >= 0 ? "good" : "bad"],
    ["等待占比降幅", `${(rep.wait_ratio.overall_relative_reduction * 100).toFixed(1)}%`,
     "归一化口径", rep.wait_ratio.overall_relative_reduction >= 0 ? "good" : "bad"],
    ["误剔除", `${rep.ejection_audit.false_eject_count} 张`,
     `正确剔除 ${rep.ejection_audit.true_eject_count} 张`,
     rep.ejection_audit.false_eject_count === 0 ? "good" : "bad"],
    ["最终损失", rep.final_loss.full.toFixed(4),
     `observe ${rep.final_loss.observe.toFixed(4)}`, "accent"],
  ];
  $("kpis").innerHTML = k.map(([l, v, sub, c]) =>
    `<div class="kpi"><div class="v ${c}">${v}</div><div class="l">${l} · ${sub}</div></div>`
  ).join("");
}

// ---------- 事件剧本 ----------
function renderEventLog() {
  $("eventLog").innerHTML = DATA.events.map((e) =>
    `<div class="row"><span class="st">[${e.start}~${e.end < 0 ? "结束" : e.end}]</span>
     <b>${e.id}</b><br><span class="muted">
     ${TRUTH_CN[e.kind] || e.kind} ｜ 卡 ${e.gpus ? e.gpus.join(",") : "全集群"}
     ｜ 分量 ${e.component} ｜ 因子 ${e.factor}${e.period ? " ｜ 周期 " + e.period : ""}
     </span></div>`).join("");
}

// ---------- 时间轴播放 ----------
function togglePlay() {
  playing = !playing;
  $("playBtn").textContent = playing ? "⏸ 暂停" : "▶ 播放时间轴";
  if (playing) {
    if (cursor >= curRun().steps.length - 1) cursor = 0;
    playTimer = setInterval(() => {
      cursor += 20;
      if (cursor >= curRun().steps.length - 1) {
        cursor = curRun().steps.length - 1;
        togglePlay();
      }
      renderAll();
    }, 120);
  } else {
    clearInterval(playTimer);
  }
}

window.addEventListener("resize", () => { if (DATA) renderAll(); });
load().catch((e) => {
  document.body.innerHTML = `<pre style="padding:20px">数据加载失败：${e}\n请先运行 python3 run_experiment.py</pre>`;
});
