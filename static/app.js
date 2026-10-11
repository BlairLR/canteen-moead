/* 交互�?MOEA/D 选址实验 �?前端逻辑 */

(function () {
  "use strict";

  // ===== 状�?=====
  const state = {
    model: null,
    jobId: null,
    lastResult: null,

    pollTimer: null,
    // 交互式地价控制点
    runCompleted: false,    // 最近一次运行是否已完成
    controlPoints: [],      // [{id, lat, lon, x, y, price}] �?x,y 是局部坐�?    cpCounter: 0,
    // 静态热力图缓存（防过多刷新�?    heatmapData: null,
    showHeatmap: true,
    showContours: true,
  };

  // ===== DOM 引用 =====
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const dom = {
    runBtn:           $('#btn-run'),

    statusText:       $('#status-text'),
    statusDot:        $('#status-dot'),
    statusExtra:      $('#status-extra'),

    // 交互式地�?    interactiveSection: $('#interactive-mode-section'),
    ipBasePrice:      $('#ip-base-price'),
    ipSigma:          $('#ip-sigma'),
    ipSigmaVal:       $('#ip-sigma-val'),
    ipPointCount:     $('#ip-point-count'),
    btnIpClear:       $('#btn-ip-clear'),
    btnIpDefaults:    $('#btn-ip-defaults'),
    chkHeatmap:       $('#chk-heatmap'),
    chkContours:      $('#chk-contours'),
  };

  // ===== API 工具 =====
  async function apiGet(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`GET ${url} �?${r.status}`);
    return r.json();
  }

  async function apiPost(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.errors ? data.errors.join('; ') : `POST ${url} �?${r.status}`);
    return data;
  }

  // ===== 提示 =====
  function showError(msg) {
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
  }

  function setStatus(stateText, dotClass, extra) {
    dom.statusText.textContent = stateText;
    dom.statusDot.className = 'status-dot ' + dotClass;
    dom.statusExtra.textContent = extra || '';
  }

  // ===== 交互式地�?�?控制点管�?=====
  const interactiveLayers = {
    markers: L.layerGroup(),
    labelLayer: L.layerGroup(),
    heatmapOverlay: null,  // L.imageOverlay �?canvasOverlay
    contourOverlay: null,
  };

  function cpFromLatLon(lat, lon) {
    return { lat, lon, x: lon, y: lat };
  }

  function latLonToLocal(lat, lon) {
    // �?affine 参数反算局部坐�?    // 正向: lat = a*y + b*x + b_lat, lon = c*y + d*x + b_lon
    // 逆矩�?(A = [[a,b],[c,d]]): A^-1 = 1/det * [[d, -b], [-c, a]]
    const m = state.model;
    if (!m || !m.affine_transform) return { x: lon, y: lat };
    const t = m.affine_transform;
    const det = t.a * t.d - t.b * t.c;
    if (Math.abs(det) < 1e-20) return { x: lon, y: lat };
    const dy = lat - t.b_lat;
    const dx = lon - t.b_lon;
    const y = (t.d * dy - t.b * dx) / det;
    const x = (-t.c * dy + t.a * dx) / det;
    if (!isFinite(x) || !isFinite(y)) return { x: lon, y: lat };
    return { x, y };
  }

  function localToLatLon(x, y) {
    const m = state.model;
    if (!m || !m.affine_transform) return { lat: y, lon: x };
    const t = m.affine_transform;
    const lat = t.a * y + t.b * x + t.b_lat;
    const lon = t.c * y + t.d * x + t.b_lon;
    return { lat, lon };
  }

  function addControlPoint(lat, lon, price) {
    const local = latLonToLocal(lat, lon);
    const id = 'cp-' + (++state.cpCounter);
    const pt = {
      id, lat, lon,
      x: local.x, y: local.y,
      price: (price !== undefined) ? price : parseFloat(dom.ipBasePrice.value),
    };
    state.controlPoints.push(pt);
    createCpMarker(pt);
    updateCpCount();
    refreshHeatmap();
  }

  function createCpMarker(pt) {
    const marker = L.marker([pt.lat, pt.lon], {
      draggable: true,
      icon: L.divIcon({
        className: 'cp-marker',
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      }),
    });
    marker._cpId = pt.id;
    marker.on('dragend', function(e) {
      const pos = e.target.getLatLng();
      const local = latLonToLocal(pos.lat, pos.lng);
      const cp = state.controlPoints.find(c => c.id === pt.id);
      if (cp) {
        cp.lat = pos.lat;
        cp.lon = pos.lng;
        cp.x = local.x;
        cp.y = local.y;
      }
      refreshHeatmap();
    });
    marker.bindPopup(buildCpPopup(pt, marker), { className: 'cp-popup', maxWidth: 220 });
    interactiveLayers.markers.addLayer(marker);

    // 价格标签
    const label = L.divIcon({
      className: 'cp-label',
      html: '¥' + pt.price.toFixed(0),
      iconSize: [0, 0],
      iconAnchor: [0, -10],
    });
    const labelMarker = L.marker([pt.lat, pt.lon], { icon: label, interactive: false });
    labelMarker._cpId = pt.id;
    interactiveLayers.labelLayer.addLayer(labelMarker);
  }

  function buildCpPopup(pt, marker) {
    const div = document.createElement('div');
    div.className = 'cp-popup';
    div.innerHTML = `
      <div class="cp-popup-row">
        <span>价格 (a.u.)</span>
        <input type="number" class="cp-price-input" value="${pt.price}" min="0" max="500" step="5">
        <button class="btn-save">�?/button>
        <button class="btn-delete">�?/button>
      </div>
      <div style="font-size:11px;color:#888;margin-top:3px">
        拖动标记移动位置
      </div>
    `;
    const input = div.querySelector('.cp-price-input');
    div.querySelector('.btn-save').onclick = function() {
      const v = parseFloat(input.value);
      if (isNaN(v) || v < 0 || v > 500) { showError('价格范围�?0-500'); return; }
      pt.price = v;
      updateCpLabels();
      refreshHeatmap();
      marker._popup.close();
    };
    div.querySelector('.btn-delete').onclick = function() {
      removeControlPoint(pt.id);
    };
    return div;
  }

  function updateCpLabels() {
    interactiveLayers.labelLayer.clearLayers();
    state.controlPoints.forEach(pt => {
      const label = L.divIcon({
        className: 'cp-label',
        html: '¥' + pt.price.toFixed(0),
        iconSize: [0, 0],
        iconAnchor: [0, -10],
      });
      const m = L.marker([pt.lat, pt.lon], { icon: label, interactive: false });
      m._cpId = pt.id;
      interactiveLayers.labelLayer.addLayer(m);
    });
  }

  function removeControlPoint(id) {
    state.controlPoints = state.controlPoints.filter(c => c.id !== id);
    interactiveLayers.markers.eachLayer(l => {
      if (l._cpId === id) {
        interactiveLayers.markers.removeLayer(l);
      }
    });
    interactiveLayers.labelLayer.eachLayer(l => {
      if (l._cpId === id) {
        interactiveLayers.labelLayer.removeLayer(l);
      }
    });
    updateCpCount();
    refreshHeatmap();
  }

  function clearAllControlPoints() {
    interactiveLayers.markers.clearLayers();
    interactiveLayers.labelLayer.clearLayers();
    state.controlPoints = [];
    state.cpCounter = 0;
    updateCpCount();
    hideInteractiveOverlays();
  }

  function updateCpCount() {
    dom.ipPointCount.textContent = state.controlPoints.length;
  }

  function getCpApiPayload() {
    return state.controlPoints.map(c => ({
      x: isFinite(c.x) ? c.x : 0,
      y: isFinite(c.y) ? c.y : 0,
      price: c.price,
    }));
  }

  // ===== 热力�?/ 等值线 =====
  function hideInteractiveOverlays() {
    if (interactiveLayers.heatmapOverlay && map.hasLayer(interactiveLayers.heatmapOverlay)) {
      map.removeLayer(interactiveLayers.heatmapOverlay);
    }
    if (interactiveLayers.contourOverlay && map.hasLayer(interactiveLayers.contourOverlay)) {
      map.removeLayer(interactiveLayers.contourOverlay);
    }
  }

  let heatmapDebounce = null;
  function refreshHeatmap() {
    if (!dom.chkHeatmap.checked && !dom.chkContours.checked) {
      hideInteractiveOverlays();
      return;
    }
    if (state.controlPoints.length === 0) {
      hideInteractiveOverlays();
      return;
    }
    if (heatmapDebounce) clearTimeout(heatmapDebounce);
    heatmapDebounce = setTimeout(fetchAndRenderHeatmap, 300);
  }

  async function fetchAndRenderHeatmap() {
    try {
      const data = await apiPost('/api/interactive-pricing/field', {
        control_points: getCpApiPayload(),
        base_price: parseFloat(dom.ipBasePrice.value),
        sigma: parseFloat(dom.ipSigma.value),
        resolution: 50,
      });

      // 验证数据有效�?      if (!data.values || !data.mask || !data.xs || !data.ys) {
        console.warn('Heatmap data missing required fields, skip render');
        hideInteractiveOverlays();
        return;
      }
      const hasInvalid = data.values.some(row =>
        row.some(v => !isFinite(v))
      );
      if (hasInvalid || !isFinite(data.vmin) || !isFinite(data.vmax)) {
        console.warn('Heatmap data contains NaN/Infinity, skip render');
        showError('地价数据异常：控制点坐标无效');
        hideInteractiveOverlays();
        return;
      }

      state.heatmapData = data;
      renderHeatmapOverlay(data);
    } catch (e) {
      console.error('Heatmap fetch error:', e);
    }
  }

  function renderHeatmapOverlay(data) {
    hideInteractiveOverlays();

    const m = state.model;
    if (!m) return;
    const mapImg = m.map_image;
    const sw = mapImg.bounds_sw;
    const ne = mapImg.bounds_ne;
    const bounds = L.latLngBounds([sw[0], sw[1]], [ne[0], ne[1]]);

    // 后端 values[j][i] = 坐标 (xs[i], ys[j]) 处的价格
    // j=0 �?ys[0] = �? j=ny-1 �?ys[-1] = �?    // Canvas row 0 = 图片顶部 = 北，所以需要翻�?Y �?    const nx = data.xs.length;   // 东西方向
    const ny = data.ys.length;   // 南北方向
    const canvas = document.createElement('canvas');
    canvas.width = nx;
    canvas.height = ny;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(nx, ny);

    const vmin = data.vmin;
    const vmax = data.vmax;
    const range = (vmax - vmin) || 1;

    // 检�?mask/values 的维度：后端返回的是 2D 数组
    const maskIs2D = Array.isArray(data.mask) && Array.isArray(data.mask[0]);
    const valIs2D = Array.isArray(data.values) && Array.isArray(data.values[0]);

    for (let j = 0; j < ny; j++) {
      // canvas row j = 图片�?j 行（从上到下），对应后端�?(ny-1-j) 行（从北到南�?      const srcRow = ny - 1 - j;
      for (let i = 0; i < nx; i++) {
        let inside, val;
        if (maskIs2D && valIs2D) {
          inside = !!data.mask[srcRow][i];
          val = inside ? data.values[srcRow][i] : 0;
        } else {
          // 兼容旧版扁平格式（万一后端改了接口�?          const idx = srcRow * nx + i;
          inside = !!data.mask[idx];
          val = inside ? data.values[idx] : 0;
        }
        const norm = inside ? (val - vmin) / range : 0;
        const pi = (j * nx + i) * 4;

        // 色图: 深蓝→蓝→绿→黄→红 (inferno-like)
        let r, g, b;
        if (!inside) {
          r = g = b = 0;
          imgData.data[pi + 3] = 0;
        } else if (norm < 0.25) {
          const t = norm / 0.25;
          r = 44 + (41 - 44) * t;   // #2c3e50 �?#2980b9
          g = 62 + (128 - 62) * t;
          b = 80 + (185 - 80) * t;
          imgData.data[pi + 3] = 180;
        } else if (norm < 0.5) {
          const t = (norm - 0.25) / 0.25;
          r = 41 + (39 - 41) * t;   // #2980b9 �?#27ae60
          g = 128 + (174 - 128) * t;
          b = 185 + (96 - 185) * t;
          imgData.data[pi + 3] = 200;
        } else if (norm < 0.75) {
          const t = (norm - 0.5) / 0.25;
          r = 39 + (241 - 39) * t;  // #27ae60 �?#f1c40f
          g = 174 + (196 - 174) * t;
          b = 96 + (15 - 96) * t;
          imgData.data[pi + 3] = 210;
        } else {
          const t = (norm - 0.75) / 0.25;
          r = 241 + (231 - 241) * t; // #f1c40f �?#e74c3c
          g = 196 + (76 - 196) * t;
          b = 15 + (60 - 15) * t;
          imgData.data[pi + 3] = 220;
        }
        imgData.data[pi] = r;
        imgData.data[pi + 1] = g;
        imgData.data[pi + 2] = b;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    const dataUrl = canvas.toDataURL('image/png');

    if (dom.chkHeatmap.checked) {
      interactiveLayers.heatmapOverlay = L.imageOverlay(dataUrl, bounds, {
        opacity: 0.65,
        interactive: false,
      });
      map.addLayer(interactiveLayers.heatmapOverlay);
    }

    // 等值线
    if (dom.chkContours.checked && data.contours && data.contours.length > 0) {
      const contourLines = [];
      data.contours.forEach((contour, ci) => {
        // 后端返回 {level, points} 字典，points �?[[x,y], ...]
        const pts = contour && contour.points;
        if (!pts || pts.length < 3) return;
        const level = contour.level;
        const wgsPts = pts.map(p => {
          const ll = localToLatLon(p[0], p[1]);
          return [ll.lat, ll.lon];
        });
        contourLines.push(L.polyline(wgsPts, {
          color: '#e74c3c',
          weight: 1.5,
          opacity: 0.5 + 0.1 * (ci / data.contours.length),
          dashArray: '4,4',
          interactive: false,
        }));
      });
      if (contourLines.length > 0) {
        interactiveLayers.contourOverlay = L.layerGroup(contourLines);
        map.addLayer(interactiveLayers.contourOverlay);
      }
    }
  }

  function onMapClick(e) {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;
    addControlPoint(lat, lon);
  }

  // ===== 交互层管理（�?map 初始化后调用�?=====
  function initInteractiveLayers() {
    if (!map) return;
    map.addLayer(interactiveLayers.markers);
    map.addLayer(interactiveLayers.labelLayer);

    // 左键点击添加控制�?    map.on('click', onMapClick);

    // 右键查询地价
    map.on('contextmenu', async function(e) {
      if (state.controlPoints.length === 0) return;
      const local = latLonToLocal(e.latlng.lat, e.latlng.lng);
      try {
        const data = await apiPost('/api/interactive-pricing/price-at', {
          x: local.x, y: local.y,
          control_points: getCpApiPayload(),
          base_price: parseFloat(dom.ipBasePrice.value),
          sigma: parseFloat(dom.ipSigma.value),
        });
        L.popup()
          .setLatLng(e.latlng)
          .setContent(`<b>地价: ¥${data.price.toFixed(1)}</b><br><small>基础: ${data.base_price}</small>`)
          .openOn(map);
      } catch (ex) {
        console.error(ex);
      }
    });
  }
  // ===== 地图 =====
  let map = null;
  let layerPolygon = null;
  let layerPoints = null;
  let layerCandidates = null;
  let layerCompCandidates = null;

  function initMap(model) {
    const mapImg = model.map_image;
    const sw = mapImg.bounds_sw;   // [lat, lon]
    const ne = mapImg.bounds_ne;   // [lat, lon]

    map = L.map('map', {
      crs: L.CRS.Simple,
      minZoom: -1,
      maxZoom: 18,
      attributionControl: false,
      zoomControl: true,
    });
    map.setView([(sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2], 0);

    // Campus map background
    L.imageOverlay(mapImg.url, [[sw[0], sw[1]], [ne[0], ne[1]]], {
      opacity: 0.85,
    }).addTo(map);

    // A3 polygon
    layerPolygon = L.polygon(model.polygon_wgs84.map(p => [p[0], p[1]]), {
      color: '#e74c3c',
      weight: 2,
      fillColor: '#e74c3c',
      fillOpacity: 0.06,
      interactive: false,
    }).addTo(map);
    map.fitBounds(layerPolygon.getBounds().pad(0.1));

    // 模型�?    layerPoints = L.layerGroup().addTo(map);
    model.points.forEach(pt => {
      const icon = L.divIcon({
        className: 'model-marker',
        html: `<div class="model-dot ${pt.type}">
                <span class="model-label">${pt.name || pt.id}</span>
               </div>`,
        iconSize: [0, 0],
        iconAnchor: [0, 0],
      });
      const marker = L.marker([pt.lat, pt.lon], { icon }).addTo(layerPoints);
      marker.bindPopup(
        `<b>${pt.name || pt.id} (${pt.id})</b><br>` +
        `类型: ${({competition:'现有食堂', dorm:'宿舍', teaching:'教学�?})[pt.type] || pt.type}<br>` +
        `坐标: ${pt.lat.toFixed(5)}, ${pt.lon.toFixed(5)}`
      );
    });

    // 初始化交互层（控制点、右键查询）
    initInteractiveLayers();

    // === DEBUG: 地图状态输出到控制�?===
    const mapImg2 = model.map_image;
    const bnd = map.getBounds();
    const createdMarkers = document.querySelectorAll('.leaflet-marker-icon').length;
    console.log('=== 校园地图调试信息 ===');
    console.log('底图 URL:', mapImg2.url);
    console.log('底图 bounds SW:', mapImg2.bounds_sw, 'NE:', mapImg2.bounds_ne);
    console.log('地图当前 zoom:', map.getZoom(), 'center:', map.getCenter());
    console.log('地图当前可视范围:', bnd.toBBoxString());
    console.log('多边形顶点数:', model.polygon_wgs84.length);
    console.log('多边�?bounds:', layerPolygon.getBounds().toBBoxString());
    console.log('模型点总数:', model.points.length);
    console.log('DOM 标记元素�?', createdMarkers);
    const types = {};
    model.points.forEach(p => { types[p.type] = (types[p.type] || 0) + 1; });
    console.log('点位类型分布:', JSON.stringify(types));
    console.log('=== 调试信息结束 ===');
  }

  // ===== 候选解 =====
  function showCandidates(data) {
    if (!map) return;
    // Clean old
    if (layerCandidates) map.removeLayer(layerCandidates);
    if (layerCompCandidates) map.removeLayer(layerCompCandidates);

    if (!data || !data.pareto || data.pareto.length === 0) return;

    // Extract pareto from result
    let pareto = data.pareto;
    if (Array.isArray(data.result) && data.result.length > 0 && data.result[0].pareto) {
      pareto = data.result[0].pareto;
    } else if (data.result && data.result.pareto) {
      pareto = data.result.pareto;
    }

    layerCandidates = L.layerGroup().addTo(map);

    pareto.forEach((pt, i) => {
      const icon = L.divIcon({
        className: 'candidate-marker',
        html: `<div class="candidate-dot"></div>`,
        iconSize: [8, 8],
        iconAnchor: [4, 4],
      });
      const marker = L.marker([pt.lat, pt.lon], { icon });
      marker.bindPopup(
        `<b>候选解 #${i + 1}</b><br>
         f1 (avg dist) = ${pt.f1.toFixed(2)} m<br>
         f2 (comp avoid) = ${(pt.f2 * 100).toFixed(1)}%<br>
         f3 (land cost) = ¥${pt.f3.toFixed(1)}<br>
         <small>局部坐�? (${pt.x_local.toFixed(1)}, ${pt.y_local.toFixed(1)})</small>`,
        { maxWidth: 300 }
      );
      layerCandidates.addLayer(marker);

      // Comparison markers for experiment B
      if (data.extraPareto) {
        if (!layerCompCandidates) layerCompCandidates = L.layerGroup().addTo(map);
        data.extraPareto.forEach((pt2, j) => {
          const icon2 = L.divIcon({
            className: 'cand-comp-marker',
            html: `<div class="cand-comp-dot"></div>`,
            iconSize: [8, 8],
            iconAnchor: [4, 4],
          });
          const m2 = L.marker([pt2.lat, pt2.lon], { icon: icon2 });
          m2.bindPopup(`<b>实验 B 候选解 #${j+1}</b><br>` +
            `f1=${pt2.f1.toFixed(2)} f2=${(pt2.f2*100).toFixed(1)}% f3=¥${pt2.f3.toFixed(1)}`);
          layerCompCandidates.addLayer(m2);
        });
      }
    });
  }

  // ===== 运行优化 =====
  async function runOptimization() {
    if (state.jobId) {
      showError('已有任务正在运行，请等待完成');
      return;
    }

    // 重置状态：清除上次结果，恢复初始占�?    state.runCompleted = false;
    showParetoPlaceholder('计算中�?);

    dom.runBtn.disabled = true;
    dom.runBtn.textContent = '运行�?..';

    try {
      if (state.controlPoints.length === 0) {
        showError('请至少添加一个价格控制点');
        dom.runBtn.disabled = false;
        dom.runBtn.textContent = '�?运行 MOEA/D';
        return;
      }
      const result = await apiPost('/api/interactive-pricing/run', {
        control_points: getCpApiPayload(),
        base_price: parseFloat(dom.ipBasePrice.value),
        sigma: parseFloat(dom.ipSigma.value),
      });

      state.jobId = result.job_id;
      setStatus('运行中�?, 'running');
      startPolling(result.job_id);
    } catch (e) {
      showError('启动失败: ' + e.message);
      dom.runBtn.disabled = false;
      dom.runBtn.textContent = '�?运行 MOEA/D';
    }
  }

  function startPolling(jobId) {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(() => pollJob(jobId), 1000);
    pollJob(jobId);
  }

  async function pollJob(jobId) {
    try {
      const data = await apiGet('/api/status/' + jobId);
      if (data.status === 'running') {
        setStatus('运行中�?, 'running');
      } else if (data.status === 'done') {
        if (!data.result) {
          // result still being computed, keep polling
          return;
        }
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        state.jobId = null;
        state.lastResult = data.result;

        const r = data.result;
        // 尝试�?experiment_results.json 或结果对象中�?        let hv = r.hv !== undefined ? r.hv : (r.summary ? r.summary.hv.mean : null);
        setStatus('完成 �?, 'done',
          `HV=${hv !== null ? hv.toFixed(4) : '?'}  ${(r.time_s || r.time || 0).toFixed(1)}s`);

        state.runCompleted = true;

        // 显示候选解（独�?try-catch，避免阻断图表绘制）
        try { showCandidates(r); } catch (e) { console.error('候选解展示失败:', e); }

        // 绘制 Pareto 前沿（独�?try-catch，确保按钮状态更新）
        try {
          drawParetoChart(r);
        } catch (e) {
          console.error('Pareto 图表绘制失败:', e);
          drawCanvasPlaceholder('图表绘制失败: ' + e.message);
        }

        // 显示静态图表下载链�?        try { showChartLinks(r); } catch (e) { console.error('图表链接展示失败:', e); }

        dom.runBtn.disabled = false;
        dom.runBtn.textContent = '�?运行 MOEA/D';
      } else if (data.status === 'error') {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        state.jobId = null;
        setStatus('错误', 'error', data.error);
        showError(data.error);
        dom.runBtn.disabled = false;
        dom.runBtn.textContent = '�?运行 MOEA/D';
      }
    } catch (e) {
      console.error('Poll error:', e);
    }
  }

  // ===== Pareto 前沿图（3D + 2D 回退�?=====

  /** 2D Canvas 回退 �?占位文字 */
  function drawCanvasPlaceholder(msg) {
    const canvas = document.getElementById('pareto-chart-3d');
    if (!canvas) return;
    if (typeof canvas.innerHTML !== 'undefined') {
      canvas.innerHTML = '<div class="pareto-3d-placeholder">' +
        (msg || '运行 MOEA/D 后将在此显示 Pareto 前沿') + '</div>';
    }
  }

  /** 首�?3D 渲染，不可用时回退�?2D Canvas */
  function showParetoPlaceholder(msg) {
    if (window.__showPareto3DPlaceholder) {
      window.__showPareto3DPlaceholder(msg);
    } else {
      drawCanvasPlaceholder(msg);
    }
  }

  function drawParetoChart(result) {
    if (!result) return;
    // 提取 pareto（支持嵌套路径）
    let pareto = result.pareto;
    if ((!pareto || pareto.length === 0) && result.result) {
      if (Array.isArray(result.result) && result.result.length > 0 && result.result[0].pareto) {
        pareto = result.result[0].pareto;
      } else if (result.result.pareto) {
        pareto = result.result.pareto;
      }
    }
    if (!pareto || pareto.length === 0) {
      showParetoPlaceholder(
        state.runCompleted ? 'MOEA/D 已完成但未获�?Pareto �? : '运行 MOEA/D 后将在此显示 Pareto 前沿');
      return;
    }

    // 检查是否有有效数据
    const has3D = typeof window.__drawParetoChart3D === 'function';
    if (has3D) {
      try {
        window.__drawParetoChart3D(result);
      } catch (e) {
        console.error('3D Pareto 渲染失败，回退�?2D:', e);
        drawParetoChart2D(result);
      }
    } else {
      // 2D 回退：至少需�?f1 + f3
      drawParetoChart2D(result);
    }
  }

  /** 2D Canvas 回退 �?二维散点图（�?f1 vs f3�?*/
  function drawParetoChart2D(result) {
    let pareto = result.pareto;
    if ((!pareto || pareto.length === 0) && result.result) {
      if (Array.isArray(result.result) && result.result.length > 0 && result.result[0].pareto) {
        pareto = result.result[0].pareto;
      } else if (result.result.pareto) {
        pareto = result.result.pareto;
      }
    }
    if (!pareto || pareto.length === 0) {
      drawCanvasPlaceholder(state.runCompleted ? 'MOEA/D 已完成但未获�?Pareto �? : '运行 MOEA/D 后将在此显示 Pareto 前沿');
      return;
    }

    const valid = pareto.filter(p =>
      p && typeof p.f1 === 'number' && isFinite(p.f1) &&
      typeof p.f3 === 'number' && isFinite(p.f3)
    );
    if (valid.length === 0) {
      drawCanvasPlaceholder('Pareto 数据全部无效');
      return;
    }

    const container = document.getElementById('pareto-chart-3d');
    if (!container) return;
    container.innerHTML = '';

    const canvas = document.createElement('canvas');
    canvas.style.display = 'block';
    canvas.style.width = '100%';
    canvas.style.background = '#f0f2f5';
    container.appendChild(canvas);

    // 尺寸
    const W = container.clientWidth || 356;
    const H = Math.max(300, W * 0.6);
    canvas.width = W * 2;
    canvas.height = H * 2;
    canvas.style.height = H + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(2, 2);

    try {
      ctx.clearRect(0, 0, W, H);
      const xs = valid.map(p => p.f1);
      const ys = valid.map(p => p.f3);
      const xMin = Math.min(...xs) * 0.95;
      const xMax = Math.max(...xs) * 1.05;
      const yMin = Math.min(...ys) * 0.95;
      const yMax = Math.max(...ys) * 1.05;

      const xRange = xMax - xMin;
      const yRange = yMax - yMin;
      if (xRange < 1e-12 || yRange < 1e-12) {
        drawCanvasPlaceholder('Pareto 解范围过�?);
        return;
      }

      const pad = { l: 65, r: 20, t: 20, b: 35 };
      const plotW = W - pad.l - pad.r;
      const plotH = H - pad.t - pad.b;

      function sx(v) { return pad.l + (v - xMin) / xRange * plotW; }
      function sy(v) { return pad.t + plotH - (v - yMin) / yRange * plotH; }

      ctx.fillStyle = '#f8f9fa';
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = '#e0e0e0';
      ctx.lineWidth = 0.5;
      for (let i = 0; i <= 5; i++) {
        const x = pad.l + (i / 5) * plotW;
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + plotH); ctx.stroke();
      }
      for (let i = 0; i <= 5; i++) {
        const y = pad.t + (i / 5) * plotH;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + plotW, y); ctx.stroke();
      }

      const hue1 = 220, hue2 = 20;
      valid.forEach(p => {
        const px = sx(p.f1);
        const py = sy(p.f3);
        if (!isFinite(px) || !isFinite(py)) return;
        const t = (px - pad.l) / plotW;
        const r = 2 + t * 2;
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = `hsl(${hue1 + (hue2 - hue1) * t}, 85%, 50%)`;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      ctx.fillStyle = '#333';
      ctx.font = '13px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('f1 �?平均距离 (m)', pad.l + plotW / 2, H - 4);

      ctx.save();
      ctx.translate(12, pad.t + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText('f3 �?土地成本 (a.u.)', 0, 0);
      ctx.restore();

      ctx.fillStyle = '#888';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      for (let i = 0; i <= 5; i++) {
        const v = xMin + (i / 5) * (xMax - xMin);
        ctx.fillText(v.toFixed(1), pad.l + (i / 5) * plotW, pad.t + plotH + 15);
      }
      ctx.textAlign = 'right';
      for (let i = 0; i <= 5; i++) {
        const v = yMin + (i / 5) * (yMax - yMin);
        ctx.fillText(v.toFixed(1), pad.l - 5, pad.t + plotH - (i / 5) * plotH + 4);
      }

      ctx.fillStyle = '#999';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'left';
      const n = valid.length;
      const hvVal = result.hv !== undefined ? result.hv : '?';
      ctx.fillText(`Pareto �? ${n} �? |  HV: ${typeof hvVal === 'number' ? hvVal.toFixed(4) : hvVal}`, pad.l, pad.t - 6);
    } catch (e) {
      console.error('2D Pareto chart error:', e);
      drawCanvasPlaceholder('绘图错误，请查看控制�?);
    }
  }

  // ===== 静态图表下载链�?=====
  function showChartLinks(result) {
    const charts = result.charts;
    if (!charts) return;

    // 移除旧链�?    const old = document.getElementById('chart-links');
    if (old) old.remove();

    const chartNames = [
      { key: 'hv_convergence',  label: 'HV 收敛曲线' },
      { key: 'pareto_f1f2',     label: 'Pareto f1-f2' },
      { key: 'a3_boundary',     label: 'A3 边界 & 21 �? },
      { key: 'decision_space',  label: '决策空间分布' },
      { key: 'pareto_3d',       label: 'Pareto 3D 散点�? },
      { key: 'canteen_schemes', label: '候选方案对�? },
      { key: 'land_cost_map',   label: '地价空间分布' },
    ];

    const container = document.createElement('div');
    container.id = 'chart-links';
    container.className = 'panel-section';
    container.innerHTML = '<h3>📥 下载静态图�?/h3>';

    const list = document.createElement('div');
    list.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;margin-top:8px';

    chartNames.forEach(c => {
      const url = charts[c.key];
      if (!url) return;
      const a = document.createElement('a');
      a.href = url;
      a.target = '_blank';
      a.textContent = c.label;
      a.className = 'chart-dl-link';
      a.style.cssText = 'font-size:11px;padding:4px 10px;' +
        'background:#f0f4f8;border-radius:4px;text-decoration:none;' +
        'color:#2c3e50;border:1px solid #dce4ec;cursor:pointer';
      a.addEventListener('mouseenter', () => { a.style.background = '#dce4ec'; });
      a.addEventListener('mouseleave', () => { a.style.background = '#f0f4f8'; });
      list.appendChild(a);
    });

    container.appendChild(list);

    // 插入到左侧面板底�?    const leftPanel = document.querySelector('.left-panel');
    if (leftPanel) leftPanel.appendChild(container);
  }

  // ===== 事件绑定 =====
    function bindEvents() {
        // 运行
        dom.runBtn.addEventListener('click', runOptimization);

        // 交互式地价控�?        dom.ipSigma.addEventListener('input', () => {
            dom.ipSigmaVal.textContent = dom.ipSigma.value;
            refreshHeatmap();
        });
        dom.ipBasePrice.addEventListener('change', () => {
            refreshHeatmap();
        });
        dom.btnIpClear.addEventListener('click', clearAllControlPoints);
        dom.chkHeatmap.addEventListener('change', () => {
            state.showHeatmap = dom.chkHeatmap.checked;
            if (state.showHeatmap && state.heatmapData) {
                renderHeatmapOverlay(state.heatmapData);
            } else {
                if (interactiveLayers.heatmapOverlay && map) map.removeLayer(interactiveLayers.heatmapOverlay);
            }
        });
        dom.chkContours.addEventListener('change', () => {
            state.showContours = dom.chkContours.checked;
            if (state.showContours && state.heatmapData) {
                renderHeatmapOverlay(state.heatmapData);
            } else {
                if (interactiveLayers.contourOverlay && map) map.removeLayer(interactiveLayers.contourOverlay);
            }
        });
    }

    // ===== 启动 =====
    async function init() {
        try {
            const model = await apiGet('/api/model');
            state.model = model;
            initMap(model);
            showParetoPlaceholder();
      bindEvents();
      setStatus('就绪 �?, 'idle');

      // �?latLonToLocal/localToLatLon 注册�?window 以便控制点管�?      window.__latLonToLocal = latLonToLocal;
      window.__localToLatLon = localToLatLon;

    } catch (e) {
      showError('初始化失�? ' + e.message);
      setStatus('初始化失�?, 'error', e.message);
    }
  }

  document.addEventListener('DOMContentLoaded', init);

  // ===== 更新控制点标注（从popup回调调用，所以在函数前声明） =====
  // updateCpLabels 已在前面定义

  // 注意: onMapClick �?map.on('click') 需要在 map 就绪后设�?  // map.on('contextmenu', ...) 也依�?map 就绪
  // 这些都在 setMode 中动态绑�?
})();
