/* 三维 Pareto 散点图 — Three.js 实现
   用于 canteen-moead 交互式 MOEA/D 实验
   在 index.html 中加载，依赖 THREE (r128) 和 THREE.OrbitControls
*/
(function () {
  "use strict";

  // ===== 内部状态 =====
  var scene = null, camera = null, renderer = null, controls = null;
  var animationId = null;
  var tooltipEl = null, infoEl = null;
  var raycastObjects = [];
  var hoveredObject = null;
  var resizeObserver = null;
  var currentResult = null;   // 保留给 resize 重绘用
  var active = false;         // true 表示正在显示 3D 场景（非占位）
  var CONTAINER_ID = 'pareto-chart-3d';

  // ===== 尺寸 =====
  function getContainerSize() {
    var el = document.getElementById(CONTAINER_ID);
    if (!el) return { w: 356, h: 300 };
    var w = el.clientWidth || 356;
    var h = Math.max(300, w * 0.6);
    // 面板 content width = 380 - 24 = 356
    return { w: w, h: h };
  }

  // ===== 清理 =====
  function cleanup() {
    if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
    if (resizeObserver) { resizeObserver.disconnect(); resizeObserver = null; }
    if (renderer) {
      renderer.dispose();
      if (renderer.domElement && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    }
    if (tooltipEl && tooltipEl.parentNode) {
      tooltipEl.parentNode.removeChild(tooltipEl); tooltipEl = null;
    }
    if (infoEl && infoEl.parentNode) {
      infoEl.parentNode.removeChild(infoEl); infoEl = null;
    }
    if (controls) { controls.dispose(); controls = null; }
    scene = null; camera = null; renderer = null;
    raycastObjects = []; hoveredObject = null;
    active = false;

    // 清理容器内残留的 info overlay（如果在 cleanup 前被外部 innerHTML 清除了）
    var container = document.getElementById(CONTAINER_ID);
    if (container) {
      var leftovers = container.querySelectorAll('.pareto-3d-info');
      for (var i = 0; i < leftovers.length; i++) leftovers[i].remove();
    }
  }

  // ===== Sprite 标签 =====
  function createTextSprite(text, opts) {
    opts = opts || {};
    var color = opts.color || '#333';
    var bg = opts.bg || 'rgba(255,255,255,0.85)';
    var fontSize = opts.fontSize || 40;
    var padding = opts.padding || 14;

    var meas = document.createElement('canvas').getContext('2d');
    meas.font = 'bold ' + fontSize + 'px sans-serif';
    var tw = Math.ceil(meas.measureText(text).width);
    var cw = tw + padding * 2;
    var ch = Math.ceil(fontSize * 1.5 + padding * 2);

    var canvas = document.createElement('canvas');
    canvas.width = cw;
    canvas.height = ch;
    var ctx = canvas.getContext('2d');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, cw, ch);
    ctx.fillStyle = color;
    ctx.font = 'bold ' + fontSize + 'px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, cw / 2, ch / 2);

    var texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    var mat = new THREE.SpriteMaterial({
      map: texture, transparent: true,
      depthTest: false, depthWrite: false
    });
    var sprite = new THREE.Sprite(mat);
    var aspect = cw / ch;
    sprite.scale.set(aspect * 0.5, 0.5, 1);
    return sprite;
  }

  function makeAxisLine(fromArr, toArr, color) {
    var mat = new THREE.LineBasicMaterial({ color: color });
    var pts = [
      new THREE.Vector3(fromArr[0], fromArr[1], fromArr[2]),
      new THREE.Vector3(toArr[0], toArr[1], toArr[2])
    ];
    var geo = new THREE.BufferGeometry().setFromPoints(pts);
    return new THREE.Line(geo, mat);
  }

  function makeArrowHead(posArr, dirArr, color) {
    var cone = new THREE.ConeGeometry(0.10, 0.28, 10);
    var mat = new THREE.MeshBasicMaterial({ color: color });
    var mesh = new THREE.Mesh(cone, mat);
    mesh.position.set(posArr[0], posArr[1], posArr[2]);
    var up = new THREE.Vector3(0, 1, 0);
    var d = new THREE.Vector3(dirArr[0], dirArr[1], dirArr[2]).normalize();
    var q = new THREE.Quaternion().setFromUnitVectors(up, d);
    mesh.quaternion.copy(q);
    return mesh;
  }

  // ===== 重新绘制（Resize 时调用） =====
  function redraw() {
    if (currentResult) {
      drawScene(currentResult);
    }
  }

  // ===== 占位/空状态 =====
  window.__showPareto3DPlaceholder = function (msg) {
    cleanup();
    currentResult = null;
    var el = document.getElementById(CONTAINER_ID);
    if (!el) return;
    el.innerHTML = '<div class="pareto-3d-placeholder">' +
      (msg || '运行 MOEA/D 后将在此显示 Pareto 前沿') + '</div>';
  };

  // ===== 主绘图函数 =====
  function drawScene(result) {
    // 1. 提取 Pareto 数据
    var pareto = result && result.pareto;
    if ((!pareto || pareto.length === 0) && result && result.result) {
      if (Array.isArray(result.result) && result.result.length > 0 && result.result[0].pareto) {
        pareto = result.result[0].pareto;
      } else if (result.result.pareto) {
        pareto = result.result.pareto;
      }
    }
    if (!pareto || pareto.length === 0) {
      window.__showPareto3DPlaceholder('无 Pareto 数据');
      return;
    }

    // 2. 过滤有效点
    var valid = pareto.filter(function (p) {
      return p && typeof p.f1 === 'number' && isFinite(p.f1) &&
             typeof p.f2 === 'number' && isFinite(p.f2) &&
             typeof p.f3 === 'number' && isFinite(p.f3);
    });
    if (valid.length === 0) {
      window.__showPareto3DPlaceholder('Pareto 数据全部无效');
      return;
    }

    // 3. 清理旧场景
    cleanup();

    var container = document.getElementById(CONTAINER_ID);
    if (!container) return;
    container.innerHTML = '';

    var size = getContainerSize();
    var W = size.w, H = size.h;
    if (W < 50 || H < 50) return;

    // 4. 统计 & 归一化
    var f1s = valid.map(function (p) { return p.f1; });
    var f2s = valid.map(function (p) { return p.f2; });
    var f3s = valid.map(function (p) { return p.f3; });
    var f1Min = Math.min.apply(null, f1s), f1Max = Math.max.apply(null, f1s);
    var f2Min = Math.min.apply(null, f2s), f2Max = Math.max.apply(null, f2s);
    var f3Min = Math.min.apply(null, f3s), f3Max = Math.max.apply(null, f3s);

    function norm(v, min, max) {
      if (max - min < 1e-12) return 0;
      return ((v - min) / (max - min)) * 4 - 2;  // -> [-2, 2]
    }

    // ===== 5. 场景 =====
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f6fa);

    camera = new THREE.PerspectiveCamera(38, W / H, 0.01, 100);
    camera.position.set(5.5, 3.8, 6.5);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.rotateSpeed = 1.0;
    controls.zoomSpeed = 1.2;
    controls.target.set(0, 0, 0);
    controls.update();

    // 灯光
    scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    var dl1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dl1.position.set(1, 2.5, 1.5);
    scene.add(dl1);
    var dl2 = new THREE.DirectionalLight(0xffffff, 0.35);
    dl2.position.set(-1, -0.5, -1.5);
    scene.add(dl2);

    // ===== 6. 坐标轴 =====
    var O = -2.0;   // 三个轴的原点（最小值端）
    var AXIS_END = 2.5;

    // f1 (X) — 红色
    scene.add(makeAxisLine([O, O, O], [AXIS_END, O, O], 0xc0392b));
    scene.add(makeArrowHead([AXIS_END, O, O], [1, 0, 0], 0xc0392b));
    // f3 (Y) — 绿色
    scene.add(makeAxisLine([O, O, O], [O, AXIS_END, O], 0x27ae60));
    scene.add(makeArrowHead([O, AXIS_END, O], [0, 1, 0], 0x27ae60));
    // f2 (Z) — 蓝色
    scene.add(makeAxisLine([O, O, O], [O, O, AXIS_END], 0x2980b9));
    scene.add(makeArrowHead([O, O, AXIS_END], [0, 0, 1], 0x2980b9));

    // 轴标签
    var labelDefs = [
      { text: 'f1 —— 平均距离', pos: [AXIS_END + 0.6, O - 0.15, O], color: '#c0392b', bg: 'rgba(255,255,255,0.8)' },
      { text: 'f3 —— 土地成本', pos: [O - 0.15, AXIS_END + 0.6, O], color: '#27ae60', bg: 'rgba(255,255,255,0.8)' },
      { text: 'f2 —— 重叠度', pos: [O - 0.15, O - 0.15, AXIS_END + 0.6], color: '#2980b9', bg: 'rgba(255,255,255,0.8)' },
    ];
    labelDefs.forEach(function (lbl) {
      var s = createTextSprite(lbl.text, { color: lbl.color, bg: lbl.bg, fontSize: 34 });
      s.position.set(lbl.pos[0], lbl.pos[1], lbl.pos[2]);
      scene.add(s);
    });

    // 轴范围标签（最小值/最大值）— 各轴错开位置避免重叠
    var rangeLabels = [
      // f1 沿 X：y 方向偏移
      { text: f1Min.toFixed(1), pos: [O, O - 0.3, O - 0.1] },
      { text: f1Max.toFixed(1), pos: [AXIS_END, O - 0.3, O - 0.1] },
      // f3 沿 Y：x 方向偏移
      { text: f3Min.toFixed(0), pos: [O - 0.3, O, O - 0.1] },
      { text: f3Max.toFixed(0), pos: [O - 0.3, AXIS_END, O - 0.1] },
      // f2 沿 Z：x 方向偏移（与 f3 不同深）
      { text: (f2Min * 100).toFixed(1) + '%', pos: [O - 0.2, O - 0.3, O] },
      { text: (f2Max * 100).toFixed(1) + '%', pos: [O - 0.2, O - 0.3, AXIS_END] },
    ];
    rangeLabels.forEach(function (lbl) {
      var s = createTextSprite(lbl.text, { color: '#777', bg: 'transparent', fontSize: 22 });
      s.position.set(lbl.pos[0], lbl.pos[1], lbl.pos[2]);
      scene.add(s);
    });

    // ===== 7. Pareto 解球体 =====
    raycastObjects = [];
    var sphereGeo = new THREE.SphereGeometry(0.065, 10, 10);

    valid.forEach(function (p) {
      var x = norm(p.f1, f1Min, f1Max);
      var y = norm(p.f3, f3Min, f3Max);
      var z = norm(p.f2, f2Min, f2Max);

      // f3 着色: 低 → 蓝(0.60), 高 → 红(0.00)
      var t = (p.f3 - f3Min) / (f3Max - f3Min + 1e-12);
      var hue = Math.max(0, Math.min(0.65, 0.58 - t * 0.48));
      var color = new THREE.Color().setHSL(hue, 0.80, 0.50);

      var mat = new THREE.MeshPhongMaterial({
        color: color,
        specular: 0x222222,
        shininess: 25
      });
      var mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.position.set(x, y, z);
      mesh.userData = {
        f1: p.f1, f2: p.f2, f3: p.f3,
        lat: p.lat, lon: p.lon,
        x_local: p.x_local, y_local: p.y_local
      };
      scene.add(mesh);
      raycastObjects.push(mesh);
    });

    // ===== 8. 悬停提示 (tooltip) =====
    tooltipEl = document.createElement('div');
    tooltipEl.className = 'pareto-3d-tooltip';
    tooltipEl.style.display = 'none';
    document.body.appendChild(tooltipEl);

    var raycaster = new THREE.Raycaster();
    var mouse = new THREE.Vector2();

    function onMouseMove(e) {
      var rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      var intersects = raycaster.intersectObjects(raycastObjects);

      if (hoveredObject) {
        if (hoveredObject.material.emissive) {
          hoveredObject.material.emissive.setHex(0x000000);
        }
        hoveredObject = null;
      }

      if (intersects.length > 0) {
        var obj = intersects[0].object;
        if (obj.material.emissive) {
          obj.material.emissive.setHex(0x444444);
        }
        hoveredObject = obj;

        var d = obj.userData;
        tooltipEl.style.display = 'block';
        tooltipEl.style.left = Math.min(e.clientX + 15, window.innerWidth - 220) + 'px';
        tooltipEl.style.top = Math.max(e.clientY - 10, 10) + 'px';
        tooltipEl.innerHTML =
          '<b>候选解</b><br>' +
          'f1 (平均距离): ' + d.f1.toFixed(2) + ' m<br>' +
          'f2 (重叠度): ' + (d.f2 * 100).toFixed(1) + '%<br>' +
          'f3 (土地成本): ' + d.f3.toFixed(1) + ' a.u.<br>' +
          '<small>(' + (d.lat ? d.lat.toFixed(5) : '?') + ', ' +
          (d.lon ? d.lon.toFixed(5) : '?') + ')</small>';
      } else {
        tooltipEl.style.display = 'none';
      }
    }

    function onMouseLeave() {
      tooltipEl.style.display = 'none';
      if (hoveredObject) {
        if (hoveredObject.material.emissive) {
          hoveredObject.material.emissive.setHex(0x000000);
        }
        hoveredObject = null;
      }
    }

    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('mouseleave', onMouseLeave);

    // ===== 9. 信息叠加 =====
    infoEl = document.createElement('div');
    infoEl.className = 'pareto-3d-info';
    var hvVal = result.hv !== undefined ? result.hv : '?';
    infoEl.innerHTML = 'Pareto 解: ' + valid.length + ' 个  |  HV: ' +
      (typeof hvVal === 'number' ? hvVal.toFixed(4) : hvVal);
    container.appendChild(infoEl);

    // ===== 10. Resize 监听 =====
    resizeObserver = new ResizeObserver(function () {
      // 只有 active 场景才重绘
      if (active && currentResult) {
        // 直接重绘
        var newSize = getContainerSize();
        if (newSize.w < 50 || newSize.h < 50) return;
        if (camera && renderer) {
          camera.aspect = newSize.w / newSize.h;
          camera.updateProjectionMatrix();
          renderer.setSize(newSize.w, newSize.h);
        }
      }
    });
    resizeObserver.observe(container);

    // ===== 11. 动画循环 =====
    active = true;

    function animate() {
      controls.update();
      renderer.render(scene, camera);
      animationId = requestAnimationFrame(animate);
    }
    animate();
  }

  // 公开 API
  window.__drawParetoChart3D = function (result) {
    currentResult = result;
    drawScene(result);
  };

})();