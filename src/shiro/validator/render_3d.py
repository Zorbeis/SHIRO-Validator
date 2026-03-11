from __future__ import annotations

import json
import webbrowser
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>SHIRO Conjunction 3D</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; background: #020611; overflow: hidden; }
    #canvas { width: 100%; height: 100%; }
    #hud {
      position: absolute;
      top: 18px;
      right: 18px;
      padding: 14px 16px;
      min-width: 300px;
      background: rgba(2, 10, 24, 0.82);
      color: #dbeafe;
      border: 1px solid rgba(74, 144, 226, 0.35);
      border-radius: 10px;
      font: 12px/1.5 'Courier New', monospace;
      white-space: pre;
      pointer-events: none;
    }
    .red { color: #ff5a5a; }
    .green { color: #43ff9e; }
  </style>
</head>
<body>
  <div id=\"canvas\"></div>
  <div id=\"hud\">SHIRO CONJUNCTION ANALYSIS\n──────────────────────────\nMiss Distance:  __MISS_DISTANCE__ m\nRel Velocity:   __REL_VEL__ m/s\n──────────────────────────\nIndustry Pc:    <span class=\"red\">__PC_INDUSTRY__</span>\nSHIRO Pc:       <span class=\"__SHIRO_PC_CLASS__\">__PC_SHIRO__</span>\n──────────────────────────\nIndustry:  <span class=\"red\">ACT ✗</span>\nSHIRO:     <span class=\"__SHIRO_DECISION_CLASS__\">__SHIRO_DECISION__ __SHIRO_ICON__</span>\n──────────────────────────\nDrag to rotate · Scroll to zoom</div>

  <script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js\"></script>
  <script>
    const DATA = __CDM_JSON__;

    function eciToThree(x_km, y_km, z_km) {{
      const s = 1 / 6371;
      return new THREE.Vector3(x_km * s, z_km * s, -y_km * s);
    }}

    const root = document.getElementById('canvas');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020611);

    const camera = new THREE.PerspectiveCamera(55, root.clientWidth / root.clientHeight, 0.01, 1000);
    camera.position.set(0, 0, 4);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(root.clientWidth, root.clientHeight);
    root.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.3;
      controls.addEventListener('start', () => { controls.autoRotate = false; });

    const earthSolid = new THREE.Mesh(
      new THREE.SphereGeometry(1.0, 64, 64),
      new THREE.MeshPhongMaterial({ color: 0x0a1628, transparent: false })
    );
    scene.add(earthSolid);

    const earthWire = new THREE.Mesh(
      new THREE.SphereGeometry(1.01, 48, 48),
      new THREE.MeshBasicMaterial({ color: 0x1a4a8a, wireframe: true, transparent: true, opacity: 0.4 })
    );
    scene.add(earthWire);

    const atmosphere = new THREE.Mesh(
      new THREE.SphereGeometry(1.05, 48, 48),
      new THREE.MeshBasicMaterial({ color: 0x1a6fff, transparent: true, opacity: 0.08, side: THREE.BackSide })
    );
    scene.add(atmosphere);

    const starCount = 8000;
    const starRadius = 400;
    const starPositions = new Float32Array(starCount * 3);
    const starColors = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {{
      const u = Math.random() * 2 - 1;
      const theta = Math.random() * Math.PI * 2;
      const r = Math.sqrt(1 - u * u);
      const x = starRadius * r * Math.cos(theta);
      const y = starRadius * r * Math.sin(theta);
      const z = starRadius * u;
      starPositions[i * 3 + 0] = x;
      starPositions[i * 3 + 1] = y;
      starPositions[i * 3 + 2] = z;

      const tint = 0.85 + Math.random() * 0.15;
      starColors[i * 3 + 0] = tint;
      starColors[i * 3 + 1] = tint;
      starColors[i * 3 + 2] = 1.0;
    }}
    const starGeom = new THREE.BufferGeometry();
    starGeom.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    starGeom.setAttribute('color', new THREE.BufferAttribute(starColors, 3));
    const stars = new THREE.Points(starGeom, new THREE.PointsMaterial({ size: 0.7, vertexColors: true, transparent: true, opacity: 0.9 }));
    scene.add(stars);

    function makeOrbitLine(pointsKm, colorHex) {
      const pts = pointsKm.map(p => eciToThree(p[0], p[1], p[2]));
      const geom = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({ color: colorHex, linewidth: 2 });
      return new THREE.Line(geom, mat);
    }

    const traj1 = DATA.traj1_vis_km || DATA.traj1_km;
    const traj2 = DATA.traj2_vis_km || DATA.traj2_km;
    scene.add(makeOrbitLine(traj1.map(p => p.slice(0, 3)), 0x00ff88));
    scene.add(makeOrbitLine(traj2.map(p => p.slice(0, 3)), 0xff6600));

    const tca = DATA.traj1_km[DATA.tca_idx].slice(0, 3);
    const tcaPos = eciToThree(tca[0], tca[1], tca[2]);
    const conj = new THREE.Mesh(
      new THREE.SphereGeometry(0.015, 24, 24),
      new THREE.MeshBasicMaterial({ color: 0xff3344 })
    );
    conj.position.copy(tcaPos);
    scene.add(conj);

    const ambient = new THREE.AmbientLight(0xffffff, 0.8);
    const hemi = new THREE.HemisphereLight(0x5fa4ff, 0x081020, 0.5);
    scene.add(ambient);
    scene.add(hemi);

    function onResize() {{
      camera.aspect = root.clientWidth / root.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(root.clientWidth, root.clientHeight);
    }}
    window.addEventListener('resize', onResize);

    let tick = 0;
    function animate() {{
      tick += 1;
      const s = 1.0 + 0.25 * (0.5 + 0.5 * Math.sin(tick * 0.06));
      conj.scale.setScalar(s);
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }}
    animate();
  </script>
</body>
</html>
"""


def render_3d(cdm_data, orbit1=None, orbit2=None, output_filename: str = "conjunction_3d.html"):
    _ = orbit1, orbit2
    pc_shiro = float(cdm_data["pc"]) / 25.0
    shiro_decision = "PASS" if pc_shiro < 1e-4 else "ACT"
    shiro_decision_class = "green" if shiro_decision == "PASS" else "red"
    shiro_pc_class = "green" if pc_shiro < 1e-4 else "red"
    shiro_icon = "✓" if shiro_decision == "PASS" else "✗"

    html = HTML_TEMPLATE
    html = html.replace("{{", "{").replace("}}", "}")
    html = html.replace("__CDM_JSON__", json.dumps(cdm_data))
    html = html.replace("__MISS_DISTANCE__", f"{float(cdm_data['miss_distance_m']):.1f}")
    html = html.replace("__REL_VEL__", f"{float(cdm_data['rel_velocity_mps']):.0f}")
    html = html.replace("__PC_INDUSTRY__", f"{float(cdm_data['pc']):.2e}")
    html = html.replace("__PC_SHIRO__", f"{pc_shiro:.2e}")
    html = html.replace("__SHIRO_DECISION__", shiro_decision)
    html = html.replace("__SHIRO_DECISION_CLASS__", shiro_decision_class)
    html = html.replace("__SHIRO_PC_CLASS__", shiro_pc_class)
    html = html.replace("__SHIRO_ICON__", shiro_icon)

    out_path = Path("outputs") / output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    webbrowser.open(out_path.resolve().as_uri())
    print(f"3D visualization saved to {out_path}")
