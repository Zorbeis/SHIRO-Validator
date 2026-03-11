import { useEffect, useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import * as THREE from "three";

import { fetchLatestConjunction } from "../lib/api";
import type { ConjunctionLatestResponse } from "../lib/types";

const EARTH_RADIUS_KM = 6371.0;
const EARTH_RADIUS_VIS = 1.0;
const KM_TO_VIS = EARTH_RADIUS_VIS / EARTH_RADIUS_KM;
const M_TO_VIS = KM_TO_VIS / 1000;

function eciToVis(x: number, y: number, z: number) {
  return {
    x: x * M_TO_VIS,
    y: z * M_TO_VIS,
    z: -y * M_TO_VIS,
  };
}

function eciArrayToVis(pos: number[]) {
  return eciToVis(pos[0], pos[1], pos[2]);
}

function toCurve(points: number[][]): THREE.Vector3[] {
  return points.map((p) => {
    const v = eciArrayToVis(p);
    return new THREE.Vector3(v.x, v.y, v.z);
  });
}

export default function ConjunctionView() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [data, setData] = useState<ConjunctionLatestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLatestConjunction()
      .then(setData)
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : "Failed to load conjunction data");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!data || !mountRef.current) return;

    const container = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070d);

    const camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.01, 1000);
    camera.position.set(0, 1.8, 2.4);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const earth = new THREE.Mesh(
      new THREE.SphereGeometry(1, 36, 36),
      new THREE.MeshBasicMaterial({ color: 0x4da6ff, wireframe: true, transparent: true, opacity: 0.45 })
    );
    scene.add(earth);

    const primaryPoints = data.orbit_points.primary;
    const secondaryPoints = data.orbit_points.secondary;

    let primaryCurve: THREE.BufferGeometry | null = null;
    let secondaryCurve: THREE.BufferGeometry | null = null;

    if (primaryPoints.length > 1) {
      primaryCurve = new THREE.BufferGeometry().setFromPoints(toCurve(primaryPoints));
      scene.add(new THREE.Line(primaryCurve, new THREE.LineBasicMaterial({ color: 0x51f5a9 })));
    }
    if (secondaryPoints.length > 1) {
      secondaryCurve = new THREE.BufferGeometry().setFromPoints(toCurve(secondaryPoints));
      scene.add(new THREE.Line(secondaryCurve, new THREE.LineBasicMaterial({ color: 0xff8f70 })));
    }

    const cp = eciToVis(
      data.conjunction_point_eci_m[0],
      data.conjunction_point_eci_m[1],
      data.conjunction_point_eci_m[2]
    );
    const pulse = new THREE.Mesh(
      new THREE.SphereGeometry(0.02, 24, 24),
      new THREE.MeshBasicMaterial({ color: 0xff4d6d })
    );
    pulse.position.set(cp.x, cp.y, cp.z);
    scene.add(pulse);

    scene.add(new THREE.AmbientLight(0xffffff, 0.9));

    let frame = 0;
    let raf = 0;
    const animate = () => {
      frame += 1;
      const t = frame * 0.01;
      const radius = 2.8;
      camera.position.x = Math.cos(t * 0.2) * radius;
      camera.position.z = Math.sin(t * 0.2) * radius;
      camera.lookAt(0, 0, 0);

      const scale = 1.0 + 0.25 * (0.5 + 0.5 * Math.sin(t * 4.0));
      pulse.scale.set(scale, scale, scale);

      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    const onResize = () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      if (primaryCurve) primaryCurve.dispose();
      if (secondaryCurve) secondaryCurve.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [data]);

  const bPlaneFigure = useMemo(() => {
    if (!data) return null;
    return {
      data: [
        {
          x: data.b_plane.covariance_ellipse_m.map((p) => p[0]),
          y: data.b_plane.covariance_ellipse_m.map((p) => p[1]),
          mode: "lines",
          name: "Covariance ellipse",
          line: { color: "#3b82f6", width: 2 },
        },
        {
          x: data.b_plane.hbr_circle_m.map((p) => p[0]),
          y: data.b_plane.hbr_circle_m.map((p) => p[1]),
          mode: "lines",
          name: "Combined HBR",
          line: { color: "#ef4444", width: 2 },
        },
        {
          x: [0, data.b_plane.miss_point_m[0]],
          y: [0, data.b_plane.miss_point_m[1]],
          mode: "lines+markers",
          name: "Miss vector",
          line: { color: "#22c55e", width: 3 },
          marker: { color: "#22c55e", size: 7 },
        },
      ],
      layout: {
        title: `B-Plane (Pc ${data.b_plane.pc.toExponential(2)})`,
        paper_bgcolor: "#0b1220",
        plot_bgcolor: "#0b1220",
        font: { color: "#dbeafe" },
        xaxis: { title: "Eta (m)", zeroline: true, gridcolor: "#1f2937" },
        yaxis: { title: "Zeta (m)", zeroline: true, gridcolor: "#1f2937", scaleanchor: "x", scaleratio: 1 },
        margin: { l: 55, r: 20, t: 50, b: 45 },
        legend: { orientation: "h" },
      },
      config: { displayModeBar: false, responsive: true },
    };
  }, [data]);

  const pcFigure = useMemo(() => {
    if (!data) return null;
    return {
      data: [
        {
          x: data.pc_timeline.time_to_tca_hours,
          y: data.pc_timeline.pc_reference,
          type: "scatter",
          mode: "lines",
          name: "Pc reference",
          line: { color: "#3b82f6", width: 2 },
        },
        {
          x: data.pc_timeline.time_to_tca_hours,
          y: data.pc_timeline.pc_degraded,
          type: "scatter",
          mode: "lines",
          name: "Pc degraded",
          line: { color: "#f59e0b", width: 2 },
        },
      ],
      layout: {
        title: "Pc Timeline",
        paper_bgcolor: "#0b1220",
        plot_bgcolor: "#0b1220",
        font: { color: "#dbeafe" },
        xaxis: { title: "Time to TCA (hours)", gridcolor: "#1f2937" },
        yaxis: { title: "Pc", type: "log", gridcolor: "#1f2937" },
        shapes: [
          {
            type: "line",
            xref: "paper",
            x0: 0,
            x1: 1,
            y0: data.pc_timeline.threshold,
            y1: data.pc_timeline.threshold,
            line: { color: "#ef4444", width: 2, dash: "dash" },
          },
        ],
        margin: { l: 55, r: 20, t: 50, b: 45 },
      },
      config: { displayModeBar: false, responsive: true },
    };
  }, [data]);

  if (error) {
    return <div style={{ color: "#fca5a5" }}>{error}</div>;
  }

  if (loading || !data) {
    return <div style={{ color: "#93c5fd", padding: "16px" }}>Loading conjunction data...</div>;
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      <div style={{ border: "1px solid #1e293b", borderRadius: 12, background: "#020617", padding: 8 }}>
        <div ref={mountRef} style={{ width: "100%", height: 680 }} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 16, height: 680 }}>
        <div style={{ flex: 1, border: "1px solid #1e293b", borderRadius: 12, background: "#020617", padding: 8 }}>
          {bPlaneFigure && (
            <Plot
              data={bPlaneFigure.data as never}
              layout={bPlaneFigure.layout as never}
              config={bPlaneFigure.config}
              style={{ width: "100%", height: "100%" }}
            />
          )}
        </div>
        <div style={{ flex: 1, border: "1px solid #1e293b", borderRadius: 12, background: "#020617", padding: 8 }}>
          {pcFigure && (
            <Plot
              data={pcFigure.data as never}
              layout={pcFigure.layout as never}
              config={pcFigure.config}
              style={{ width: "100%", height: "100%" }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
