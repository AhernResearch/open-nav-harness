import { useRef, useState } from "react";
import type { Trace } from "./trace";

type Box = [number, number, number, number];

/** Fit the entire observed map and any movement since that map was recorded. */
export function MapView({ trace, upto, base }: { trace: Trace; upto: number; base: string }) {
  const found = trace.map(upto);
  if (!found?.item.map?.size) return <div className="map-empty">智能体查看地图后会显示在这里</div>;
  return <MapCanvas key={`${base}:${found.item.map.artifact}`} trace={trace} upto={upto} base={base} />;
}

function MapCanvas({ trace, upto, base }: { trace: Trace; upto: number; base: string }) {
  const [viewport, setViewport] = useState<Box | null>(null);
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<{ id: number; x: number; y: number; box: Box } | null>(null);
  const found = trace.map(upto)!;
  const { item, index } = found;
  const map = item.map!;
  const [w, h] = map.size!;
  const m = map.transform?.matrix;
  const project = (p: number[]) => [m![0] * p[0] + m![1] * p[2] + m![2], m![3] * p[0] + m![4] * p[2] + m![5]];
  const current = trace.latestMotion(upto);
  const trail = m ? trace.motions(index + 1, upto).flatMap((i) => i.poses).map((p) => project(p.position)) : [];
  const live = m && current ? current.poses.map((p) => project(p.position)) : [];
  const pose = m ? current?.poses.at(-1) : undefined;
  const places = Object.entries(map.places_px);
  const focus = memoryFocus(trace, upto);
  const points = [[0, 0], [w, h], ...places.map(([, p]) => p), ...trail, ...live];
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const fit: Box = [Math.min(...xs) - 32, Math.min(...ys) - 32, Math.max(...xs) - Math.min(...xs) + 64, Math.max(...ys) - Math.min(...ys) + 64];
  const box = viewport ?? fit;
  const zoom = (factor: number) => {
    const scale = box[2] / fit[2];
    const next = Math.min(4, Math.max(0.125, scale * factor)) / scale;
    setViewport([box[0] + (box[2] - box[2] * next) / 2, box[1] + (box[3] - box[3] * next) / 2, box[2] * next, box[3] * next]);
  };
  const at = (id: string) => map.places_px[id];
  return (
    <div className="map-view">
    <div className="map-stage">
    <svg ref={svg} className={"map-svg" + (viewport ? " zoomed" : "")} viewBox={box.join(" ")} preserveAspectRatio="xMidYMid meet" role="img" aria-label="记忆地图"
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY, box };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        const start = drag.current;
        const matrix = svg.current?.getScreenCTM();
        if (!start || start.id !== event.pointerId || !matrix) return;
        setViewport([start.box[0] - (event.clientX - start.x) / matrix.a, start.box[1] - (event.clientY - start.y) / matrix.d, start.box[2], start.box[3]]);
      }}
      onPointerUp={(event) => {
        if (drag.current?.id === event.pointerId) drag.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onPointerCancel={() => { drag.current = null; }}
      onLostPointerCapture={() => { drag.current = null; }}>
      <rect className="map-void" x={box[0]} y={box[1]} width={box[2]} height={box[3]} />
      <image key={map.artifact} className="map-image" href={`${base}/artifact/${map.artifact}`} width={w} height={h} />
      {m && (
        <g className="overlay">
          {focus.edge && at(focus.edge[0]) && at(focus.edge[1]) && (
            <line key={`${focus.edge}-${upto}`} className="memory-edge" x1={at(focus.edge[0])[0]} y1={at(focus.edge[0])[1]} x2={at(focus.edge[1])[0]} y2={at(focus.edge[1])[1]} pathLength={1} />
          )}
          {trail.length > 1 && <polyline className="trail" points={trail.map((p) => p.join(",")).join(" ")} />}
          {live.length > 1 && <polyline key={current!.id} className="live" points={live.map((p) => p.join(",")).join(" ")} pathLength={1} />}
          {places.map(([id, [x, y]]) => focus.nodes.has(id) && <circle key={`${id}-${upto}`} className="node-focus" cx={x} cy={y} r={18} />)}
          {pose && <Heading position={project(pose.position)} heading={pose.heading_rad ?? 0} m={m} />}
        </g>
      )}
    </svg>
    </div>
    <div className="map-controls" role="group" aria-label="地图视图">
      <button type="button" aria-label="缩小地图" title="缩小地图" onClick={() => zoom(1.4)}>−</button>
      <button type="button" aria-label="放大地图" title="放大地图" onClick={() => zoom(1 / 1.4)}>＋</button>
      <button type="button" className="map-fit" onClick={() => setViewport(null)}>显示全图</button>
    </div>
    </div>
  );
}

function Heading({ position: [x, y], heading, m }: { position: number[]; heading: number; m: number[] }) {
  // Habitat faces -z at heading 0 and turns about +y.
  const fx = -Math.sin(heading), fz = -Math.cos(heading);
  const angle = (Math.atan2(m[3] * fx + m[4] * fz, m[0] * fx + m[1] * fz) * 180) / Math.PI;
  return (
    <g className="pose" transform={`translate(${x} ${y})`}>
      <circle className="halo" r="16" />
      <g transform={`rotate(${angle})`}>
        <path d="M 14 0 L -7 -8 L -3 0 L -7 8 Z" />
      </g>
    </g>
  );
}

/** Places a memory read or write at this point of the trace refers to. */
function memoryFocus(trace: Trace, upto: number): { nodes: Set<string>; edge: [string, string] | null } {
  const item = trace.itemAt(upto);
  const a = item?.arguments ?? {};
  if (item?.type === "memory_read") {
    const ids = [a.node_id, a.other_node_id].filter((v) => v != null).map(String);
    return { nodes: new Set(ids), edge: ids.length === 2 ? [ids[0], ids[1]] : null };
  }
  if (item?.type === "memory_write") {
    const ids: [string, string] = [String(a.start_node_id), String(a.end_node_id)];
    return { nodes: new Set(ids), edge: ids };
  }
  return { nodes: new Set(), edge: null };
}
