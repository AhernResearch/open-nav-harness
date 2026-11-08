import type { Trace } from "./trace";

/** The latest map the model saw, framed on where the robot has been, with live overlays. */
export function MapView({ trace, upto, attempt }: { trace: Trace; upto: number; attempt: string }) {
  const found = trace.map(upto);
  if (!found?.item.map?.size) return <div className="map-empty">智能体查看地图后会显示在这里</div>;
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
  // Frame what matters (visited places and paths), not the whole observed footprint.
  const points = [...places.map(([, p]) => p), ...trail, ...live];
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const span = Math.max(260, Math.max(...xs) - Math.min(...xs) + 160, Math.max(...ys) - Math.min(...ys) + 160);
  const cx = points.length ? (Math.max(...xs) + Math.min(...xs)) / 2 : w / 2;
  const cy = points.length ? (Math.max(...ys) + Math.min(...ys)) / 2 : h / 2;
  const box = [cx - span / 2, cy - span / 2, span, span];
  const at = (id: string) => map.places_px[id];
  return (
    <svg className="map-svg" viewBox={box.join(" ")} role="img" aria-label="记忆地图">
      <rect className="map-void" x={box[0]} y={box[1]} width={span} height={span} />
      <image key={map.artifact} className="map-image" href={`/api/attempts/${attempt}/artifact/${map.artifact}`} width={w} height={h} />
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
