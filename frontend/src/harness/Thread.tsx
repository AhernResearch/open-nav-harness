import { useEffect, useRef, useState } from "react";
import { Icon } from "./icons";
import { toolImages, type Entry, type Item, type Trace } from "./trace";
import { CameraGallery } from "./CameraGallery";

/** What a row says in plain words; raw arguments stay behind the expander. */
const viewNames: Record<string, string> = { front: "前方", right: "右侧", back: "后方", left: "左侧" };
const legStatus: Record<string, string> = {
  reached: "已到达", stopped_short: "未完全到达", blocked: "受阻", invalid_target: "目标点无效",
  unreachable: "无法到达", leg_budget: "本段步数用尽", episode_budget: "总步数用尽",
};

function describe(item: Item): { icon: string; label: string } {
  const a = item.arguments ?? {};
  switch (item.name) {
    case "observe":
      return { icon: "eye", label: "观察四周" };
    case "observe_map":
      return { icon: "map", label: "查看地图" };
    case "update_plan":
      return { icon: "list", label: a.patch ? `更新计划 · 调整 ${a.patch.length} 项状态` : `更新计划 · ${(a.steps ?? a.subtasks)?.length ?? 0} 项` };
    case "delegate":
      return { icon: "route", label: `${a.instruction} · ${viewNames[a.view] ?? a.view}` };
    case "return_to":
      return { icon: "route", label: `沿原路返回第 ${a.leg_id} 段终点` };
    case "execute_subtask":
      return { icon: "route", label: `${a.instruction || `执行任务 ${a.task_id}`}${a.stop_at ? ` · Stop: ${a.stop_at}` : ""}${a.max_distance_m ? ` · ≤${a.max_distance_m} 米` : ""}` };
    case "navigate_relative":
      return { icon: "route", label: `转向 ${a.turn_deg ?? 0}°，前进 ${a.distance_m ?? 0} 米` };
    case "navigate_to_node":
      return { icon: "route", label: `返回地点 ${a.node_id}` };
    case "update_memory":
      return { icon: "pencil", label: `记住 ${a.start_node_id} → ${a.end_node_id}` };
    case "read_memory":
      return { icon: "book", label: a.other_node_id == null ? `回忆地点 ${a.node_id}` : `回忆 ${a.node_id} → ${a.other_node_id}` };
    case "stop":
    case "terminate_episode":
      return { icon: "flag", label: a.status === "abort" ? "放弃任务" : `宣布到达${a.evidence?.length ? ` · ${a.evidence.length} 条证据` : ""}` };
    default:
      return { icon: "tool", label: item.name };
  }
}

export function PlanCard({ tasks }: { tasks: Record<string, any>[] }) {
  const [open, setOpen] = useState(true);
  if (!tasks.length) return null;
  const done = tasks.filter((t) => t.status === "completed").length;
  return (
    <div className="plan-card">
      <button className="plan-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <Icon name="list" />
        <span>计划</span>
        <span className="plan-count">
          {done}/{tasks.length}
        </span>
        <span className="progress">
          <i style={{ width: `${(100 * done) / tasks.length}%` }} />
        </span>
        <Icon name="chevron" />
      </button>
      {open && (
        <ol>
          {tasks.map((t) => (
            <li key={t.id} className={t.status}>
              <span className="check">{t.status === "completed" && <Icon name="check" size={12} />}</span>
              <span className={t.status === "in_progress" ? "shimmer" : ""}>{t.instruction}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

/** The conversation: the instruction, model messages and one row per tool call. */
export function Thread({ trace, upto, live, select, base }: { trace: Trace; upto: number; live: boolean; select: (i: number) => void; base: string }) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const current = el.querySelector<HTMLElement>(".current");
    el.scrollTop = live || !current ? el.scrollHeight : current.offsetTop - el.clientHeight / 2;
  }, [live, upto, trace.entries.length]);
  const working = !trace.ended && trace.status?.status === "running";
  return (
    <div className="thread" ref={box}>
      <div className="thread-col">
        {trace.meta && (
          <div className="user-msg">
            <p>{trace.meta.instruction}</p>
            <span className="scene-chip">{String(trace.meta.scene_id ?? "").split("/").at(-1)?.replace(".glb", "")}</span>
          </div>
        )}
        {trace.entries.map((e, i) => (
          <Row key={i} entry={e} trace={trace} base={base} state={i === upto ? "current" : i > upto ? "future" : ""} onSelect={() => select(i)} />
        ))}
        {working && live && trace.entries.at(-1)?.kind !== "item" && <p className="thinking shimmer">正在思考…</p>}
        {trace.session && (
          <p className="ended">{trace.session.explicit_stop ? "智能体已停止" : trace.session.error ? "会话异常结束" : "会话结束"}</p>
        )}
      </div>
    </div>
  );
}

function Row({ entry, trace, base, state, onSelect }: { entry: Entry; trace: Trace; base: string; state: string; onSelect: () => void }) {
  const [open, setOpen] = useState(false);
  if (entry.kind === "user") return <div className="user-msg"><p>{entry.text}</p></div>;
  if (entry.kind === "turn") {
    if (!entry.text && !entry.reasoning) return null;
    return (
      <div className={"agent-msg " + state} onClick={onSelect}>
        {entry.text || <span className="muted">{entry.reasoning}</span>}
      </div>
    );
  }
  if (entry.kind === "error") return <div className={"tool-row failed " + state}>出现错误：{String(entry.source)}</div>;
  const item = trace.items.get(entry.id)!;
  const { icon, label } = describe(item);
  const steps = item.leg ? [legStatus[item.leg.status] ?? item.leg.status, item.frames.length ? `${item.frames.length} 步` : ""].filter(Boolean).join(" · ")
    : item.type === "motion" && item.poses.length ? `${item.poses.length} 步` : "";
  return (
    <div className={`tool ${item.type} ${state}`}>
      <button
        className={"tool-row " + item.status}
        onClick={() => {
          onSelect();
          setOpen(!open);
        }}
        aria-expanded={open}
      >
        <Icon name={icon} />
        <span className={"label " + (item.status === "in_progress" ? "shimmer" : "")}>{label}</span>
        {steps && <span className="meta">{steps}</span>}
        <span className="status">
          {item.status === "in_progress" ? <i className="spinner" /> : <Icon name={item.status === "failed" ? "x" : "check"} size={14} />}
        </span>
      </button>
      {item.name === "update_memory" && item.status !== "failed" && <blockquote className="memory-quote">{item.arguments?.caption}</blockquote>}
      {open && (
        <div className="tool-detail">
          {(item.result ?? []).filter(Boolean).map((line, i) => (
            <div key={i}>{line}</div>
          ))}
          {!!item.images?.length && (
            <CameraGallery className="thumbs" images={toolImages(item, base, "工具返回的图像")} />
          )}
        </div>
      )}
    </div>
  );
}
