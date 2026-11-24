import { useEffect, useRef, useState } from "react";
import { api, type Row } from "../api";
import { Icon } from "./icons";
import { ModelPicker } from "./ModelPicker";
import type { Conversation } from "./useConversation";

export function Pill({ label, value, options, onChange }: { label: string; value: string; options: [string, string, boolean?][]; onChange: (v: string) => void }) {
  return (
    <label className="pill">
      <span>{label}</span>
      <select aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}>
        {!options.length && <option value="">暂无可用选项</option>}
        {options.map(([v, text, disabled]) => (
          <option key={v} value={v} disabled={disabled}>
            {text}
          </option>
        ))}
      </select>
    </label>
  );
}

/** An environment alias plus one of its dataset episodes: what a binding opens. */
function EnvironmentPicker({ conversation: c }: { conversation: Conversation }) {
  const [catalog, setCatalog] = useState<Row | null>(null);
  const [scene, setScene] = useState("");
  const [episodes, setEpisodes] = useState<Row[]>([]);
  const target = (c.options.navigation as Row[]).find((n) => n.name === c.navigation);
  useEffect(() => {
    setCatalog(null); setScene(""); setEpisodes([]);
    if (!target) return;
    api<Row[]>("/catalog").then(rows => {
      const found = rows.find((row) => row.name === target.catalog);
      setCatalog(found ?? null);
      setScene(found?.scenes?.[0]?.scene ?? "");
    }).catch(e => c.setError(String(e)));
  }, [target?.catalog]);
  useEffect(() => {
    if (!catalog || !scene) return;
    api<Row[]>(`/catalog/${catalog.name}/episodes?scene=${encodeURIComponent(scene)}&limit=200`)
      .then(setEpisodes).catch(e => c.setError(String(e)));
  }, [catalog?.name, scene]);
  const choose = (index: string) => void c.bind(c.navigation, index || null);  // the alias is already bound
  return <>
    <Pill label="环境" value={c.navigation} options={[["", "暂不绑定"], ...(c.options.navigation as Row[]).map((n) => [n.name, n.name] as [string, string])]} onChange={value => void c.bind(value, null)}/>
    {target && !c.episode && catalog && <>
      <Pill label="场景" value={scene} options={catalog.scenes.map((s: Row) => [s.scene, String(s.scene).split("/").at(-1)?.replace(".glb", "") ?? s.scene] as [string, string])} onChange={setScene}/>
      <Pill label="episode" value="" options={[["", "选择 episode"], ...episodes.map((e) => [String(e.index), `#${e.index} ${String(e.instruction ?? "").slice(0, 28)}`] as [string, string])]} onChange={choose}/>
    </>}
    {target && c.episode && <span className="pill"><Icon name="route" size={13}/>#{c.episode}</span>}
  </>;
}

/** The existing input surface also accepts arbitrary conversation instructions. */
export function ConversationComposer({ conversation: c }: { conversation: Conversation }) {
  const [message, setMessage] = useState("");
  useEffect(() => setMessage(""), [c.id]);
  const submitting = useRef(false);
  const send = async () => {
    if (submitting.current || !message.trim()) return;
    submitting.current = true;
    try { if (await c.send(message)) setMessage(""); }
    finally { submitting.current = false; }
  };
  const services = c.options.models.map((model: string) => ({ id: model, available: true, unavailable_reason: "新建会话可选择其他模型", spec: { name: "对话模型", planner: { model } } }));
  const input = <>
    <div className="composer">
      <textarea aria-label="指令" rows={2} placeholder="自由聊天、查找地图物体，或描述你想去的地方…" value={message} onChange={e => setMessage(e.target.value)} onKeyDown={e => {
        if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void send(); }
      }}/>
      <div className="composer-bar">
        <ModelPicker disabled={!!c.id} services={services} value={c.model} onChange={c.setModel}/>
        {!c.id && <label className="pill"><input type="checkbox" checked={c.useMcp} onChange={e => c.setUseMcp(e.target.checked)}/>MCP 工具</label>}
        {c.id && c.useMcp && <span className="pill"><Icon name="map" size={13}/>MCP 工具</span>}
        {c.options.navigation.length > 0 && <EnvironmentPicker conversation={c}/>}
        <span className="spacer"/>
        {c.status === "running" && <button className="pill-button danger" onClick={() => void c.stop()}><Icon name="stop" size={14}/>停止</button>}
        <button className="send" aria-label="发送指令" disabled={!message.trim() || c.busy || !c.model} onClick={() => void send()}><Icon name="send"/></button>
      </div>
    </div>
    {c.error && <p className="composer-error" role="alert">{c.error}</p>}
  </>;
  return c.id ? <div className="conversation-input">{input}</div> : <div className="hero"><h1>今天让机器人去哪里？</h1><p className="muted">自由交流、查询地图，或下达导航指令。</p>{input}<p className="hint muted">{c.options.navigation.length ? "可以随时补充条件、改口或停止。" : "可直接聊天和查地图；连接执行环境后可进行导航。"}</p></div>;
}
