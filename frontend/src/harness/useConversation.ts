import { useEffect, useMemo, useState } from "react";
import { api, type Row } from "../api";
import { Trace } from "./trace";

/** Conversation transport projected onto the same timeline used by navigation runs. */
export function useConversation() {
  const [id, setId] = useState<string | null>(new URLSearchParams(location.search).get("conversation"));
  const [options, setOptions] = useState<Row>({ models: [], mcp: [], navigation: [] });
  const [ready, setReady] = useState(false);
  const [sessions, setSessions] = useState<Row[]>([]);
  const [model, setModel] = useState("");
  const [navigation, setNavigation] = useState("");
  const [episode, setEpisode] = useState("");
  const [useMcp, setUseMcp] = useState(true);
  const [events, setEvents] = useState<Row[]>([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const refresh = () => api<Row[]>("/conversations").then(rows => setSessions(rows.map(row => ({ ...row, spec: typeof row.spec === "string" ? JSON.parse(row.spec) : row.spec })))).catch(e => setError(String(e)));
  useEffect(() => {
    api("/conversations/options").then(o => { setOptions(o); setReady(true); setModel(o.default_model ?? ""); }).catch(e => setError(String(e)));
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    setEvents([]); setError(""); setStatus("idle");
    if (!id) return;
    let active = true;
    let source: EventSource | undefined;
    api(`/conversations/${id}`).then(row => {
      if (!active) return;
      setEvents(row.events); setStatus(row.status); setModel(row.spec.model); setNavigation(row.spec.navigation ?? ""); setEpisode(row.spec.episode ?? ""); setUseMcp(row.spec.mcp.length > 0);
      source = new EventSource(`/api/conversations/${id}/events?after=${row.events.at(-1)?.seq ?? 0}`);
      source.onmessage = message => {
        const event = JSON.parse(message.data);
        setEvents(old => old.some(e => e.seq === event.seq) ? old : [...old, event]);
        if (event.kind === "user_text") setStatus("running");
        if (event.kind === "status") { setStatus(event.payload.status); refresh(); }
      };
      source.onerror = () => setError("连接中断，正在自动重连…");
      source.onopen = () => setError("");
    }).catch(e => { if (active) setError(String(e)); });
    return () => { active = false; source?.close(); };
  }, [id]);
  const open = (next: string | null) => {
    setId(next);
    if (!next) { setModel(options.default_model ?? ""); setNavigation(""); setEpisode(""); setUseMcp(true); }
    history.replaceState(null, "", next ? `/harness?conversation=${next}` : "/harness");
  };
  const send = async (content: string) => {
    if (busy || !content.trim()) return false;
    setBusy(true); setError("");
    try {
      let cid = id;
      if (!cid) {
        const row = await api("/conversations", { model, mcp: useMcp ? options.mcp : [], navigation: navigation || null, episode: episode || null });
        cid = row.id; open(cid); refresh();
      }
      await api(`/conversations/${cid}/messages`, { request_id: crypto.randomUUID(), content });
      setStatus("running");
      return true;
    } catch (e) { setError(String(e)); return false; }
    finally { setBusy(false); }
  };
  const stop = async () => {
    try { if (id) { const row = await api(`/conversations/${id}/stop`, {}); setStatus(row.status); } }
    catch (e) { setError(String(e)); }
  };
  /** Bind an environment to one dataset episode; the kernel opens a sim-engine session for it. */
  const bind = async (value: string, chosen: string | null) => {
    setNavigation(value); setEpisode(chosen ?? "");
    if (!id) return;
    try { await api(`/conversations/${id}/navigation`, { navigation: value || null, episode: chosen }); }
    catch (e) { setError(String(e)); }
  };
  const trace = useMemo(() => {
    const result = new Trace();
    for (const event of events) {
      const p = event.payload;
      const t = event.seq;
      if (event.kind === "user_text") result.apply([{ type: "user.message", text: p.text, t }]);
      if (event.kind === "navigation_session") result.apply([{ type: "session.started", instruction: p.instruction, episode_id: p.episode }]);
      if (event.kind === "assistant_text") result.apply([{ type: "turn.completed", text: p.text, t }]);
      if (event.kind === "tool_use") result.apply([{ type: "item.started", t, item: { id: p.id, type: "tool", name: p.name, arguments: p.input, status: "in_progress", media_base: `/api/conversations/${id}/media/` } }]);
      if (event.kind === "tool_result") result.apply([{ type: "item.completed", t, item: { id: p.tool_use_id, status: p.is_error ? "failed" : "completed", result: p.texts, images: (p.images ?? []).filter((part: Row) => part.type === "image_url").map((part: Row) => part.image_url.url.artifact) } }]);
      if (["error", "connection_error", "navigation_blocked"].includes(event.kind)) result.apply([{ type: "error", t, kind: p.message ?? p.error ?? event.kind, error: p }]);
    }
    result.status = { status }; return result;
  }, [events, status, id]);
  return { id, options, ready, sessions, model, setModel, navigation, episode, bind, useMcp, setUseMcp, status, error, setError, busy, open, send, stop, trace };
}
export type Conversation = ReturnType<typeof useConversation>;
