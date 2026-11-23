import type { Row } from "../api";

export type Pose = { position: number[]; heading_rad?: number };
export type CameraImage = { src: string; caption: string };
/** Tool-result images paired with their recorded camera captions; `base` is the episode's API path. */
export function toolImages(item: Item, base: string, fallback: string): CameraImage[] {
  return (item.images ?? []).map((artifact, index) => ({
    src: item.media_base ? item.media_base + artifact : `${base}/artifact/${artifact}`,
    caption: item.image_views?.[index]?.caption
      ?? (item.name === "observe" ? item.arguments.views?.[index] : undefined) ?? fallback,
  }));
}
export type MapInfo = {
  artifact: string;
  size: [number, number] | null;
  places_px: Record<string, [number, number]>;
  nodes?: Record<string, number[]>;
  transform: { matrix: number[]; ppm: number; residual_px: number; nodes: number } | null;
};
export type Item = {
  id: string;
  type: "tool" | "plan" | "memory_write" | "memory_read" | "map" | "motion";
  name: string;
  arguments: Row;
  status: "in_progress" | "completed" | "failed";
  result?: string[];
  state?: Row | null;
  images?: string[];
  media_base?: string;
  image_views?: { artifact: string; caption: string }[];
  map?: MapInfo;
  poses: Pose[];
  frames: Record<string, string>[];
  execution?: Row;
  leg?: Row;
  t?: number;
  done?: number;
};
export type Entry =
  | { kind: "user"; t: number; text: string }
  | { kind: "turn"; t: number; text: string | null; reasoning: string | null; usage: Row | null; elapsed_s: number }
  | { kind: "item"; id: string; t: number }
  | { kind: "error"; t: number; error: unknown; source: string };

/** One attempt's events, folded by item id. Panels read it "as of" a timeline index. */
export class Trace {
  items = new Map<string, Item>();
  entries: Entry[] = [];
  status: Row | null = null;
  session: Row | null = null;
  meta: Row | null = null;
  ended = false;
  private version = 0;
  private listeners = new Set<() => void>();

  apply(events: Row[]) {
    for (const e of events) this.fold(e);
    this.changed();
  }

  finish() {
    this.ended = true;
    this.changed();
  }

  clear() {
    this.items.clear();
    this.entries = [];
    this.session = null;
    this.meta = null;
    this.ended = false;
    this.changed();
  }

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => void this.listeners.delete(listener);
  };
  snapshot = () => this.version;

  /** Latest plan as of entry `upto`; every tool result carries the persisted task state. */
  plan(upto: number): Row[] {
    return this.find(upto, (i) => i.state?.tasks)?.state?.tasks ?? [];
  }

  /** Latest map as of `upto`, plus the index of the entry that produced it. */
  map(upto: number): { item: Item; index: number } | null {
    for (let i = upto; i >= 0; i--) {
      const item = this.itemAt(i);
      if (item?.map) return { item, index: i };
    }
    return null;
  }

  /** Movement items between two entries, in execution order. */
  motions(from: number, upto: number): Item[] {
    const out: Item[] = [];
    for (let i = Math.max(0, from); i <= upto; i++) {
      const item = this.itemAt(i);
      if (item?.type === "motion") out.push(item);
    }
    return out;
  }

  latestMotion(upto: number) {
    return this.find(upto, (i) => i.type === "motion");
  }

  /** Observe results and motion cameras share one timeline; maps are not cameras. */
  camera(upto: number, base: string): CameraImage[] {
    for (let i = upto; i >= 0; i--) {
      const item = this.itemAt(i);
      if (item?.name === "observe" && item.images?.length) return toolImages(item, base, "相机画面");
      const frame = item?.type === "motion" && item.frames.at(-1);
      if (frame) return Object.entries(frame).filter(([, path]) => path).map(([name, path]) => ({
        src: `${base}/files/${path}`, caption: name,
      }));
      if (item?.type === "motion" && item.images?.length) return toolImages(item, base, "相机画面");
    }
    return [];
  }

  itemAt(index: number): Item | undefined {
    const e = this.entries[index];
    return e?.kind === "item" ? this.items.get(e.id) : undefined;
  }

  private find(upto: number, test: (item: Item) => unknown) {
    for (let i = upto; i >= 0; i--) {
      const item = this.itemAt(i);
      if (item && test(item)) return item;
    }
    return undefined;
  }

  private fold(e: Row) {
    const patch = e.item as Row | undefined;
    const item = patch && this.items.get(patch.id);
    switch (e.type) {
      case "user.message":
        this.entries.push({ kind: "user", t: e.t, text: e.text });
        break;
      case "turn.completed":
        this.entries.push({ kind: "turn", t: e.t, text: e.text, reasoning: e.reasoning, usage: e.usage, elapsed_s: e.elapsed_s });
        break;
      case "item.started":
        this.items.set(patch!.id, { ...(patch as Item), poses: [], frames: [], t: e.t });
        this.entries.push({ kind: "item", id: patch!.id, t: e.t });
        break;
      case "item.completed":
        if (item) Object.assign(item, patch, { done: e.t });
        break;
      case "item.updated":
        if (!item) break;
        if (patch!.pose) item.poses = [...item.poses, patch!.pose];
        if (patch!.frame) item.frames = [...item.frames, patch!.frame];
        if (patch!.frames) item.frames = patch!.frames; // a whole leg at once: idempotent on replay
        if (patch!.execution) item.execution = patch!.execution;
        if (patch!.map) item.map = patch!.map;
        break;
      case "error":
        this.entries.push({ kind: "error", t: e.t, error: e.error, source: e.kind });
        break;
      case "session.started":
        this.meta = e;
        break;
      case "session.ended":
        this.session = e;
        break;
      case "attempt.status":
        this.status = e;
        break;
    }
  }

  private changed() {
    this.version++;
    this.listeners.forEach((l) => l());
  }
}
