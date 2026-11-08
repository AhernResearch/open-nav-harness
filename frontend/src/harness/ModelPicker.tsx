import { useEffect, useId, useRef, useState } from "react";
import type { Row } from "../api";

export function modelLabel(service: Row) {
  return service.spec.metadata?.display_name || service.spec.planner.model;
}

/** Registered profiles keep endpoint, context budget and model selection together. */
export function ModelPicker({ services, value, onChange }: { services: Row[]; value: string; onChange: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const menu = useId();
  const selected = services.find((s) => s.id === value);

  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    root.current?.querySelector<HTMLButtonElement>('[role="menuitemradio"]:not(:disabled)')?.focus();
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);

  return (
    <div className="model-picker" ref={root} onKeyDown={(event) => {
      if (event.key === "Escape") {
        setOpen(false);
        trigger.current?.focus();
      }
      if (open && ["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const items = Array.from(root.current!.querySelectorAll<HTMLButtonElement>('[role="menuitemradio"]:not(:disabled)'));
        const index = items.indexOf(document.activeElement as HTMLButtonElement);
        const next = event.key === "Home" ? 0 : event.key === "End" ? items.length - 1 : (index + (event.key === "ArrowUp" ? -1 : 1) + items.length) % items.length;
        items[next]?.focus();
      }
    }} onBlur={(event) => {
      if (!event.currentTarget.contains(event.relatedTarget as Node)) setOpen(false);
    }}>
      <button type="button" ref={trigger} className="pill model-trigger" aria-label="选择模型" aria-haspopup="menu" aria-expanded={open} aria-controls={menu} onClick={() => setOpen(!open)}>
        <span>模型</span>
        <span className="model-current">{selected ? modelLabel(selected) : "选择模型"}</span>
        {selected && <i className={"model-dot " + (selected.available ? "online" : "offline")} />}
        <span aria-hidden="true">⌄</span>
      </button>
      {open && (
        <div className="model-menu" id={menu} role="menu" aria-label="模型列表">
          <div className="model-menu-title">选择导航规划模型</div>
          {services.map((service) => (
            <button type="button" key={service.id} role="menuitemradio" aria-checked={value === service.id} disabled={!service.available} onClick={() => {
              onChange(service.id);
              setOpen(false);
              trigger.current?.focus();
            }}>
              <i className={"model-dot " + (service.available ? "online" : "offline")} />
              <span className="model-option-text">
                <strong>{modelLabel(service)}</strong>
                <small>{service.available ? service.spec.name : service.unavailable_reason}</small>
              </span>
              <span className="model-check" aria-hidden="true">{value === service.id ? "✓" : ""}</span>
            </button>
          ))}
          {!services.length && <p className="muted">在 Workbench 注册模型后即可选择。</p>}
        </div>
      )}
    </div>
  );
}
