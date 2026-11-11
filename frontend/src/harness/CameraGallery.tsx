import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { CameraImage } from "./trace";

function caption(text: string) {
  const value = text.replace(/^Tool [^:]+:\s*/, "");
  const direction = value.split(/[\s(—]/)[0].toLowerCase();
  return ({ ahead: "前方", front: "前方", policy_front: "前方（动作相机）", left: "左方",
    right: "右方", back: "后方", behind: "后方" } as Record<string, string>)[direction] ?? value;
}

/** Display the recorded pixels without cropping; inspect them at full size on click. */
export function CameraGallery({ images, className = "camera" }: { images: CameraImage[]; className?: string }) {
  const [selected, setSelected] = useState<CameraImage | null>(null);
  const [small, setSmall] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (selected && !dialog.current?.open) dialog.current?.showModal();
    if (!selected && dialog.current?.open) dialog.current?.close();
  }, [selected]);
  return <>
    <div className={className}>
      {images.map((image, index) => <figure key={`${image.src}-${index}`}>
        <button className="camera-open" aria-label={`放大${caption(image.caption)}图像`} onClick={() => { setSmall(false); setSelected(image); }}>
          <img alt={`${caption(image.caption)}视角`} src={image.src} />
        </button>
        <figcaption>{caption(image.caption)}</figcaption>
      </figure>)}
    </div>
    {createPortal(<dialog className="camera-dialog" ref={dialog} aria-label="相机图像" onCancel={() => setSelected(null)}
      onClick={(event) => { if (event.target === event.currentTarget) setSelected(null); }}>
      <header><strong>{selected && caption(selected.caption)}</strong><button onClick={() => setSelected(null)} aria-label="关闭图像">关闭</button></header>
      {selected && <img src={selected.src} alt={`${caption(selected.caption)}原始图像`}
        onLoad={(event) => setSmall(Math.min(event.currentTarget.naturalWidth, event.currentTarget.naturalHeight) < 256)} />}
      {small && <p>这张历史图像分辨率较低，放大无法恢复被标签遮挡的场景。</p>}
    </dialog>, document.body)}
  </>;
}
