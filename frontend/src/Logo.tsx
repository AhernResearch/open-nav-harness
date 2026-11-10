// Pixel rover mascot: `#` body (currentColor), `*` beacon (accent), `.` empty.
const GRID = `
......*......
......#......
..#########..
.###########.
.##.......##.
.##.#...#.##.
.##.#...#.##.
.##.......##.
.###########.
..#########..
....#####....
.###########.
#.#.#.#.#.#.#
.###########.`
  .trim()
  .split("\n");

const SIZE = Math.max(GRID.length, ...GRID.map((r) => r.length));
const BEACON = "#ff6b35";

function pixels(body: string): string {
  return GRID.flatMap((row, y) =>
    [...row].flatMap((c, x) =>
      c === "." ? [] : [`<rect x="${x}" y="${y}" width="1.02" height="1.02" fill="${c === "*" ? BEACON : body}"/>`],
    ),
  ).join("");
}

function svg(body: string, extra = ""): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${SIZE} ${SIZE}" shape-rendering="crispEdges">${extra}${pixels(body)}</svg>`;
}

const inline = svg("currentColor");

export function Logo({ size = 26 }: { size?: number }) {
  return (
    <span className="logo-mark" style={{ width: size, height: size }} aria-hidden dangerouslySetInnerHTML={{ __html: inline }} />
  );
}

/** Favicon from the same grid; follows the browser's colour scheme. */
export function installFavicon() {
  const body = `rect:not([fill='${BEACON}'])`;
  const style = `<style>@media(prefers-color-scheme:dark){${body}{fill:#f2f2ee}}</style>`;
  const link = document.querySelector<HTMLLinkElement>("link[rel=icon]") ?? document.head.appendChild(document.createElement("link"));
  link.rel = "icon";
  link.type = "image/svg+xml";
  link.href = "data:image/svg+xml," + encodeURIComponent(svg("#1a1a1a", style));
}
