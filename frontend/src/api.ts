import type { components } from "./api-schema";
export type ServiceSpec = components["schemas"]["ServiceSpec"];
export type Row = Record<string, any>;
export async function api<T = Row>(path: string, body?: unknown): Promise<T> {
  const response = await fetch("/api" + path, {
    method: body === undefined ? "GET" : "POST",
    credentials: "same-origin",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let message;
    try {
      message = (await response.json()).detail;
    } catch {
      message = response.statusText;
    }
    throw new Error(
      typeof message === "string" ? message : JSON.stringify(message),
    );
  }
  return response.json();
}
