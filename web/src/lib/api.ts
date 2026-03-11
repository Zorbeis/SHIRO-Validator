import type { ConjunctionLatestResponse } from "./types";

export async function fetchLatestConjunction(): Promise<ConjunctionLatestResponse> {
  const base = (import.meta.env.VITE_API_BASE as string | undefined)?.trim() || "";
  const url = `${base}/api/conjunction/latest`;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<ConjunctionLatestResponse>;
}
