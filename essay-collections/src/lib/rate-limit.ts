import "server-only";

/**
 * Lightweight fixed-window rate limiter.
 *
 * The default implementation is in-memory, which is per-serverless-instance
 * (not globally shared). That is enough to blunt accidental floods and casual
 * abuse for a small-to-medium community. For hard, cross-instance guarantees at
 * scale, swap the store for Upstash Redis (`@upstash/ratelimit`) or a Postgres
 * RPC — the call sites only use `checkRateLimit`, so nothing else changes.
 */
type Bucket = { count: number; resetAt: number };
const store = new Map<string, Bucket>();

export interface RateLimitResult {
  ok: boolean;
  remaining: number;
  resetAt: number;
}

/**
 * @param key    stable identity, e.g. `comment:<userId>`
 * @param limit  max actions per window
 * @param windowMs  window length in ms
 */
export function checkRateLimit(key: string, limit: number, windowMs: number): RateLimitResult {
  const now = Date.now();
  const existing = store.get(key);

  if (!existing || existing.resetAt <= now) {
    const resetAt = now + windowMs;
    store.set(key, { count: 1, resetAt });
    return { ok: true, remaining: limit - 1, resetAt };
  }

  if (existing.count >= limit) {
    return { ok: false, remaining: 0, resetAt: existing.resetAt };
  }

  existing.count += 1;
  return { ok: true, remaining: limit - existing.count, resetAt: existing.resetAt };
}

// Periodically evict expired buckets so the map cannot grow unbounded.
if (typeof setInterval !== "undefined") {
  const timer = setInterval(() => {
    const now = Date.now();
    for (const [k, v] of store) if (v.resetAt <= now) store.delete(k);
  }, 60_000);
  // Do not keep the process alive just for cleanup.
  (timer as { unref?: () => void }).unref?.();
}

/** Sensible defaults per action. Tune in one place. */
export const RATE_LIMITS = {
  post: { limit: 10, windowMs: 60 * 60 * 1000 }, // 10 essays / hour
  comment: { limit: 20, windowMs: 5 * 60 * 1000 }, // 20 comments / 5 min
  chat: { limit: 30, windowMs: 60 * 1000 }, // 30 chat msgs / min
  report: { limit: 20, windowMs: 60 * 60 * 1000 },
} as const;
