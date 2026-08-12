import "server-only";
import { createClient as createSupabaseClient } from "@supabase/supabase-js";
import type { Database } from "@/lib/types";

/**
 * SERVICE-ROLE client. Bypasses Row Level Security entirely. Only ever import
 * this in trusted server-side code (cron jobs, the moderation bot, and server
 * actions that have ALREADY verified the caller's authority). The
 * `server-only` import guarantees a build error if this leaks into a client
 * bundle.
 *
 * Prefer the RLS-scoped clients (server.ts / client.ts). Reach for this only
 * when you genuinely need to bypass a user's RLS — e.g. writing notifications
 * for another user, or the bot reading all content.
 */
export function createAdminClient() {
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!key) {
    throw new Error("SUPABASE_SERVICE_ROLE_KEY is not set");
  }
  return createSupabaseClient<Database>(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    key,
    { auth: { persistSession: false, autoRefreshToken: false } },
  );
}
