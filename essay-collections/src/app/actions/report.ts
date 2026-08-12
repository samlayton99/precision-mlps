"use server";

import { createClient } from "@/lib/supabase/server";
import { getViewer } from "@/lib/auth";
import { reportSchema } from "@/lib/validation";
import { checkRateLimit, RATE_LIMITS } from "@/lib/rate-limit";
import { fail, type ActionResult } from "./_helpers";

/** A member reports content for admin review. Creates an open moderation flag. */
export async function reportContent(input: unknown): Promise<ActionResult> {
  const viewer = await getViewer();
  if (!viewer) return fail("Please sign in.");
  if (viewer.profile.is_banned) return fail("Your account is restricted.");

  const parsed = reportSchema.safeParse(input);
  if (!parsed.success) return fail(parsed.error.issues[0]?.message ?? "Invalid report.");

  const rl = checkRateLimit(`report:${viewer.id}`, RATE_LIMITS.report.limit, RATE_LIMITS.report.windowMs);
  if (!rl.ok) return fail("You've filed several reports recently. Please wait.");

  const supabase = createClient();
  const { error } = await supabase.from("moderation_flags").insert({
    content_type: parsed.data.contentType,
    content_id: parsed.data.contentId,
    reason: parsed.data.reason,
    flagged_by: "user",
    reporter_id: viewer.id,
    status: "open",
  });
  if (error) return fail(error.message);
  return { ok: true };
}
