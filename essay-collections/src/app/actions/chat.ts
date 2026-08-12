"use server";

import { createClient } from "@/lib/supabase/server";
import { getViewer } from "@/lib/auth";
import { chatSchema } from "@/lib/validation";
import { checkRateLimit, RATE_LIMITS } from "@/lib/rate-limit";
import { fail, type ActionResult } from "./_helpers";

/**
 * Send a board chat message. The realtime subscription in <BoardChat> delivers
 * it to everyone; we don't revalidate the page.
 */
export async function sendChatMessage(input: unknown): Promise<ActionResult> {
  const viewer = await getViewer();
  if (!viewer) return fail("Please sign in.");
  if (viewer.profile.is_banned) return fail("Your account is restricted.");

  const parsed = chatSchema.safeParse(input);
  if (!parsed.success) return fail("Message must be 1–2000 characters.");

  const rl = checkRateLimit(`chat:${viewer.id}`, RATE_LIMITS.chat.limit, RATE_LIMITS.chat.windowMs);
  if (!rl.ok) return fail("Slow down a little.");

  const supabase = createClient();
  const { error } = await supabase
    .from("board_chat_messages")
    .insert({ board_id: parsed.data.boardId, author_id: viewer.id, body: parsed.data.body });
  if (error) return fail(error.message);
  return { ok: true };
}

export async function deleteOwnChatMessage(messageId: string): Promise<ActionResult> {
  const viewer = await getViewer();
  if (!viewer) return fail("Please sign in.");
  const supabase = createClient();
  const { error } = await supabase
    .from("board_chat_messages")
    .delete()
    .eq("id", messageId)
    .eq("author_id", viewer.id);
  if (error) return fail(error.message);
  return { ok: true };
}
