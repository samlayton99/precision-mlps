"use server";

import { revalidatePath } from "next/cache";
import { createClient } from "@/lib/supabase/server";
import { getViewer } from "@/lib/auth";
import { boardSchema, banSchema } from "@/lib/validation";
import { z } from "zod";
import { fail, type ActionResult } from "./_helpers";

/**
 * Admin actions. Each one re-checks admin status in the app layer AND relies on
 * the SECURITY DEFINER RPC to re-check authority in the database — two gates.
 * The RPCs also write the audit_log.
 */
type Guard = { ok: false; error: string } | { ok: true };

async function guardAdmin(): Promise<Guard> {
  const viewer = await getViewer();
  if (!viewer) return { ok: false, error: "Please sign in." };
  if (!viewer.isAdmin) return { ok: false, error: "Admin privilege required." };
  return { ok: true };
}

// --- Roster ----------------------------------------------------------------
export async function inviteAdminByEmail(email: string): Promise<ActionResult<{ status: string }>> {
  const g = await guardAdmin();
  if (!g.ok) return fail(g.error);
  const parsed = z.string().email().safeParse(email.trim().toLowerCase());
  if (!parsed.success) return fail("Enter a valid email address.");
  const supabase = createClient();
  const { data, error } = await supabase.rpc("invite_admin_by_email", { p_email: parsed.data });
  if (error) return fail(error.message);
  revalidatePath("/admin/admins");
  const status = (data as { status?: string } | null)?.status ?? "ok";
  return { ok: true, status };
}

export async function selfRevokeAdmin(): Promise<ActionResult> {
  const g = await guardAdmin();
  if (!g.ok) return fail(g.error);
  const supabase = createClient();
  const { error } = await supabase.rpc("self_revoke_admin");
  if (error) return fail(error.message);
  revalidatePath("/admin/admins");
  return { ok: true };
}

export async function requestAdminRemoval(
  targetId: string,
  reason?: string,
): Promise<ActionResult<{ removed: boolean; votes: number; required: number }>> {
  const g = await guardAdmin();
  if (!g.ok) return fail(g.error);
  const supabase = createClient();
  const { data, error } = await supabase.rpc("request_admin_removal", {
    p_target: targetId,
    p_reason: reason ?? undefined,
  });
  if (error) return fail(error.message);
  revalidatePath("/admin/admins");
  const r = data as { removed: boolean; votes: number; required: number };
  return { ok: true, ...r };
}

export async function cancelAdminRemoval(targetId: string): Promise<ActionResult> {
  const g = await guardAdmin();
  if (!g.ok) return fail(g.error);
  const supabase = createClient();
  const { error } = await supabase.rpc("cancel_admin_removal", { p_target: targetId });
  if (error) return fail(error.message);
  revalidatePath("/admin/admins");
  return { ok: true };
}

// --- Bans ------------------------------------------------------------------
export async function setUserBan(input: unknown): Promise<ActionResult> {
  const g = await guardAdmin();
  if (!g.ok) return fail(g.error);
  const parsed = banSchema.safeParse(input);
  if (!parsed.success) return fail("Invalid ban request.");
  const supabase = createClient();
  const { error } = await supabase.rpc("set_user_ban", {
    p_target: parsed.data.userId,
    p_banned: parsed.data.banned,
    p_reason: parsed.data.reason || undefined,
  });
  if (error) return fail(error.message);
  revalidatePath("/admin/users");
  return { ok: true };
}

// --- Content moderation ----------------------------------------------------
export async function adminDeletePost(postId: string, reason?: string): Promise<ActionResult> {
  const g = await guardAdmin();
  if (!g.ok) return fail(g.error);
  const supabase = createClient();
  const { error } = await supabase.rpc("admin_delete_post", { p_post: postId, p_reason: reason ?? undefined });
  if (error) return fail(error.message);
  revalidatePath("/");
  revalidatePath("/admin/moderation");
  return { ok: true };
}

export async function adminSetPostHidden(
  postId: string,
  hidden: boolean,
  reason?: string,
): Promise<ActionResult> {
  const g = await guardAdmin();
  if (!g.ok) return fail(g.error);
  const supabase = createClient();
  const { error } = await supabase.rpc("admin_set_post_hidden", {
    p_post: postId,
    p_hidden: hidden,
    p_reason: reason ?? undefined,
  });
  if (error) return fail(error.message);
  revalidatePath(`/posts/${postId}`);
  revalidatePath("/admin/moderation");
  return { ok: true };
}

export async function adminDeleteComment(commentId: string, reason?: string): Promise<ActionResult> {
  const g = await guardAdmin();
  if (!g.ok) return fail(g.error);
  const supabase = createClient();
  const { error } = await supabase.rpc("admin_delete_comment", {
    p_comment: commentId,
    p_reason: reason ?? undefined,
  });
  if (error) return fail(error.message);
  revalidatePath("/admin/moderation");
  return { ok: true };
}

export async function adminDeleteChatMessage(messageId: string, reason?: string): Promise<ActionResult> {
  const g = await guardAdmin();
  if (!g.ok) return fail(g.error);
  const supabase = createClient();
  const { error } = await supabase.rpc("admin_delete_chat_message", {
    p_msg: messageId,
    p_reason: reason ?? undefined,
  });
  if (error) return fail(error.message);
  return { ok: true };
}

// --- Boards ----------------------------------------------------------------
export async function createBoard(input: unknown): Promise<ActionResult<{ boardId: string }>> {
  const g = await guardAdmin();
  if (!g.ok) return fail(g.error);
  const parsed = boardSchema.safeParse(input);
  if (!parsed.success) return fail(parsed.error.issues[0]?.message ?? "Invalid board.");
  const supabase = createClient();
  const { data, error } = await supabase.rpc("create_board", {
    p_slug: parsed.data.slug,
    p_name: parsed.data.name,
    p_description: parsed.data.description || undefined,
  });
  if (error) {
    if (error.code === "23505") return fail("That slug is already in use.");
    return fail(error.message);
  }
  revalidatePath("/boards");
  revalidatePath("/admin/boards");
  return { ok: true, boardId: data as string };
}

export async function deleteBoard(boardId: string): Promise<ActionResult> {
  const g = await guardAdmin();
  if (!g.ok) return fail(g.error);
  const supabase = createClient();
  const { error } = await supabase.rpc("delete_board", { p_board: boardId });
  if (error) return fail(error.message);
  revalidatePath("/boards");
  revalidatePath("/admin/boards");
  return { ok: true };
}

// --- Moderation flags ------------------------------------------------------
export async function resolveModerationFlag(flagId: string, status: "dismissed" | "actioned"): Promise<ActionResult> {
  const g = await guardAdmin();
  if (!g.ok) return fail(g.error);
  const supabase = createClient();
  const { error } = await supabase.rpc("resolve_moderation_flag", { p_flag: flagId, p_status: status });
  if (error) return fail(error.message);
  revalidatePath("/admin/moderation");
  return { ok: true };
}
