import "server-only";
import { createAdminClient } from "@/lib/supabase/admin";
import { extractHandles } from "@/lib/mentions";

export type ActionResult<T = unknown> =
  | ({ ok: true } & T)
  | { ok: false; error: string };

export function fail(error: string): { ok: false; error: string } {
  return { ok: false, error };
}

/**
 * Turn @handles inside a post/comment into mention rows + notifications for the
 * mentioned users. Uses the service-role client because it writes rows owned by
 * OTHER users (their notifications), which RLS would otherwise forbid. The
 * author has already been authenticated by the caller.
 */
export async function createMentions(opts: {
  sourceType: "post" | "comment";
  sourceId: string;
  authorId: string;
  text: string;
  isHtml: boolean;
  postId: string;
  commentId?: string;
  bodyPreview: string;
}): Promise<void> {
  const handles = extractHandles(opts.text, opts.isHtml);
  if (handles.length === 0) return;

  const admin = createAdminClient();
  const { data: users } = await admin
    .from("profiles")
    .select("id, handle")
    .in("handle", handles);

  const targets = (users ?? []).filter((u) => u.id !== opts.authorId);
  if (targets.length === 0) return;

  await admin.from("mentions").upsert(
    targets.map((u) => ({
      source_type: opts.sourceType,
      source_id: opts.sourceId,
      mentioned_user_id: u.id,
      author_id: opts.authorId,
    })),
    { onConflict: "source_type,source_id,mentioned_user_id", ignoreDuplicates: true },
  );

  await admin.from("notifications").insert(
    targets.map((u) => ({
      user_id: u.id,
      type: "mention" as const,
      actor_id: opts.authorId,
      post_id: opts.postId,
      comment_id: opts.commentId ?? null,
      body: opts.bodyPreview.slice(0, 200),
    })),
  );
}

/** Notify a post's author that someone commented (unless it's their own). */
export async function notifyPostAuthorOfComment(opts: {
  postId: string;
  postAuthorId: string;
  commentId: string;
  actorId: string;
  preview: string;
}): Promise<void> {
  if (opts.postAuthorId === opts.actorId) return;
  const admin = createAdminClient();
  await admin.from("notifications").insert({
    user_id: opts.postAuthorId,
    type: "comment",
    actor_id: opts.actorId,
    post_id: opts.postId,
    comment_id: opts.commentId,
    body: opts.preview.slice(0, 200),
  });
}
