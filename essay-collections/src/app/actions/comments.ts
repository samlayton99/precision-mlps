"use server";

import { revalidatePath } from "next/cache";
import { createClient } from "@/lib/supabase/server";
import { getViewer } from "@/lib/auth";
import { commentSchema } from "@/lib/validation";
import { checkRateLimit, RATE_LIMITS } from "@/lib/rate-limit";
import { createMentions, notifyPostAuthorOfComment, fail, type ActionResult } from "./_helpers";

export async function addComment(input: unknown): Promise<ActionResult<{ commentId: string }>> {
  const viewer = await getViewer();
  if (!viewer) return fail("Please sign in.");
  if (viewer.profile.is_banned) return fail("Your account is restricted.");

  const parsed = commentSchema.safeParse(input);
  if (!parsed.success) return fail(parsed.error.issues[0]?.message ?? "Invalid comment.");

  const rl = checkRateLimit(`comment:${viewer.id}`, RATE_LIMITS.comment.limit, RATE_LIMITS.comment.windowMs);
  if (!rl.ok) return fail("You're commenting too quickly. Please wait a moment.");

  const supabase = createClient();
  const { data, error } = await supabase
    .from("comments")
    .insert({ post_id: parsed.data.postId, author_id: viewer.id, body: parsed.data.body })
    .select("id")
    .single();
  if (error || !data) return fail(error?.message ?? "Could not post the comment.");

  // Notify @mentions and the post author.
  const { data: post } = await supabase
    .from("posts")
    .select("author_id")
    .eq("id", parsed.data.postId)
    .single();

  await createMentions({
    sourceType: "comment",
    sourceId: data.id,
    authorId: viewer.id,
    text: parsed.data.body,
    isHtml: false,
    postId: parsed.data.postId,
    commentId: data.id,
    bodyPreview: `${viewer.profile.display_name} mentioned you: ${parsed.data.body}`,
  });
  if (post) {
    await notifyPostAuthorOfComment({
      postId: parsed.data.postId,
      postAuthorId: post.author_id,
      commentId: data.id,
      actorId: viewer.id,
      preview: `${viewer.profile.display_name} commented: ${parsed.data.body}`,
    });
  }

  revalidatePath(`/posts/${parsed.data.postId}`);
  return { ok: true, commentId: data.id };
}

export async function deleteOwnComment(commentId: string, postId: string): Promise<ActionResult> {
  const viewer = await getViewer();
  if (!viewer) return fail("Please sign in.");
  const supabase = createClient();
  const { error } = await supabase
    .from("comments")
    .delete()
    .eq("id", commentId)
    .eq("author_id", viewer.id);
  if (error) return fail(error.message);
  revalidatePath(`/posts/${postId}`);
  return { ok: true };
}
