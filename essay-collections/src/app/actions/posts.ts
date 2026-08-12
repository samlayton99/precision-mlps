"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { getViewer } from "@/lib/auth";
import { postSchema } from "@/lib/validation";
import { sanitizeEssayHtml, makeExcerpt } from "@/lib/sanitize";
import { checkRateLimit, RATE_LIMITS } from "@/lib/rate-limit";
import { createMentions, fail, type ActionResult } from "./_helpers";

export async function createPost(input: unknown): Promise<ActionResult<{ postId: string }>> {
  const viewer = await getViewer();
  if (!viewer) return fail("Please sign in.");
  if (viewer.profile.is_banned) return fail("Your account is restricted.");

  const parsed = postSchema.safeParse(input);
  if (!parsed.success) return fail(parsed.error.issues[0]?.message ?? "Invalid essay.");

  const rl = checkRateLimit(`post:${viewer.id}`, RATE_LIMITS.post.limit, RATE_LIMITS.post.windowMs);
  if (!rl.ok) return fail("You're posting too quickly. Please wait a bit.");

  const clean = sanitizeEssayHtml(parsed.data.contentHtml);
  const supabase = createClient();
  const { data, error } = await supabase
    .from("posts")
    .insert({
      board_id: parsed.data.boardId,
      author_id: viewer.id,
      title: parsed.data.title,
      subtitle: parsed.data.subtitle || null,
      content_html: clean,
      excerpt: makeExcerpt(clean),
      cover_image_url: parsed.data.coverImageUrl || null,
      status: parsed.data.status,
    })
    .select("id")
    .single();

  if (error || !data) return fail(error?.message ?? "Could not save the essay.");

  await createMentions({
    sourceType: "post",
    sourceId: data.id,
    authorId: viewer.id,
    text: clean,
    isHtml: true,
    postId: data.id,
    bodyPreview: `${viewer.profile.display_name} mentioned you in “${parsed.data.title}”`,
  });

  revalidatePath("/");
  revalidatePath("/boards");
  return { ok: true, postId: data.id };
}

export async function updatePost(
  postId: string,
  input: unknown,
): Promise<ActionResult<{ postId: string }>> {
  const viewer = await getViewer();
  if (!viewer) return fail("Please sign in.");
  if (viewer.profile.is_banned) return fail("Your account is restricted.");

  const parsed = postSchema.safeParse(input);
  if (!parsed.success) return fail(parsed.error.issues[0]?.message ?? "Invalid essay.");

  const clean = sanitizeEssayHtml(parsed.data.contentHtml);
  const supabase = createClient();
  // RLS guarantees only the author (non-banned) can update; we also scope by id.
  const { error } = await supabase
    .from("posts")
    .update({
      board_id: parsed.data.boardId,
      title: parsed.data.title,
      subtitle: parsed.data.subtitle || null,
      content_html: clean,
      excerpt: makeExcerpt(clean),
      cover_image_url: parsed.data.coverImageUrl || null,
      status: parsed.data.status,
    })
    .eq("id", postId)
    .eq("author_id", viewer.id);

  if (error) return fail(error.message);
  revalidatePath(`/posts/${postId}`);
  return { ok: true, postId };
}

/** Author deletes their own essay. */
export async function deleteOwnPost(postId: string): Promise<ActionResult> {
  const viewer = await getViewer();
  if (!viewer) return fail("Please sign in.");
  const supabase = createClient();
  const { error } = await supabase.from("posts").delete().eq("id", postId).eq("author_id", viewer.id);
  if (error) return fail(error.message);
  revalidatePath("/");
  redirect("/");
}
