"use server";

import { revalidatePath } from "next/cache";
import { createClient } from "@/lib/supabase/server";
import { getViewer } from "@/lib/auth";
import { fail, type ActionResult } from "./_helpers";

/** Toggle the viewer's like on a post. Returns the new liked state. */
export async function toggleLike(postId: string): Promise<ActionResult<{ liked: boolean }>> {
  const viewer = await getViewer();
  if (!viewer) return fail("Please sign in.");
  if (viewer.profile.is_banned) return fail("Your account is restricted.");

  const supabase = createClient();
  const { data: existing } = await supabase
    .from("post_likes")
    .select("post_id")
    .eq("post_id", postId)
    .eq("user_id", viewer.id)
    .maybeSingle();

  if (existing) {
    const { error } = await supabase
      .from("post_likes")
      .delete()
      .eq("post_id", postId)
      .eq("user_id", viewer.id);
    if (error) return fail(error.message);
    revalidatePath(`/posts/${postId}`);
    return { ok: true, liked: false };
  }

  const { error } = await supabase.from("post_likes").insert({ post_id: postId, user_id: viewer.id });
  if (error) return fail(error.message);
  revalidatePath(`/posts/${postId}`);
  return { ok: true, liked: true };
}
