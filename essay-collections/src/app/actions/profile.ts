"use server";

import { revalidatePath } from "next/cache";
import { createClient } from "@/lib/supabase/server";
import { getViewer } from "@/lib/auth";
import { profileSchema } from "@/lib/validation";
import { fail, type ActionResult } from "./_helpers";

export async function updateProfile(input: unknown): Promise<ActionResult> {
  const viewer = await getViewer();
  if (!viewer) return fail("Please sign in.");

  const parsed = profileSchema.safeParse(input);
  if (!parsed.success) return fail(parsed.error.issues[0]?.message ?? "Invalid profile.");

  const supabase = createClient();
  const { error } = await supabase
    .from("profiles")
    .update({
      display_name: parsed.data.displayName,
      handle: parsed.data.handle,
      bio: parsed.data.bio || null,
    })
    .eq("id", viewer.id);

  if (error) {
    if (error.code === "23505") return fail("That handle is already taken.");
    return fail(error.message);
  }
  revalidatePath(`/u/${parsed.data.handle}`);
  revalidatePath("/settings");
  return { ok: true };
}
