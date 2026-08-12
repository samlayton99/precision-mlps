import "server-only";
import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import type { Profile } from "@/lib/types";

export interface Viewer {
  id: string;
  email: string | null;
  profile: Profile;
  isAdmin: boolean;
}

/**
 * The current signed-in user with their profile and admin flag, or null.
 * Always uses supabase.auth.getUser() (verifies the JWT with the auth server)
 * rather than getSession() (which trusts the cookie) — important for authz.
 */
export async function getViewer(): Promise<Viewer | null> {
  const supabase = createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return null;

  const [{ data: profile }, { data: adminRow }] = await Promise.all([
    supabase.from("profiles").select("*").eq("id", user.id).single(),
    supabase.from("admins").select("user_id").eq("user_id", user.id).maybeSingle(),
  ]);

  if (!profile) return null; // profile is created by trigger; treat missing as unauthenticated
  return {
    id: user.id,
    email: user.email ?? null,
    profile,
    isAdmin: Boolean(adminRow),
  };
}

/** Require a signed-in, non-banned viewer or redirect to login. */
export async function requireViewer(nextPath?: string): Promise<Viewer> {
  const viewer = await getViewer();
  if (!viewer) {
    redirect(`/login${nextPath ? `?next=${encodeURIComponent(nextPath)}` : ""}`);
  }
  if (viewer.profile.is_banned) {
    redirect("/banned");
  }
  return viewer;
}

/** Require an admin viewer or redirect. */
export async function requireAdmin(): Promise<Viewer> {
  const viewer = await requireViewer();
  if (!viewer.isAdmin) {
    redirect("/");
  }
  return viewer;
}
