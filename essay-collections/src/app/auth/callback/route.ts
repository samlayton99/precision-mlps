import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";
import { createAdminClient } from "@/lib/supabase/admin";

/**
 * Google OAuth callback. Supabase redirects here with a `code`; we exchange it
 * for a session, then (a) accept any pending admin invite and (b) bootstrap
 * admins listed in BOOTSTRAP_ADMIN_EMAILS. Both are cap-safe.
 */
export async function GET(request: Request) {
  const url = new URL(request.url);
  const code = url.searchParams.get("code");
  const next = safeNext(url.searchParams.get("next"));

  if (!code) {
    return NextResponse.redirect(new URL("/login?error=missing_code", url.origin));
  }

  const supabase = createClient();
  const { error } = await supabase.auth.exchangeCodeForSession(code);
  if (error) {
    return NextResponse.redirect(new URL("/login?error=auth", url.origin));
  }

  // Best-effort admin resolution; never block sign-in on failure.
  try {
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (user) {
      // Accept a pending email invite (SECURITY DEFINER, uses auth.uid()).
      await supabase.rpc("ensure_admin_from_invite");

      // Bootstrap the founding admins from the env allowlist.
      const allow = (process.env.BOOTSTRAP_ADMIN_EMAILS ?? "")
        .split(",")
        .map((e) => e.trim().toLowerCase())
        .filter(Boolean);
      if (user.email && allow.includes(user.email.toLowerCase())) {
        const admin = createAdminClient();
        const { count } = await admin
          .from("admins")
          .select("user_id", { count: "exact", head: true });
        if ((count ?? 0) < 20) {
          // Trigger re-checks the cap; ignore duplicate/cap errors.
          await admin.from("admins").upsert({ user_id: user.id }, { onConflict: "user_id" });
        }
      }
    }
  } catch {
    // swallow — sign-in already succeeded
  }

  return NextResponse.redirect(new URL(next, url.origin));
}

/** Only allow same-origin relative redirects. */
function safeNext(next: string | null): string {
  if (!next || !next.startsWith("/") || next.startsWith("//")) return "/";
  return next;
}
