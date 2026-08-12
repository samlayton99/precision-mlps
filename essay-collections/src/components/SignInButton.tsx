"use client";

import { useState } from "react";
import { createClient } from "@/lib/supabase/client";

export function SignInButton({ next = "/" }: { next?: string }) {
  const [loading, setLoading] = useState(false);

  async function signIn() {
    setLoading(true);
    const supabase = createClient();
    const redirectTo = `${window.location.origin}/auth/callback?next=${encodeURIComponent(next)}`;
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo,
        queryParams: { access_type: "offline", prompt: "consent" },
      },
    });
    if (error) {
      setLoading(false);
      alert(`Sign-in failed: ${error.message}`);
    }
    // On success the browser is redirected to Google, so no further UI needed.
  }

  return (
    <button
      onClick={signIn}
      disabled={loading}
      className="inline-flex items-center justify-center gap-3 rounded-full border border-line bg-surface px-6 py-3 font-medium shadow-sm transition hover:bg-brand-soft disabled:opacity-60"
    >
      <GoogleGlyph />
      {loading ? "Redirecting…" : "Continue with Google"}
    </button>
  );
}

function GoogleGlyph() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden>
      <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9 3.6l6.7-6.7C35.6 2.5 30.1 0 24 0 14.6 0 6.4 5.4 2.5 13.3l7.8 6c1.9-5.6 7.1-9.8 13.7-9.8z" />
      <path fill="#4285F4" d="M46.1 24.6c0-1.6-.1-3.1-.4-4.6H24v9.1h12.4c-.5 2.9-2.1 5.4-4.6 7.1l7.1 5.5c4.2-3.9 6.2-9.6 6.2-17.1z" />
      <path fill="#FBBC05" d="M10.3 28.3c-.5-1.4-.7-2.9-.7-4.3s.3-2.9.7-4.3l-7.8-6C.9 16.9 0 20.3 0 24s.9 7.1 2.5 10.3l7.8-6z" />
      <path fill="#34A853" d="M24 48c6.1 0 11.3-2 15-5.5l-7.1-5.5c-2 1.3-4.6 2.1-7.9 2.1-6.6 0-12.2-4.4-14.2-10.4l-7.8 6C6.4 42.6 14.6 48 24 48z" />
    </svg>
  );
}
