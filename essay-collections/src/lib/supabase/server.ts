import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";
import type { Database } from "@/lib/types";

/**
 * Server Supabase client bound to the caller's session cookies. Use this in
 * Server Components, Route Handlers, and Server Actions. It runs as the signed
 * in user, so RLS applies exactly as it would for that user.
 */
export function createClient() {
  const cookieStore = cookies();

  return createServerClient<Database>(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options),
            );
          } catch {
            // `set` throws in a Server Component render. The session is still
            // refreshed by middleware, so this is safe to ignore here.
          }
        },
      },
    },
  );
}
