import { type NextRequest } from "next/server";
import { updateSession } from "@/lib/supabase/middleware";

/**
 * Runs on every matched request to keep the Supabase auth session fresh
 * (refresh tokens are rotated in cookies). Auth cookies are httpOnly and
 * managed server-side; the client never touches the refresh token.
 */
export async function middleware(request: NextRequest) {
  return await updateSession(request);
}

export const config = {
  matcher: [
    /*
     * Match all request paths except static assets and image optimization,
     * so we don't burn work refreshing sessions on asset requests.
     */
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
