/** @type {import('next').NextConfig} */

// Security headers applied to every response. The Content-Security-Policy is
// intentionally strict; loosen it deliberately (and document why) rather than
// by reflex. `img-src` and `frame-src` allow the media hosts we embed.
const cspDirectives = [
  "default-src 'self'",
  // Next.js requires 'unsafe-inline' for its inline bootstrap; in production
  // consider a nonce-based CSP via middleware if you want to remove it.
  "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob: https:",
  "font-src 'self' data:",
  // Supabase (API + realtime websocket) and Google avatar host.
  "connect-src 'self' https://*.supabase.co wss://*.supabase.co",
  // YouTube embeds only (essays can embed video). Add hosts here as needed.
  "frame-src 'self' https://www.youtube.com https://www.youtube-nocookie.com",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
  "upgrade-insecure-requests",
].join("; ");

const securityHeaders = [
  { key: "Content-Security-Policy", value: cspDirectives },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
  {
    key: "Strict-Transport-Security",
    value: "max-age=63072000; includeSubDomains; preload",
  },
];

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  images: {
    remotePatterns: [
      // Google profile pictures.
      { protocol: "https", hostname: "lh3.googleusercontent.com" },
      // Supabase Storage (user-uploaded essay images). Replace the project ref
      // or use a wildcard; env-driven hostname is resolved at build time below.
      { protocol: "https", hostname: "*.supabase.co" },
    ],
  },
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;
