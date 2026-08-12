# Security Model

Essay Collections is built defense-in-depth: no single control is trusted alone. The database enforces access even if application code has a bug; the application enforces policy even if a client is hostile; and the transport layer is locked down by headers. This document describes each layer, references the file that implements it, and ends with a pre-launch hardening checklist.

---

## The layers at a glance

| Layer | Mechanism | Where |
| --- | --- | --- |
| Database access | Row Level Security, deny by default | `supabase/migrations/0002_rls.sql` |
| Privileged actions | `SECURITY DEFINER` RPCs that re-check authority + audit | `supabase/migrations/0003_functions.sql` |
| Application authz | JWT-verified `getUser()`, admin/ban gates | `src/lib/auth.ts` |
| Input validation | Zod schemas on every mutation | `src/lib/validation.ts` |
| XSS defense | DOMPurify allowlist, sanitize on store *and* render | `src/lib/sanitize.ts` |
| Rate limiting | Fixed-window limiter | `src/lib/rate-limit.ts` |
| Transport / browser | Strict CSP + security headers | `next.config.mjs` |
| Secrets | Service-role key server-only | `src/lib/supabase/admin.ts`, `.env.example` |

## 1. Row Level Security (the last line of defense)

Every table in the `public` schema has RLS **enabled**, and the default posture is **deny** — a table with RLS on and no matching policy returns nothing and accepts nothing. Policies then open specific doors. Highlights from `0002_rls.sql`:

- **Reads.** Profiles, boards, the admin roster, and likes are world-readable (governance transparency and public essays). Posts are readable only when `status = 'published'`, or you are the author, or you are an admin. Comments are readable exactly when their parent post is. Notifications and mentions are readable only by their recipient. `audit_log`, `moderation_flags`, `moderation_runs`, and `admin_removal_requests` are **admin-read-only**.
- **Writes.** Users may insert/update/delete only their *own* posts, comments, likes, and chat messages, and only if they can write. Commenting and liking additionally require the target post to be `published`.
- **`can_write(uid)`** gates every write: signed in **and not banned**. Banned users keep read access but cannot create anything.
- **No self-escalation.** The `profiles_update_own` policy explicitly forbids a user from flipping their own `is_banned`. Ban state changes only through the admin RPC.
- **Helper predicates** (`is_admin`, `is_banned`, `can_write`) are `SECURITY DEFINER` so they can read `admins`/`profiles` without tripping those tables' own RLS (and without policy recursion).

Because RLS is the floor, a bug in a page or a forgotten check in a server action cannot leak data the database would refuse to serve.

## 2. Privileged actions: `SECURITY DEFINER` RPCs

All admin power is concentrated in a small, readable set of RPCs in `0003_functions.sql`. Each one:

1. Runs as the table owner (bypassing RLS **on purpose**),
2. **Re-checks the caller's authority itself** via `_require_admin()` (raises if not signed in / not admin), and
3. Writes an **append-only `audit_log` row** via `_audit(...)`.

This covers board create/delete, post delete, post hide/unhide ("pause"), comment delete, chat-message delete, user ban/unban, moderation-flag resolution, and the whole admin-governance surface (grant, invite, self-revoke, removal vote). `anon` cannot call any of them; `authenticated` can, and each function decides for itself whether the caller is allowed. See [GOVERNANCE.md](GOVERNANCE.md) for the governance semantics.

The `audit_log` is insert-only in practice — no RPC updates or deletes it, and its RLS grants admins read but no one write (writes happen inside definer functions). Treat it as the tamper-evident record of who did what.

## 3. Application authorization

- **JWT-verified identity.** `getViewer()` (`src/lib/auth.ts`) always uses `supabase.auth.getUser()`, which verifies the JWT against the auth server, **never** `getSession()`, which merely trusts the cookie. Authorization decisions are made only on the verified user.
- **Gates.** `requireViewer()` redirects anonymous users to `/login` and banned users to `/banned`. `requireAdmin()` builds on it and redirects non-admins away. Server code should call these before any sensitive work.
- **Three-client Supabase pattern** (see [ARCHITECTURE.md](ARCHITECTURE.md)): the browser anon client and the server user-scoped client both run **under the caller's RLS**; the service-role client bypasses RLS and is reserved for trusted paths (cron, the moderation bot, verified-authority server actions).

## 4. Input validation

Every mutation is validated with a **Zod** schema before it touches the database (`src/lib/validation.ts`): posts, comments, chat, boards, profiles, reports, and bans. Schemas trim, bound lengths (mirroring the DB `CHECK` constraints), constrain enums (e.g. a user may only submit `published`/`draft`, never `hidden`), validate UUIDs and URLs, and constrain the board slug to `^[a-z0-9]([a-z0-9-]{0,48}[a-z0-9])?$`. Validation and DB constraints are intentionally redundant — the database rejects bad data even if a call site forgets to validate.

## 5. XSS mitigation

Essays are rich HTML, which is the highest-risk surface. `src/lib/sanitize.ts` uses DOMPurify with a **strict allowlist** and is applied **twice** — when storing and again when rendering (defense against stored XSS):

- **Allowed tags** are a Substack-like formatting set (`p`, `h1`–`h4`, `strong`/`em`/`u`/`s`, `blockquote`, lists, `a`, `img`, `figure`, `pre`/`code`, `span`, and `iframe`). Everything else — including `<script>` — is stripped.
- **iframes are YouTube-only.** An `uponSanitizeElement` hook removes any iframe whose `src` host is not in the YouTube allowlist (`youtube.com` / `youtube-nocookie.com`).
- **Links are hardened.** An `afterSanitizeAttributes` hook forces external links to `target="_blank"` with `rel="noopener noreferrer nofollow"`; images get `loading="lazy"`.
- **URL scheme allowlist.** Only `https:`, `mailto:`, root-relative, and fragment URLs are allowed — `javascript:` and friends are rejected.
- `htmlToText()` provides a tags-stripped projection used for excerpts and as the moderation bot's input.

This pairs with the CSP (below): even if a payload somehow survived, `script-src` and `frame-src` restrict what a browser will execute or embed.

## 6. Content-Security-Policy & security headers

`next.config.mjs` applies these headers to **every** response (`source: "/:path*"`):

- **CSP** — `default-src 'self'`; `object-src 'none'`; `base-uri 'self'`; `form-action 'self'`; `frame-ancestors 'none'`; `upgrade-insecure-requests`. `connect-src` allows Supabase API + realtime websockets (`https://*.supabase.co`, `wss://*.supabase.co`). `frame-src` is **YouTube-only**. `img-src` allows `https:`/`data:`/`blob:` (covers Supabase Storage and Google avatars).
- **`X-Frame-Options: DENY`** and **`frame-ancestors 'none'`** — the site cannot be framed (clickjacking defense).
- **`X-Content-Type-Options: nosniff`**, **`Referrer-Policy: strict-origin-when-cross-origin`**, **`Permissions-Policy`** disabling camera/microphone/geolocation.
- **HSTS** — `max-age=63072000; includeSubDomains; preload`.
- `poweredByHeader: false` removes the `X-Powered-By` fingerprint.

> **Known trade-off:** `script-src` currently includes `'unsafe-inline' 'unsafe-eval'` because Next.js's inline bootstrap needs it. This is the main CSP weakness. For a hardened deployment, move to a **nonce-based CSP** via middleware (the file already flags this in a comment). Tracked in the checklist below.

## 7. Rate limiting

`src/lib/rate-limit.ts` is a fixed-window limiter with sensible per-action defaults (`RATE_LIMITS`): 10 essays/hour, 20 comments/5 min, 30 chat messages/min, 20 reports/hour. It is `server-only` and self-evicts expired buckets.

**Limitation to understand:** the default store is **in-memory, per serverless instance** — not globally shared. It blunts accidental floods and casual abuse but is not a hard, cross-instance guarantee. For strict enforcement at scale, swap the store for **Upstash Redis** (`@upstash/ratelimit`) or a **Postgres RPC**. Call sites only use `checkRateLimit(key, limit, windowMs)`, so the upgrade is localized to this one file.

## 8. Secrets handling

- **`SUPABASE_SERVICE_ROLE_KEY`** bypasses RLS. It is **server-only**, never prefixed with `NEXT_PUBLIC_`, and `src/lib/supabase/admin.ts` imports `"server-only"` so a build **fails** if it is ever pulled into a client bundle.
- **`ANTHROPIC_API_KEY`** and **`CRON_SECRET`** are likewise server-only.
- Only `NEXT_PUBLIC_*` values reach the browser, and those are safe precisely because RLS protects the anon key.
- `.gitignore` excludes every `.env*.local`; only `.env.example` (placeholders) is tracked. Never commit real secrets.

## 9. Authentication & session handling

- **Google OAuth only**, via Supabase Auth — no passwords to store or leak.
- **httpOnly cookies.** The middleware (`src/middleware.ts` → `src/lib/supabase/middleware.ts`) refreshes the session on each matched request and rotates refresh tokens in httpOnly cookies. The client never touches the refresh token.
- **Verified authz.** As in §3, all authorization uses JWT-verified `getUser()`.

## 10. Ban enforcement

Banning is layered: `can_write()` blocks all writes at the **RLS** level; `requireViewer()` redirects banned users to `/banned` at the **app** level; and a user cannot self-unban (the profile update policy freezes `is_banned`). Bans are set only through `set_user_ban()`, which **refuses to ban a current admin** — admin must be removed first (see [GOVERNANCE.md](GOVERNANCE.md)).

---

## Responsible disclosure

If you find a vulnerability, please report it privately to the maintainers (add a security contact / email here before launch) rather than opening a public issue. Give a clear reproduction and reasonable time to remediate before any disclosure.

## Hardening checklist before going public

- [ ] **Confirm RLS is ON for every `public` table** and each has only the intended policies (spot-check in the Supabase dashboard).
- [ ] **Move to a nonce-based CSP** and remove `'unsafe-inline'`/`'unsafe-eval'` from `script-src`.
- [ ] **Upgrade the rate limiter** to Upstash/Postgres so limits hold across serverless instances.
- [ ] **Rotate and scope all keys**; verify `SUPABASE_SERVICE_ROLE_KEY`, `ANTHROPIC_API_KEY`, and `CRON_SECRET` exist only in server env and never in `NEXT_PUBLIC_*`.
- [ ] **Verify the cron endpoint rejects requests without the correct `Authorization: Bearer <CRON_SECRET>`**.
- [ ] **Verify the `essay-media` Storage policies** — public read, authenticated non-banned write, owner-only update/delete.
- [ ] **Confirm sanitization runs on both store and render**, and test a hostile essay payload (`<script>`, `onerror=`, `javascript:` link, non-YouTube iframe) end to end.
- [ ] **Empty `BOOTSTRAP_ADMIN_EMAILS`** once the founding admins are set.
- [ ] **Review the audit log** surfaces every privileged action and is admin-read-only.
- [ ] **Test ban enforcement** (banned user cannot post/comment/like/chat and cannot self-unban) and the admin-before-ban ordering.
- [ ] **Add a security contact** and a privacy note before inviting the public.
- [ ] **Enable Supabase Auth rate limits / bot protection** and review provider settings.
