# Deploying Essay Collections

This is a copy-pasteable, in-order walkthrough: Supabase → migrations → Storage → Google OAuth → Vercel → Cron → custom domain → first admin. Do the steps in order; several later steps depend on values (project ref, URLs, secrets) from earlier ones.

You will need:

- A [Supabase](https://supabase.com) account
- A [Google Cloud](https://console.cloud.google.com) account (for OAuth)
- A [Vercel](https://vercel.com) account
- An [Anthropic](https://console.anthropic.com) API key (for the moderation bot)
- The [Supabase CLI](https://supabase.com/docs/guides/cli) installed locally (optional but recommended)

Keep a scratch note open — you will collect the Project Ref, anon key, service-role key, Google client id/secret, and a generated `CRON_SECRET` as you go.

---

## (a) Create a Supabase project

1. In the Supabase dashboard, **New project**. Pick a strong database password and a region close to your users.
2. Wait for provisioning, then open **Project Settings → API** and copy:
   - **Project URL** → `NEXT_PUBLIC_SUPABASE_URL` (looks like `https://<PROJECT-REF>.supabase.co`)
   - **anon public** key → `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - **service_role** key → `SUPABASE_SERVICE_ROLE_KEY` (**server-only — never expose this**)
3. Note the **Project Ref** (the `<PROJECT-REF>` subdomain). You will reuse it for the OAuth redirect URI.

## (b) Run the SQL migrations

The migrations live in `supabase/migrations/` and **must be applied in numeric order**:

| File | What it creates |
| --- | --- |
| `0001_init.sql` | Tables, indexes, maintenance triggers, the 20-admin cap trigger, the new-user profile trigger, and Realtime publication |
| `0002_rls.sql` | Row Level Security — enabled on every table, deny by default |
| `0003_functions.sql` | `SECURITY DEFINER` RPCs: governance, bans, content moderation, boards |
| `0004_seed.sql` | The six starter discussion boards |

### Option 1 — Supabase CLI (recommended)

```bash
supabase login
supabase link --project-ref <PROJECT-REF>
supabase db push          # applies every migration in supabase/migrations, in order
```

(The repo exposes `npm run db:push` for the same thing.)

### Option 2 — SQL editor (fallback)

In the Supabase dashboard, open **SQL Editor → New query**, then paste and **Run** each file, one at a time, in this exact order:

1. `0001_init.sql`
2. `0002_rls.sql`
3. `0003_functions.sql`
4. `0004_seed.sql`

Do not run them out of order — `0002` and `0003` reference objects created in `0001`, and `0004` seeds boards created in `0001`.

**Verify:** open **Table Editor** and confirm the `boards` table has six rows, and **Database → Roles/Policies** shows RLS enabled on every `public` table.

## (c) Create the `essay-media` Storage bucket

Uploaded essay images live in Supabase Storage.

1. **Storage → New bucket**. Name it exactly **`essay-media`**. Enable **Public bucket** (public read) so images render without signed URLs.
2. Add upload/write policies so only signed-in, non-banned users can upload, and each user manages their own objects. In **Storage → Policies** for the `essay-media` bucket, create policies equivalent to:

   ```sql
   -- Public read
   create policy "essay-media public read"
     on storage.objects for select
     using ( bucket_id = 'essay-media' );

   -- Authenticated, non-banned upload
   create policy "essay-media authed insert"
     on storage.objects for insert to authenticated
     with check ( bucket_id = 'essay-media' and public.can_write(auth.uid()) );

   -- Owners manage their own objects
   create policy "essay-media owner update"
     on storage.objects for update to authenticated
     using ( bucket_id = 'essay-media' and owner = auth.uid() );
   create policy "essay-media owner delete"
     on storage.objects for delete to authenticated
     using ( bucket_id = 'essay-media' and owner = auth.uid() );
   ```

   `public.can_write(...)` is defined in `0002_rls.sql` (signed in and not banned).

> The app's image `remotePatterns` in `next.config.mjs` already allow `*.supabase.co`, so bucket URLs render through `next/image` without further config.

## (d) Set up Google OAuth

All authentication is **Google only**, brokered by Supabase Auth.

1. **Google Cloud Console → APIs & Services → Credentials → Create credentials → OAuth client ID.**
   - Application type: **Web application**.
   - **Authorized redirect URI** (this is the Supabase callback, not your app URL):
     ```
     https://<PROJECT-REF>.supabase.co/auth/v1/callback
     ```
   - **Authorized JavaScript origins**: your app origins — `http://localhost:3000` for dev and your Vercel/production URL.
2. Copy the generated **Client ID** and **Client secret**.
3. In Supabase: **Authentication → Providers → Google** → enable it and paste the Client ID and Client secret. Save.
4. In Supabase: **Authentication → URL Configuration**:
   - **Site URL**: your production URL (the Vercel domain, or your custom domain once you have one).
   - **Redirect URLs**: add both `http://localhost:3000/**` and your production URL (`https://<your-domain>/**`). These are the URLs Supabase is allowed to send users back to after sign-in.

## (e) Deploy to Vercel

1. **Import** the Git repository into Vercel. It auto-detects Next.js.
2. In **Project Settings → Environment Variables**, set everything from `.env.example`:

   | Variable | Notes |
   | --- | --- |
   | `NEXT_PUBLIC_SUPABASE_URL` | From step (a) |
   | `NEXT_PUBLIC_SUPABASE_ANON_KEY` | From step (a) |
   | `SUPABASE_SERVICE_ROLE_KEY` | From step (a). **Server-only** — do *not* prefix with `NEXT_PUBLIC_` |
   | `NEXT_PUBLIC_SITE_URL` | Your production URL |
   | `NEXT_PUBLIC_SITE_NAME` | Display name (default `Essay Collections`) |
   | `NEXT_PUBLIC_SITE_TAGLINE` | Tagline |
   | `BOOTSTRAP_ADMIN_EMAILS` | Comma-separated Google emails to auto-grant admin — see step (h) |
   | `ANTHROPIC_API_KEY` | **Server-only.** Powers the moderation bot |
   | `MODERATION_MODEL` | Model id the bot uses (a fast, cheap Claude model is fine for triage) |
   | `CRON_SECRET` | A long random string that authorizes the cron endpoint |

3. **Deploy.** Note the assigned `*.vercel.app` URL — the app ships on that default domain until you add a custom one in step (g).
4. After the first deploy, double-check that `NEXT_PUBLIC_SITE_URL` and the Supabase **Redirect URLs** / **Site URL** and the Google **origins** all point at the Vercel URL, so OAuth round-trips succeed.

## (f) The moderation Cron

`vercel.json` declares a daily Cron job:

```json
{
  "crons": [{ "path": "/api/cron/moderation", "schedule": "0 9 * * *" }]
}
```

- It runs **once a day at 09:00 UTC** and hits `GET /api/cron/moderation`.
- **Authorization:** Vercel Cron sends `Authorization: Bearer <CRON_SECRET>`. The route handler must compare that header against `process.env.CRON_SECRET` and reject anything that does not match — this is what stops the public from triggering (or spoofing) a moderation run. Because the value is a shared secret, keep it long and random and store it only in Vercel's env (never `NEXT_PUBLIC_`).
- The job scans the previous 24 hours of essays, comments, and chat, evaluates them against the Community Guidelines using the model in `MODERATION_MODEL`, and writes `moderation_flags` rows (and a `moderation_runs` row for observability). It **flags for admin review only — it does not delete anything.**

> The `/api/cron/moderation` route handler is implemented at `src/app/api/cron/moderation/route.ts`; it enforces exactly this secret contract.

To test on demand, you can invoke the endpoint yourself once it exists:

```bash
curl -H "Authorization: Bearer $CRON_SECRET" https://<your-domain>/api/cron/moderation
```

## (g) Custom domain

The site is fully functional on the default `*.vercel.app` URL; a custom domain is optional polish.

1. **Vercel → Project → Settings → Domains → Add** your domain, and follow the DNS instructions (usually a CNAME or A record at your registrar).
2. Once DNS resolves and the certificate is issued, update every place that hard-codes the URL:
   - `NEXT_PUBLIC_SITE_URL` (Vercel env) → your new domain, then redeploy.
   - Supabase **Authentication → URL Configuration**: update **Site URL** and add the new domain to **Redirect URLs**.
   - Google Cloud **OAuth client**: add the new domain to **Authorized JavaScript origins** (the Supabase `/auth/v1/callback` redirect URI does **not** change — it stays tied to your Project Ref).

## (h) Bootstrap the first admin

There is no admin at first, and admin grants require an existing admin — so the first one is seeded from an env allowlist.

1. Set `BOOTSTRAP_ADMIN_EMAILS` (Vercel env) to a comma-separated list of the Google account emails that should become admins, e.g. `you@example.com,cofounder@example.com`.
2. Each listed person signs in with Google. On sign-in, the app grants them admin **if the 20-admin cap has room** (the grant path is app-side, since it reads the env allowlist; the DB cap trigger still enforces the ceiling).
3. Once your admin team is established, **empty `BOOTSTRAP_ADMIN_EMAILS`** and manage admins in-app (invite by email or grant by user) per [GOVERNANCE.md](GOVERNANCE.md).

> Email invites made *inside* the app to people who have not signed in yet are stored in `admin_invites` and auto-applied on that person's next sign-in via `ensure_admin_from_invite()`.

---

## Scaling notes

The scaffold is built to scale on managed infrastructure without architectural changes:

- **Connection pooling (Supavisor).** Serverless functions open many short-lived connections. Use Supabase's pooled connection string (**Supavisor**, transaction mode) for the app, and reserve the direct connection for migrations. This is the single most important knob as traffic grows.
- **Vercel serverless.** Server Components, server actions, and the cron route are stateless and scale horizontally. Keep the **service-role** client confined to trusted server code (cron, verified-authority actions) — never the client bundle.
- **Indexes.** The hot read paths are already indexed in `0001_init.sql` — board timelines (`posts_board_created_idx`), the published feed (partial `posts_published_idx`), comment threads, chat history, notifications, mentions, and the moderation queue. Denormalized `like_count` / `comment_count` (kept fresh by triggers) avoid `COUNT(*)` on every list render.
- **Realtime.** `board_chat_messages` and `comments` are added to the `supabase_realtime` publication, so chat and live comments stream over a websocket. Realtime respects RLS — clients only receive rows they are allowed to read.
- **Rate limiting.** The in-memory limiter (`src/lib/rate-limit.ts`) is per-instance. For hard, cross-instance guarantees at scale, swap its store for Upstash Redis or a Postgres RPC — see [SECURITY.md](SECURITY.md). Call sites only use `checkRateLimit`, so nothing else changes.
