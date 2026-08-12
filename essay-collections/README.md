# Essay Collections

**A collection of essays worth reading** — thoughtful, faithful writing by members of The Church of Jesus Christ of Latter-day Saints, organized into discussion boards.

Essay Collections is a LessWrong-style essay forum: long-form, rigorously argued essays that help one another come closer to Christ and build Zion. It is a shared library first and a social app second. There are no followers and no feeds to chase — just essays, the boards they live in, and honest conversation around them.

> The name comes from Isaiah 28:10 — *"line upon line, precept upon precept."* Understanding, and Zion, are built a little at a time.

---

## Features

- **Essays** — a Substack-like rich-text editor (headings, bold/italic, quotes, lists, links, images, and YouTube embeds). Essay HTML is sanitized on the way in *and* on the way out.
- **Discussion boards** — essays are organized into topical boards (Doctrine & Scripture, Faith & Reason, Discipleship, Church History, Science & Faith, Building Zion). Admins can add, rename, archive, or remove boards.
- **Comments** — text-only discussion under each essay.
- **Likes** — a simple, one-per-person appreciation signal.
- **@mentions** — tag other members in essays and comments; they get a notification.
- **Per-board realtime chat** — each board has its own live, text-only global chat.
- **Daily moderation bot** — a Claude-powered scan of the last 24 hours of content against the [Community Guidelines](docs/COMMUNITY_GUIDELINES.md). It *flags for human review*; it never auto-deletes.
- **Admin governance** — a hard cap of 20 admins, an append-only audit log, and a vote-based process for removing an admin (see [GOVERNANCE.md](docs/GOVERNANCE.md)).

## Tech stack

| Layer | Choice |
| --- | --- |
| Framework | Next.js 14 (App Router) + TypeScript |
| Styling | Tailwind CSS |
| Editor | Tiptap (ProseMirror) |
| Backend | Supabase — Postgres, Auth, Realtime, Storage |
| Auth | **Google OAuth only**, via Supabase Auth |
| Validation | Zod on every mutation |
| Sanitization | DOMPurify (allowlisted tags + YouTube-only iframes) |
| Moderation | Anthropic Claude (`@anthropic-ai/sdk`) |
| Hosting | Vercel (serverless + Cron) |

Security is defense-in-depth: Row Level Security on every table (deny by default), `SECURITY DEFINER` RPCs for every privileged action, a strict Content-Security-Policy, JWT-verified auth, and server-side rate limiting. See [SECURITY.md](docs/SECURITY.md).

## Quickstart (local development)

```bash
# 1. Clone
git clone <your-repo-url> essay-collections
cd essay-collections

# 2. Install dependencies
npm install

# 3. Configure environment
cp .env.example .env.local
# Then edit .env.local with your Supabase + Google + Anthropic values.

# 4. Apply the database migrations to your Supabase project
#    (requires the Supabase CLI, logged in and linked to your project)
npm run db:push
# ...or paste supabase/migrations/0001–0004 into the Supabase SQL editor in order.

# 5. Run the dev server
npm run dev
# http://localhost:3000
```

You will need a Supabase project and a Google OAuth client before sign-in works end to end. The full walkthrough is in [docs/DEPLOY.md](docs/DEPLOY.md).

### Useful scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | Start the local dev server |
| `npm run build` | Production build |
| `npm run start` | Serve the production build |
| `npm run typecheck` | `tsc --noEmit` — type safety, no output |
| `npm run lint` | ESLint (Next.js config) |
| `npm run db:push` | `supabase db push` — apply migrations |
| `npm run db:reset` | `supabase db reset` — rebuild the local DB from migrations |

## Documentation

| Doc | Purpose |
| --- | --- |
| [docs/DEPLOY.md](docs/DEPLOY.md) | Step-by-step Supabase + Google + Vercel setup and go-live |
| [docs/SECURITY.md](docs/SECURITY.md) | The security model in depth, plus a pre-launch hardening checklist |
| [docs/GOVERNANCE.md](docs/GOVERNANCE.md) | Admin roles, the 20-admin cap, and the removal-vote rules |
| [docs/COMMUNITY_GUIDELINES.md](docs/COMMUNITY_GUIDELINES.md) | Community standard and the moderation-bot rubric |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Codebase map, data model, and request/auth flow |
| [docs/RENAMING.md](docs/RENAMING.md) | Rebranding the site (name, tagline, and the word "board") in one place |

## Branding

The site name, tagline, and the noun used for a "board" all come from one place: `src/config/site.ts`, backed by `NEXT_PUBLIC_SITE_*` env vars. Renaming the whole site is a one-place change — see [RENAMING.md](docs/RENAMING.md).

## Project status

This is a **complete application**: the data model, security policies, governance RPCs, the full App Router UI (pages, server actions, the editor, board chat, admin dashboard), and the daily moderation cron (`/api/cron/moderation`) are all implemented. `npm run typecheck` and `npm run build` both pass cleanly.

What it has **not** done yet is run against a live backend — it needs a Supabase project, Google OAuth credentials, and a Vercel deployment wired up per [docs/DEPLOY.md](docs/DEPLOY.md) before the first real sign-in. Expect the usual round of small integration fixes on first deploy (redirect URLs, storage-bucket policies, env values).

## License

Add a license of your choice before making the repository public.
