# Renaming & Rebranding

The site's identity is deliberately centralized so you can rebrand in one place. The name, tagline, description, canonical URL, and even the word "board" all flow from **`src/config/site.ts`**, which reads `NEXT_PUBLIC_SITE_*` environment variables with sensible fallbacks.

```ts
// src/config/site.ts
export const siteConfig = {
  name:    process.env.NEXT_PUBLIC_SITE_NAME    ?? "Essay Collections",
  tagline: process.env.NEXT_PUBLIC_SITE_TAGLINE ?? "A collection of essays by faithful Latter-day Saints.",
  description: "Thoughtful, faithful essays organized into discussion boards — written to help each other come closer to Christ and build Zion.",
  url:     process.env.NEXT_PUBLIC_SITE_URL     ?? "http://localhost:3000",
  boardNounSingular: "board",
  boardNounPlural:   "boards",
} as const;
```

Every page title, header, and footer derives from `siteConfig`, so you never hunt through components.

## Rename the site (name & tagline)

Preferred — set the env vars (no code change), in `.env.local` for dev and in the Vercel project settings for production:

```bash
NEXT_PUBLIC_SITE_NAME="Your New Name"
NEXT_PUBLIC_SITE_TAGLINE="Your new tagline."
```

Because these are `NEXT_PUBLIC_*`, they are baked in at **build time** — redeploy (or restart `npm run dev`) after changing them.

Alternatively, change the **fallback defaults** directly in `src/config/site.ts` (the `?? "..."` values). Do this if you want a new default that doesn't depend on env being set. Update `description` there too — it's the one-line framing used for `<meta>` tags and the home hero.

> Keep `NEXT_PUBLIC_SITE_URL` in sync as well when the domain changes — it feeds OAuth redirects and absolute links. See [DEPLOY.md](DEPLOY.md) step (g) for the full domain-change checklist (env + Supabase redirect URLs + Google origins).

## Rename the word "board"

If "board" isn't your preferred term (say you want "Collections" or "Shelves"), change the two noun fields in `src/config/site.ts`:

```ts
boardNounSingular: "collection",
boardNounPlural:   "collections",
```

The UI reads these wherever it labels a board, so the change propagates across the interface. Note this only changes the **display label** — the database table is still `boards`, the URL slug is still `/boards/...` unless you also rename routes, and the seeded rows are unchanged. Renaming those is optional and larger; the noun fields cover the user-facing wording.

## Rename the starter boards (content)

The six starter boards are seeded in **`supabase/migrations/0004_seed.sql`** (Doctrine & Scripture, Faith & Reason, Discipleship & Christlike Living, Church History, Science & Faith, Building Zion). Two ways to change them:

- **After launch (recommended):** an admin renames, adds, archives, or deletes boards from the admin dashboard — no migration needed. These flow through the `create_board` / `delete_board` RPCs.
- **Before first run:** edit `0004_seed.sql` (each row is `(slug, name, description, sort_order)`) so a fresh database seeds your boards. The insert is `on conflict (slug) do nothing`, so editing it will not overwrite boards that already exist in a live database.

## Rename the repository / project folder

1. Rename the folder on disk and update your Git remote as usual.
2. Update the `"name"` and `"description"` fields in `package.json` if you want them to match.
3. Nothing else references the folder name — the TypeScript path alias is `@/*` → `./src/*` (see `tsconfig.json`), which is independent of the directory name.

## Quick checklist

- [ ] `NEXT_PUBLIC_SITE_NAME` / `NEXT_PUBLIC_SITE_TAGLINE` set in `.env.local` **and** Vercel
- [ ] `src/config/site.ts` fallback defaults + `description` updated (if not relying solely on env)
- [ ] `boardNounSingular` / `boardNounPlural` changed if renaming "board"
- [ ] `NEXT_PUBLIC_SITE_URL` + Supabase/Google URLs updated if the domain changed
- [ ] Starter boards edited in `0004_seed.sql` (pre-launch) or via the admin dashboard (post-launch)
- [ ] `package.json` name/description updated if you renamed the repo
- [ ] Rebuild / redeploy so the `NEXT_PUBLIC_*` values take effect
