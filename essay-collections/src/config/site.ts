/**
 * Single source of naming. "Essay Collections" is deliberately a description,
 * not a brand — this is a collection of essays worth reading, not a community
 * to join. To change the wording site-wide, change these values (or the
 * NEXT_PUBLIC_SITE_* env vars they read); every page title, header, and footer
 * derives from here.
 */
export const siteConfig = {
  name: process.env.NEXT_PUBLIC_SITE_NAME ?? "Essay Collections",
  tagline:
    process.env.NEXT_PUBLIC_SITE_TAGLINE ??
    "A collection of essays by faithful Latter-day Saints.",
  // Short description used for <meta> and the home hero.
  description:
    "Thoughtful, faithful essays organized into discussion boards — written to " +
    "help each other come closer to Christ and build Zion.",
  url: process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000",
  // What a top-level bucket of essays is called. Change once here to rename
  // "boards" to "collections", "shelves", etc. across the UI.
  boardNounSingular: "board",
  boardNounPlural: "boards",
} as const;

export type SiteConfig = typeof siteConfig;
