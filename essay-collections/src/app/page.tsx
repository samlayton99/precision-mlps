import Link from "next/link";
import { createClient } from "@/lib/supabase/server";
import { siteConfig } from "@/config/site";
import { PostCard, type PostCardData } from "@/components/PostCard";
import type { Board } from "@/lib/types";

export const revalidate = 30;

const POST_SELECT =
  "id,title,subtitle,excerpt,like_count,comment_count,created_at,published_at,status," +
  "author:profiles!posts_author_id_fkey(handle,display_name,avatar_url),board:boards(slug,name)";

export default async function HomePage() {
  const supabase = createClient();

  const [{ data: posts }, { data: boards }] = await Promise.all([
    supabase
      .from("posts")
      .select(POST_SELECT)
      .eq("status", "published")
      .order("published_at", { ascending: false })
      .limit(20)
      .returns<PostCardData[]>(),
    supabase
      .from("boards")
      .select("*")
      .eq("is_archived", false)
      .order("sort_order")
      .returns<Board[]>(),
  ]);

  return (
    <div className="grid grid-cols-1 gap-10 lg:grid-cols-[1fr_260px]">
      <div>
        <section className="mb-8 border-b border-line pb-8">
          <h1 className="font-serif text-3xl font-semibold sm:text-4xl">{siteConfig.name}</h1>
          <p className="mt-3 max-w-2xl text-lg text-muted">{siteConfig.description}</p>
        </section>

        <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-muted">
          Latest essays
        </h2>
        {posts && posts.length > 0 ? (
          <div>
            {posts.map((p) => (
              <PostCard key={p.id} post={p} />
            ))}
          </div>
        ) : (
          <p className="py-12 text-muted">
            No essays yet. <Link href="/write" className="text-brand underline">Write the first one.</Link>
          </p>
        )}
      </div>

      <aside className="lg:sticky lg:top-20 lg:self-start">
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wide text-muted">
          {siteConfig.boardNounPlural[0]!.toUpperCase() + siteConfig.boardNounPlural.slice(1)}
        </h2>
        <ul className="space-y-1">
          {(boards ?? []).map((b) => (
            <li key={b.id}>
              <Link
                href={`/boards/${b.slug}`}
                className="block rounded-lg px-3 py-2 text-sm hover:bg-brand-soft"
              >
                <span className="font-medium">{b.name}</span>
                {b.description && (
                  <span className="mt-0.5 block line-clamp-2 text-xs text-muted">{b.description}</span>
                )}
              </Link>
            </li>
          ))}
        </ul>
        <Link href="/boards" className="mt-3 inline-block text-sm text-brand hover:underline">
          Browse all →
        </Link>
      </aside>
    </div>
  );
}
