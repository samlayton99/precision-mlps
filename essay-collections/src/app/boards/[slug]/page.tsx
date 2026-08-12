import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { createClient } from "@/lib/supabase/server";
import { getViewer } from "@/lib/auth";
import { PostCard, type PostCardData } from "@/components/PostCard";
import { BoardChat, type ChatItem } from "@/components/BoardChat";
import type { Board } from "@/lib/types";

export const revalidate = 20;

const POST_SELECT =
  "id,title,subtitle,excerpt,like_count,comment_count,created_at,published_at,status," +
  "author:profiles!posts_author_id_fkey(handle,display_name,avatar_url)";

const CHAT_SELECT =
  "id,author_id,body,created_at,author:profiles!board_chat_messages_author_id_fkey(handle,display_name,avatar_url)";

export async function generateMetadata({
  params,
}: {
  params: { slug: string };
}): Promise<Metadata> {
  const supabase = createClient();
  const { data } = await supabase.from("boards").select("name,description").eq("slug", params.slug).maybeSingle();
  return { title: data?.name ?? "Board", description: data?.description ?? undefined };
}

export default async function BoardPage({ params }: { params: { slug: string } }) {
  const supabase = createClient();

  const { data: board } = await supabase
    .from("boards")
    .select("*")
    .eq("slug", params.slug)
    .maybeSingle<Board>();
  if (!board) notFound();

  const [{ data: posts }, { data: chat }, viewer] = await Promise.all([
    supabase
      .from("posts")
      .select(POST_SELECT)
      .eq("board_id", board.id)
      .eq("status", "published")
      .order("published_at", { ascending: false })
      .limit(30)
      .returns<PostCardData[]>(),
    supabase
      .from("board_chat_messages")
      .select(CHAT_SELECT)
      .eq("board_id", board.id)
      .order("created_at", { ascending: false })
      .limit(50)
      .returns<ChatItem[]>(),
    getViewer(),
  ]);

  const initialChat = (chat ?? []).slice().reverse(); // oldest → newest

  return (
    <div>
      <div className="border-b border-line pb-6">
        <Link href="/boards" className="text-sm text-muted hover:text-ink">← All boards</Link>
        <h1 className="mt-2 font-serif text-3xl font-semibold">{board.name}</h1>
        {board.description && <p className="mt-2 max-w-2xl text-muted">{board.description}</p>}
      </div>

      <div className="mt-8 grid grid-cols-1 gap-10 lg:grid-cols-[1fr_360px]">
        <div>
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-sm font-medium uppercase tracking-wide text-muted">Essays</h2>
            {viewer && (
              <Link
                href={`/write?board=${board.slug}`}
                className="text-sm text-brand hover:underline"
              >
                Write in this board →
              </Link>
            )}
          </div>
          {posts && posts.length > 0 ? (
            posts.map((p) => <PostCard key={p.id} post={p} />)
          ) : (
            <p className="py-12 text-muted">No essays in this board yet.</p>
          )}
        </div>

        <aside className="lg:sticky lg:top-20 lg:self-start">
          <BoardChat
            boardId={board.id}
            initialMessages={initialChat}
            viewer={viewer ? { id: viewer.id, isAdmin: viewer.isAdmin } : null}
          />
        </aside>
      </div>
    </div>
  );
}
