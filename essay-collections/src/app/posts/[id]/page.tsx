import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { createClient } from "@/lib/supabase/server";
import { getViewer } from "@/lib/auth";
import { RenderedContent } from "@/components/RenderedContent";
import { LikeButton } from "@/components/LikeButton";
import { CommentSection, type CommentItem } from "@/components/CommentSection";
import { PostControls } from "@/components/PostControls";
import { Avatar } from "@/components/ui/Avatar";
import { timeAgo } from "@/lib/utils";

export const dynamic = "force-dynamic";

const POST_SELECT =
  "*,author:profiles!posts_author_id_fkey(handle,display_name,avatar_url,bio),board:boards(slug,name)";
const COMMENT_SELECT =
  "id,author_id,body,created_at,author:profiles!comments_author_id_fkey(handle,display_name,avatar_url)";

interface FullPost {
  id: string;
  board_id: string;
  author_id: string;
  title: string;
  subtitle: string | null;
  content_html: string;
  status: string;
  hidden_reason: string | null;
  like_count: number;
  created_at: string;
  published_at: string | null;
  author: { handle: string; display_name: string; avatar_url: string | null; bio: string | null } | null;
  board: { slug: string; name: string } | null;
}

export async function generateMetadata({ params }: { params: { id: string } }): Promise<Metadata> {
  const supabase = createClient();
  const { data } = await supabase.from("posts").select("title,subtitle,excerpt").eq("id", params.id).maybeSingle();
  if (!data) return { title: "Essay" };
  return { title: data.title, description: data.subtitle ?? data.excerpt ?? undefined };
}

export default async function PostPage({ params }: { params: { id: string } }) {
  const supabase = createClient();
  const viewer = await getViewer();

  const { data: post } = await supabase
    .from("posts")
    .select(POST_SELECT)
    .eq("id", params.id)
    .maybeSingle<FullPost>();

  // RLS already hides drafts/hidden from non-owners; treat missing as 404.
  if (!post) notFound();

  const isAuthor = viewer?.id === post.author_id;

  const [{ data: comments }, likedRes] = await Promise.all([
    supabase.from("comments").select(COMMENT_SELECT).eq("post_id", post.id).order("created_at").returns<CommentItem[]>(),
    viewer
      ? supabase.from("post_likes").select("post_id").eq("post_id", post.id).eq("user_id", viewer.id).maybeSingle()
      : Promise.resolve({ data: null }),
  ]);

  const liked = Boolean(likedRes?.data);

  return (
    <article className="mx-auto max-w-2xl">
      {post.status !== "published" && (
        <div className="mb-6 rounded-lg border border-line bg-brand-soft px-4 py-2 text-sm">
          This essay is <strong>{post.status}</strong>
          {post.hidden_reason ? ` — ${post.hidden_reason}` : ""}. Only you and admins can see it.
        </div>
      )}

      {post.board && (
        <Link href={`/boards/${post.board.slug}`} className="text-sm font-medium text-brand hover:underline">
          {post.board.name}
        </Link>
      )}
      <h1 className="mt-2 font-serif text-4xl font-semibold leading-tight">{post.title}</h1>
      {post.subtitle && <p className="mt-3 text-xl text-muted">{post.subtitle}</p>}

      <div className="mt-6 flex items-center gap-3 border-b border-line pb-6">
        {post.author && (
          <>
            <Avatar src={post.author.avatar_url} name={post.author.display_name} size={44} />
            <div>
              <Link href={`/u/${post.author.handle}`} className="font-medium hover:text-brand">
                {post.author.display_name}
              </Link>
              <p className="text-sm text-muted">{timeAgo(post.published_at ?? post.created_at)}</p>
            </div>
          </>
        )}
      </div>

      <div className="mt-8">
        <RenderedContent html={post.content_html} />
      </div>

      <div className="mt-10 flex items-center justify-between border-t border-line pt-6">
        <LikeButton
          postId={post.id}
          initialLiked={liked}
          initialCount={post.like_count}
          canInteract={Boolean(viewer && !viewer.profile.is_banned)}
        />
        <PostControls
          postId={post.id}
          status={post.status}
          isAuthor={Boolean(isAuthor)}
          isAdmin={Boolean(viewer?.isAdmin)}
          isAuthed={Boolean(viewer)}
        />
      </div>

      <CommentSection
        postId={post.id}
        initialComments={comments ?? []}
        viewer={
          viewer
            ? { id: viewer.id, isAdmin: viewer.isAdmin, canWrite: !viewer.profile.is_banned }
            : null
        }
      />
    </article>
  );
}
