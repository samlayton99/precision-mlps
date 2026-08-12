import Link from "next/link";
import { Avatar } from "@/components/ui/Avatar";
import { Heart, MessageCircle } from "lucide-react";
import { timeAgo } from "@/lib/utils";

export interface PostCardData {
  id: string;
  title: string;
  subtitle: string | null;
  excerpt: string | null;
  like_count: number;
  comment_count: number;
  created_at: string;
  published_at: string | null;
  status?: string;
  author: { handle: string; display_name: string; avatar_url: string | null } | null;
  board?: { slug: string; name: string } | null;
}

export function PostCard({ post }: { post: PostCardData }) {
  return (
    <article className="border-b border-line py-6 first:pt-0">
      <div className="flex items-center gap-2 text-sm text-muted">
        {post.author && (
          <>
            <Avatar src={post.author.avatar_url} name={post.author.display_name} size={22} />
            <Link href={`/u/${post.author.handle}`} className="hover:text-ink">
              {post.author.display_name}
            </Link>
          </>
        )}
        {post.board && (
          <>
            <span aria-hidden>·</span>
            <Link href={`/boards/${post.board.slug}`} className="hover:text-ink">
              {post.board.name}
            </Link>
          </>
        )}
        <span aria-hidden>·</span>
        <time dateTime={post.published_at ?? post.created_at}>
          {timeAgo(post.published_at ?? post.created_at)}
        </time>
        {post.status && post.status !== "published" && (
          <span className="rounded bg-brand-soft px-1.5 py-0.5 text-xs uppercase tracking-wide text-brand">
            {post.status}
          </span>
        )}
      </div>

      <Link href={`/posts/${post.id}`} className="group mt-2 block">
        <h2 className="font-serif text-2xl font-semibold leading-snug group-hover:text-brand">
          {post.title}
        </h2>
        {(post.subtitle || post.excerpt) && (
          <p className="mt-1 line-clamp-2 text-muted">{post.subtitle || post.excerpt}</p>
        )}
      </Link>

      <div className="mt-3 flex items-center gap-5 text-sm text-muted">
        <span className="flex items-center gap-1.5">
          <Heart size={15} /> {post.like_count}
        </span>
        <span className="flex items-center gap-1.5">
          <MessageCircle size={15} /> {post.comment_count}
        </span>
      </div>
    </article>
  );
}
