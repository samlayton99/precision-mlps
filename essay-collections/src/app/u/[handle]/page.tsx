import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { createClient } from "@/lib/supabase/server";
import { getViewer } from "@/lib/auth";
import { Avatar } from "@/components/ui/Avatar";
import { PostCard, type PostCardData } from "@/components/PostCard";
import type { Profile } from "@/lib/types";

const POST_SELECT =
  "id,title,subtitle,excerpt,like_count,comment_count,created_at,published_at,status," +
  "author:profiles!posts_author_id_fkey(handle,display_name,avatar_url),board:boards(slug,name)";

export async function generateMetadata({ params }: { params: { handle: string } }): Promise<Metadata> {
  return { title: `@${params.handle}` };
}

export default async function ProfilePage({ params }: { params: { handle: string } }) {
  const supabase = createClient();
  const viewer = await getViewer();

  const { data: profile } = await supabase
    .from("profiles")
    .select("*")
    .eq("handle", params.handle)
    .maybeSingle<Profile>();
  if (!profile) notFound();

  const isSelf = viewer?.id === profile.id;

  // Own profile shows drafts/hidden too; RLS enforces this regardless.
  let query = supabase
    .from("posts")
    .select(POST_SELECT)
    .eq("author_id", profile.id)
    .order("created_at", { ascending: false });
  if (!isSelf) query = query.eq("status", "published");

  const { data: posts } = await query.returns<PostCardData[]>();

  return (
    <div className="mx-auto max-w-2xl">
      <div className="flex items-center gap-4 border-b border-line pb-6">
        <Avatar src={profile.avatar_url} name={profile.display_name} size={64} />
        <div className="flex-1">
          <h1 className="font-serif text-2xl font-semibold">{profile.display_name}</h1>
          <p className="text-muted">@{profile.handle}</p>
        </div>
        {isSelf && (
          <Link href="/settings" className="rounded-full border border-line px-4 py-1.5 text-sm hover:bg-brand-soft">
            Edit profile
          </Link>
        )}
      </div>

      {profile.bio && <p className="mt-4 text-muted">{profile.bio}</p>}
      {profile.is_banned && (
        <p className="mt-4 rounded-lg border border-danger/40 bg-danger/5 px-3 py-2 text-sm text-danger">
          This account is currently restricted.
        </p>
      )}

      <h2 className="mt-8 text-sm font-medium uppercase tracking-wide text-muted">Essays</h2>
      <div className="mt-2">
        {posts && posts.length > 0 ? (
          posts.map((p) => <PostCard key={p.id} post={p} />)
        ) : (
          <p className="py-8 text-muted">No essays yet.</p>
        )}
      </div>
    </div>
  );
}
