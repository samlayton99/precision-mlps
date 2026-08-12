import { notFound, redirect } from "next/navigation";
import type { Metadata } from "next";
import { createClient } from "@/lib/supabase/server";
import { requireViewer } from "@/lib/auth";
import { EssayComposer } from "@/components/editor/EssayComposer";
import type { Board } from "@/lib/types";

export const metadata: Metadata = { title: "Edit essay" };

interface EditPost {
  id: string;
  author_id: string;
  title: string;
  subtitle: string | null;
  board_id: string;
  content_html: string;
  cover_image_url: string | null;
  status: string;
}

export default async function EditPostPage({ params }: { params: { id: string } }) {
  const viewer = await requireViewer(`/posts/${params.id}/edit`);
  const supabase = createClient();

  const { data: post } = await supabase
    .from("posts")
    .select("id,author_id,title,subtitle,board_id,content_html,cover_image_url,status")
    .eq("id", params.id)
    .maybeSingle<EditPost>();

  if (!post) notFound();
  if (post.author_id !== viewer.id) redirect(`/posts/${params.id}`); // only the author edits

  const { data: boards } = await supabase
    .from("boards")
    .select("id,name")
    .eq("is_archived", false)
    .order("sort_order")
    .returns<Pick<Board, "id" | "name">[]>();

  return <EssayComposer boards={boards ?? []} post={post} />;
}
