import type { Metadata } from "next";
import { createClient } from "@/lib/supabase/server";
import { requireViewer } from "@/lib/auth";
import { EssayComposer } from "@/components/editor/EssayComposer";
import type { Board } from "@/lib/types";

export const metadata: Metadata = { title: "Write an essay" };

export default async function WritePage({ searchParams }: { searchParams: { board?: string } }) {
  await requireViewer("/write");
  const supabase = createClient();
  const { data: boards } = await supabase
    .from("boards")
    .select("id,name,slug")
    .eq("is_archived", false)
    .order("sort_order")
    .returns<Pick<Board, "id" | "name" | "slug">[]>();

  const list = boards ?? [];
  const defaultBoardId = list.find((b) => b.slug === searchParams.board)?.id;

  if (list.length === 0) {
    return <p className="py-16 text-center text-muted">No boards exist yet. An admin needs to create one.</p>;
  }

  return (
    <EssayComposer
      boards={list.map((b) => ({ id: b.id, name: b.name }))}
      defaultBoardId={defaultBoardId}
    />
  );
}
