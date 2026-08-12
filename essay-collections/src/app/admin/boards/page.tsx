import { createClient } from "@/lib/supabase/server";
import { BoardManager } from "@/components/admin/BoardManager";
import type { Board } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function AdminBoardsPage() {
  const supabase = createClient();
  const { data: boards } = await supabase
    .from("boards")
    .select("*")
    .order("sort_order")
    .returns<Board[]>();

  return (
    <BoardManager
      boards={(boards ?? []).map((b) => ({
        id: b.id,
        slug: b.slug,
        name: b.name,
        description: b.description,
      }))}
    />
  );
}
