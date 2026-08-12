import Link from "next/link";
import type { Metadata } from "next";
import { createClient } from "@/lib/supabase/server";
import type { Board } from "@/lib/types";

export const metadata: Metadata = { title: "Boards" };
export const revalidate = 60;

export default async function BoardsPage() {
  const supabase = createClient();
  const { data: boards } = await supabase
    .from("boards")
    .select("*")
    .eq("is_archived", false)
    .order("sort_order")
    .returns<Board[]>();

  return (
    <div>
      <h1 className="font-serif text-3xl font-semibold">Discussion boards</h1>
      <p className="mt-2 text-muted">Essays are organized into boards. Each board has its own chat.</p>

      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        {(boards ?? []).map((b) => (
          <Link
            key={b.id}
            href={`/boards/${b.slug}`}
            className="rounded-xl border border-line bg-surface p-5 transition hover:border-brand"
          >
            <h2 className="font-serif text-xl font-semibold">{b.name}</h2>
            {b.description && <p className="mt-2 text-sm text-muted">{b.description}</p>}
          </Link>
        ))}
      </div>
    </div>
  );
}
