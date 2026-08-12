import { createClient } from "@/lib/supabase/server";
import { ModerationQueue, type EnrichedFlag } from "@/components/admin/ModerationQueue";

export const dynamic = "force-dynamic";

interface FlagRow {
  id: string;
  content_type: "post" | "comment" | "chat" | "profile";
  content_id: string;
  category: string | null;
  severity: "low" | "medium" | "high" | null;
  reason: string;
  flagged_by: "bot" | "user";
  status: string;
  created_at: string;
  reporter: { handle: string; display_name: string } | null;
}

export default async function ModerationPage() {
  const supabase = createClient();
  const { data: flags } = await supabase
    .from("moderation_flags")
    .select(
      "id,content_type,content_id,category,severity,reason,flagged_by,status,created_at," +
        "reporter:profiles!moderation_flags_reporter_id_fkey(handle,display_name)",
    )
    .eq("status", "open")
    .order("created_at", { ascending: false })
    .returns<FlagRow[]>();

  const rows = flags ?? [];

  // Batch-enrich each flag with a preview + link to the content.
  const ids = (t: string) => rows.filter((f) => f.content_type === t).map((f) => f.content_id);
  const [posts, comments, chats, profiles] = await Promise.all([
    ids("post").length
      ? supabase.from("posts").select("id,title").in("id", ids("post"))
      : Promise.resolve({ data: [] as { id: string; title: string }[] }),
    ids("comment").length
      ? supabase.from("comments").select("id,post_id,body").in("id", ids("comment"))
      : Promise.resolve({ data: [] as { id: string; post_id: string; body: string }[] }),
    ids("chat").length
      ? supabase.from("board_chat_messages").select("id,body").in("id", ids("chat"))
      : Promise.resolve({ data: [] as { id: string; body: string }[] }),
    ids("profile").length
      ? supabase.from("profiles").select("id,handle,display_name").in("id", ids("profile"))
      : Promise.resolve({ data: [] as { id: string; handle: string; display_name: string }[] }),
  ]);

  const postMap = new Map((posts.data ?? []).map((p) => [p.id, p]));
  const commentMap = new Map((comments.data ?? []).map((c) => [c.id, c]));
  const chatMap = new Map((chats.data ?? []).map((c) => [c.id, c]));
  const profileMap = new Map((profiles.data ?? []).map((p) => [p.id, p]));

  const enriched: EnrichedFlag[] = rows.map((f) => {
    let preview = "";
    let href: string | null = null;
    if (f.content_type === "post") {
      preview = postMap.get(f.content_id)?.title ?? "(deleted post)";
      href = `/posts/${f.content_id}`;
    } else if (f.content_type === "comment") {
      const c = commentMap.get(f.content_id);
      preview = c?.body ?? "(deleted comment)";
      href = c ? `/posts/${c.post_id}` : null;
    } else if (f.content_type === "chat") {
      preview = chatMap.get(f.content_id)?.body ?? "(deleted message)";
    } else {
      const p = profileMap.get(f.content_id);
      preview = p ? `@${p.handle} — ${p.display_name}` : "(unknown member)";
      href = p ? `/u/${p.handle}` : null;
    }
    return { ...f, preview, href };
  });

  return (
    <div>
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="font-serif text-xl font-semibold">Moderation queue</h2>
        <span className="text-sm text-muted">{enriched.length} open</span>
      </div>
      {enriched.length === 0 ? (
        <p className="rounded-xl border border-line bg-surface p-8 text-center text-muted">
          Nothing to review. The daily scan and member reports will surface here.
        </p>
      ) : (
        <ModerationQueue flags={enriched} />
      )}
    </div>
  );
}
