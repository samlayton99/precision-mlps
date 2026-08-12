import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { createAdminClient } from "@/lib/supabase/admin";
import { htmlToText } from "@/lib/sanitize";
import { judgeBatch, type ModerationItem } from "@/lib/moderation";

/**
 * Daily moderation sweep (Vercel Cron -> GET, authorized by CRON_SECRET).
 *
 * Scans the last 24h of posts, comments, and chat messages against the
 * community-guidelines rubric using Claude, and files moderation_flags for
 * HUMAN admin review. The bot never deletes or hides anything itself —
 * admins act on flags from /admin/moderation.
 */
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const BATCH_SIZE = 15;
const MODEL = process.env.MODERATION_MODEL || "claude-opus-5";

export async function GET(request: Request) {
  const secret = process.env.CRON_SECRET;
  const auth = request.headers.get("authorization");
  if (!secret || auth !== `Bearer ${secret}`) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  if (!process.env.ANTHROPIC_API_KEY) {
    return NextResponse.json({ error: "ANTHROPIC_API_KEY not configured" }, { status: 500 });
  }

  const db = createAdminClient(); // service role: the bot must read all content
  const anthropic = new Anthropic();
  const since = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();

  const { data: run } = await db
    .from("moderation_runs")
    .insert({ model: MODEL })
    .select("id")
    .single();

  try {
    // ---- collect the last 24h of content --------------------------------
    const [posts, comments, chats] = await Promise.all([
      db
        .from("posts")
        .select("id,title,content_html,board_id")
        .eq("status", "published")
        .gte("created_at", since),
      db.from("comments").select("id,body").gte("created_at", since),
      db.from("board_chat_messages").select("id,body,board_id").gte("created_at", since),
    ]);

    const items: ModerationItem[] = [
      ...(posts.data ?? []).map((p) => ({
        id: p.id,
        type: "post" as const,
        text: `${p.title}\n\n${htmlToText(p.content_html)}`,
      })),
      ...(comments.data ?? []).map((c) => ({ id: c.id, type: "comment" as const, text: c.body })),
      ...(chats.data ?? []).map((m) => ({ id: m.id, type: "chat" as const, text: m.body })),
    ];

    // Skip items that already have an open or previously-reviewed bot flag,
    // so re-runs don't duplicate work.
    const ids = items.map((i) => i.id);
    const { data: existing } = ids.length
      ? await db.from("moderation_flags").select("content_id").in("content_id", ids)
      : { data: [] as { content_id: string }[] };
    const alreadyFlagged = new Set((existing ?? []).map((f) => f.content_id));
    const toScan = items.filter((i) => !alreadyFlagged.has(i.id));

    // Board lookup so flags link back to where the content lives.
    const boardOf = new Map<string, string>();
    for (const p of posts.data ?? []) boardOf.set(p.id, p.board_id);
    for (const m of chats.data ?? []) boardOf.set(m.id, m.board_id);

    // ---- judge in batches -------------------------------------------------
    let flags = 0;
    for (let i = 0; i < toScan.length; i += BATCH_SIZE) {
      const batch = toScan.slice(i, i + BATCH_SIZE);
      const verdicts = await judgeBatch(anthropic, MODEL, batch);
      const typeOf = new Map(batch.map((b) => [b.id, b.type]));

      const violations = verdicts.filter((v) => v.violation && typeOf.has(v.id));
      if (violations.length > 0) {
        const { error } = await db.from("moderation_flags").insert(
          violations.map((v) => ({
            content_type: typeOf.get(v.id)!,
            content_id: v.id,
            board_id: boardOf.get(v.id) ?? null,
            category: v.category,
            severity: v.severity ?? "low",
            reason: v.reason || "Flagged by daily guideline scan.",
            flagged_by: "bot" as const,
            status: "open" as const,
          })),
        );
        if (!error) flags += violations.length;
      }
    }

    if (run) {
      await db
        .from("moderation_runs")
        .update({ finished_at: new Date().toISOString(), items_scanned: toScan.length, flags_created: flags })
        .eq("id", run.id);
    }

    return NextResponse.json({ ok: true, scanned: toScan.length, flagged: flags, model: MODEL });
  } catch (err) {
    const message = err instanceof Error ? err.message : "unknown error";
    if (run) {
      await db
        .from("moderation_runs")
        .update({ finished_at: new Date().toISOString(), error: message })
        .eq("id", run.id);
    }
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
