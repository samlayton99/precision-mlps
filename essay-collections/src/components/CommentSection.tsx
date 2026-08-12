"use client";

import { useState, useTransition } from "react";
import Link from "next/link";
import { Avatar } from "@/components/ui/Avatar";
import { timeAgo } from "@/lib/utils";
import { addComment, deleteOwnComment } from "@/app/actions/comments";
import { adminDeleteComment } from "@/app/actions/admin";
import { reportContent } from "@/app/actions/report";
import { Trash2, Flag } from "lucide-react";

export interface CommentItem {
  id: string;
  author_id: string;
  body: string;
  created_at: string;
  author: { handle: string; display_name: string; avatar_url: string | null } | null;
}
interface Viewer {
  id: string;
  isAdmin: boolean;
  canWrite: boolean;
}

export function CommentSection({
  postId,
  initialComments,
  viewer,
}: {
  postId: string;
  initialComments: CommentItem[];
  viewer: Viewer | null;
}) {
  const [comments, setComments] = useState(initialComments);
  const [body, setBody] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const text = body.trim();
    if (!text) return;
    setError(null);
    startTransition(async () => {
      const res = await addComment({ postId, body: text });
      if (res.ok) {
        setBody("");
        // Refresh from server to pick up the new comment with author info.
        window.location.reload();
      } else {
        setError(res.error);
      }
    });
  }

  async function remove(id: string, isAdminAction: boolean) {
    const reason = isAdminAction
      ? window.prompt("Reason for removing this comment (optional):") ?? undefined
      : undefined;
    setComments((prev) => prev.filter((c) => c.id !== id)); // optimistic
    if (isAdminAction) await adminDeleteComment(id, reason);
    else await deleteOwnComment(id, postId);
  }

  async function report(id: string) {
    const reason = window.prompt("Why are you reporting this comment?");
    if (!reason) return;
    const res = await reportContent({ contentType: "comment", contentId: id, reason });
    alert(res.ok ? "Thank you. An admin will review it." : `Could not file report: ${res.error}`);
  }

  return (
    <section className="mt-12">
      <h2 className="font-serif text-xl font-semibold">
        {comments.length} {comments.length === 1 ? "comment" : "comments"}
      </h2>

      {viewer ? (
        viewer.canWrite ? (
          <form onSubmit={submit} className="mt-4">
            {error && <p className="mb-2 text-sm text-danger">{error}</p>}
            <textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              rows={3}
              maxLength={10000}
              placeholder="Add a thoughtful, charitable comment…"
              className="w-full rounded-lg border border-line bg-surface p-3 text-sm focus:border-brand focus:outline-none"
            />
            <div className="mt-2 flex justify-end">
              <button
                type="submit"
                disabled={pending || !body.trim()}
                className="rounded-full bg-brand px-5 py-2 text-sm font-medium text-white disabled:opacity-50"
              >
                {pending ? "Posting…" : "Comment"}
              </button>
            </div>
          </form>
        ) : (
          <p className="mt-4 text-sm text-muted">Your account is restricted from commenting.</p>
        )
      ) : (
        <p className="mt-4 text-sm text-muted">
          <Link href="/login" className="text-brand underline">Sign in</Link> to join the discussion.
        </p>
      )}

      <ul className="mt-6 space-y-6">
        {comments.map((c) => {
          const canDeleteOwn = viewer?.id === c.author_id;
          const canDeleteAdmin = viewer?.isAdmin && !canDeleteOwn;
          return (
            <li key={c.id} className="group flex gap-3">
              <Avatar src={c.author?.avatar_url} name={c.author?.display_name ?? "?"} size={32} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 text-sm">
                  <Link href={`/u/${c.author?.handle ?? ""}`} className="font-medium hover:text-brand">
                    {c.author?.display_name ?? "Member"}
                  </Link>
                  <span className="text-muted">{timeAgo(c.created_at)}</span>
                </div>
                <p className="mt-1 whitespace-pre-wrap break-words">{c.body}</p>
              </div>
              <div className="flex items-start gap-2 opacity-0 transition group-hover:opacity-100">
                {viewer && !canDeleteOwn && (
                  <button onClick={() => report(c.id)} title="Report">
                    <Flag size={14} className="text-muted hover:text-danger" />
                  </button>
                )}
                {(canDeleteOwn || canDeleteAdmin) && (
                  <button onClick={() => remove(c.id, Boolean(canDeleteAdmin))} title="Delete">
                    <Trash2 size={14} className="text-muted hover:text-danger" />
                  </button>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
