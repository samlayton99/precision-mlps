"use client";

import { useTransition } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { deleteOwnPost } from "@/app/actions/posts";
import { adminDeletePost, adminSetPostHidden } from "@/app/actions/admin";
import { reportContent } from "@/app/actions/report";
import { Pencil, Trash2, EyeOff, Eye, Flag } from "lucide-react";

export function PostControls({
  postId,
  status,
  isAuthor,
  isAdmin,
  isAuthed,
}: {
  postId: string;
  status: string;
  isAuthor: boolean;
  isAdmin: boolean;
  isAuthed: boolean;
}) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();

  function ownerDelete() {
    if (!window.confirm("Delete this essay permanently?")) return;
    startTransition(async () => {
      await deleteOwnPost(postId);
    });
  }

  function adminDelete() {
    const reason = window.prompt("Reason for deleting this essay (recorded in the audit log):") ?? undefined;
    if (!window.confirm("Delete this essay as an admin? This cannot be undone.")) return;
    startTransition(async () => {
      const res = await adminDeletePost(postId, reason);
      if (res.ok) router.push("/");
      else alert(res.error);
    });
  }

  function adminToggleHide() {
    const hide = status !== "hidden";
    const reason = hide
      ? window.prompt("Reason for hiding/pausing this essay:") ?? undefined
      : undefined;
    startTransition(async () => {
      const res = await adminSetPostHidden(postId, hide, reason);
      if (res.ok) router.refresh();
      else alert(res.error);
    });
  }

  async function report() {
    const reason = window.prompt("Why are you reporting this essay?");
    if (!reason) return;
    const res = await reportContent({ contentType: "post", contentId: postId, reason });
    alert(res.ok ? "Thank you. An admin will review it." : `Could not file report: ${res.error}`);
  }

  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      {isAuthor && (
        <>
          <Link
            href={`/posts/${postId}/edit`}
            className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 hover:bg-brand-soft"
          >
            <Pencil size={14} /> Edit
          </Link>
          <button
            onClick={ownerDelete}
            disabled={pending}
            className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 hover:text-danger"
          >
            <Trash2 size={14} /> Delete
          </button>
        </>
      )}

      {isAdmin && (
        <>
          <button
            onClick={adminToggleHide}
            disabled={pending}
            className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 hover:bg-brand-soft"
            title="Admin: hide/unhide"
          >
            {status === "hidden" ? <Eye size={14} /> : <EyeOff size={14} />}
            {status === "hidden" ? "Unhide" : "Hide"}
          </button>
          {!isAuthor && (
            <button
              onClick={adminDelete}
              disabled={pending}
              className="inline-flex items-center gap-1.5 rounded-full border border-danger px-3 py-1.5 text-danger hover:bg-danger hover:text-white"
              title="Admin: delete"
            >
              <Trash2 size={14} /> Delete
            </button>
          )}
        </>
      )}

      {isAuthed && !isAuthor && (
        <button onClick={report} className="inline-flex items-center gap-1.5 text-muted hover:text-danger">
          <Flag size={14} /> Report
        </button>
      )}
    </div>
  );
}
