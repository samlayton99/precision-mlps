"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { createPost, updatePost } from "@/app/actions/posts";

// Load the TipTap editor client-side only.
const Editor = dynamic(() => import("./Editor").then((m) => m.Editor), {
  ssr: false,
  loading: () => <div className="min-h-[24rem] animate-pulse rounded-xl bg-surface" />,
});

interface BoardOpt {
  id: string;
  name: string;
}
interface ExistingPost {
  id: string;
  title: string;
  subtitle: string | null;
  board_id: string;
  content_html: string;
  cover_image_url: string | null;
  status: string;
}

export function EssayComposer({
  boards,
  defaultBoardId,
  post,
}: {
  boards: BoardOpt[];
  defaultBoardId?: string;
  post?: ExistingPost;
}) {
  const router = useRouter();
  const isEdit = Boolean(post);

  const [title, setTitle] = useState(post?.title ?? "");
  const [subtitle, setSubtitle] = useState(post?.subtitle ?? "");
  const [boardId, setBoardId] = useState(post?.board_id ?? defaultBoardId ?? boards[0]?.id ?? "");
  const [coverImageUrl, setCoverImageUrl] = useState(post?.cover_image_url ?? "");
  const [html, setHtml] = useState(post?.content_html ?? "");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function save(status: "published" | "draft") {
    setError(null);
    if (!title.trim()) return setError("Please add a title.");
    if (!boardId) return setError("Please choose a board.");
    if (html.replace(/<[^>]*>/g, "").trim().length === 0) return setError("Your essay is empty.");

    setBusy(true);
    const payload = {
      boardId,
      title: title.trim(),
      subtitle: subtitle.trim(),
      contentHtml: html,
      coverImageUrl: coverImageUrl.trim(),
      status,
    };
    const res = isEdit ? await updatePost(post!.id, payload) : await createPost(payload);
    setBusy(false);
    if (res.ok) {
      router.push(`/posts/${res.postId}`);
      router.refresh();
    } else {
      setError(res.error);
    }
  }

  return (
    <div className="mx-auto max-w-2xl">
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <select
          value={boardId}
          onChange={(e) => setBoardId(e.target.value)}
          className="rounded-lg border border-line bg-surface px-3 py-2 text-sm"
        >
          {boards.map((b) => (
            <option key={b.id} value={b.id}>{b.name}</option>
          ))}
        </select>
        <span className="text-sm text-muted">{isEdit ? "Editing essay" : "New essay"}</span>
      </div>

      <input
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Title"
        maxLength={200}
        className="w-full bg-transparent font-serif text-4xl font-semibold leading-tight placeholder:text-muted focus:outline-none"
      />
      <input
        value={subtitle}
        onChange={(e) => setSubtitle(e.target.value)}
        placeholder="Subtitle (optional)"
        maxLength={300}
        className="mt-3 w-full bg-transparent text-xl text-muted placeholder:text-muted/70 focus:outline-none"
      />

      <div className="mt-6">
        <Editor initialHtml={post?.content_html} onChange={setHtml} />
      </div>

      <details className="mt-4 text-sm text-muted">
        <summary className="cursor-pointer">Cover image URL (optional)</summary>
        <input
          value={coverImageUrl}
          onChange={(e) => setCoverImageUrl(e.target.value)}
          placeholder="https://…"
          className="mt-2 w-full rounded-lg border border-line bg-surface px-3 py-2"
        />
      </details>

      {error && <p className="mt-4 text-sm text-danger">{error}</p>}

      <div className="mt-6 flex items-center gap-3">
        <button
          onClick={() => save("published")}
          disabled={busy}
          className="rounded-full bg-brand px-6 py-2.5 font-medium text-white disabled:opacity-50"
        >
          {busy ? "Saving…" : isEdit ? "Save & publish" : "Publish"}
        </button>
        <button
          onClick={() => save("draft")}
          disabled={busy}
          className="rounded-full border border-line px-6 py-2.5 font-medium hover:bg-brand-soft disabled:opacity-50"
        >
          Save draft
        </button>
      </div>
    </div>
  );
}
