"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { createBoard, deleteBoard } from "@/app/actions/admin";
import { Trash2 } from "lucide-react";

interface BoardItem {
  id: string;
  slug: string;
  name: string;
  description: string | null;
}

export function BoardManager({ boards }: { boards: BoardItem[] }) {
  const router = useRouter();
  const [name, setName] = useState("");
  const [slug, setSlug] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();

  // auto-suggest slug from name until the user edits slug directly
  const [slugTouched, setSlugTouched] = useState(false);
  function onName(v: string) {
    setName(v);
    if (!slugTouched) {
      setSlug(
        v.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 50),
      );
    }
  }

  function create(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    startTransition(async () => {
      const res = await createBoard({ slug, name, description });
      if (res.ok) {
        setName("");
        setSlug("");
        setDescription("");
        setSlugTouched(false);
        router.refresh();
      } else {
        setError(res.error);
      }
    });
  }

  function remove(b: BoardItem) {
    if (!window.confirm(`Delete "${b.name}"? This permanently deletes ALL its essays, comments, and chat.`)) return;
    startTransition(async () => {
      await deleteBoard(b.id);
      router.refresh();
    });
  }

  return (
    <div className="space-y-8">
      <section className="rounded-xl border border-line bg-surface p-5">
        <h2 className="font-serif text-lg font-semibold">Create a board</h2>
        <form onSubmit={create} className="mt-3 space-y-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block">
              <span className="text-sm">Name</span>
              <input
                value={name}
                onChange={(e) => onName(e.target.value)}
                required
                maxLength={80}
                className="mt-1 w-full rounded-lg border border-line bg-canvas px-3 py-2 text-sm"
              />
            </label>
            <label className="block">
              <span className="text-sm">Slug</span>
              <input
                value={slug}
                onChange={(e) => {
                  setSlugTouched(true);
                  setSlug(e.target.value.toLowerCase());
                }}
                required
                className="mt-1 w-full rounded-lg border border-line bg-canvas px-3 py-2 text-sm"
              />
            </label>
          </div>
          <label className="block">
            <span className="text-sm">Description</span>
            <input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              maxLength={500}
              className="mt-1 w-full rounded-lg border border-line bg-canvas px-3 py-2 text-sm"
            />
          </label>
          {error && <p className="text-sm text-danger">{error}</p>}
          <button
            type="submit"
            disabled={pending}
            className="rounded-full bg-brand px-5 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            Create board
          </button>
        </form>
      </section>

      <section>
        <h2 className="font-serif text-lg font-semibold">Boards</h2>
        <ul className="mt-3 space-y-2">
          {boards.map((b) => (
            <li key={b.id} className="flex items-center gap-3 rounded-xl border border-line bg-surface p-4">
              <div className="min-w-0 flex-1">
                <p className="font-medium">
                  {b.name} <span className="text-sm text-muted">/{b.slug}</span>
                </p>
                {b.description && <p className="truncate text-sm text-muted">{b.description}</p>}
              </div>
              <button
                onClick={() => remove(b)}
                disabled={pending}
                className="rounded-full border border-line p-2 text-muted hover:border-danger hover:text-danger"
                title="Delete board"
              >
                <Trash2 size={16} />
              </button>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
