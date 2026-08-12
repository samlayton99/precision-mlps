"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { updateProfile } from "@/app/actions/profile";

export function ProfileSettingsForm({
  initial,
}: {
  initial: { displayName: string; handle: string; bio: string };
}) {
  const router = useRouter();
  const [displayName, setDisplayName] = useState(initial.displayName);
  const [handle, setHandle] = useState(initial.handle);
  const [bio, setBio] = useState(initial.bio);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  async function save(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setMsg(null);
    const res = await updateProfile({ displayName, handle, bio });
    setBusy(false);
    if (res.ok) {
      setMsg({ ok: true, text: "Saved." });
      router.refresh();
    } else {
      setMsg({ ok: false, text: res.error });
    }
  }

  return (
    <form onSubmit={save} className="space-y-5">
      <label className="block">
        <span className="text-sm font-medium">Display name</span>
        <input
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          maxLength={60}
          className="mt-1 w-full rounded-lg border border-line bg-surface px-3 py-2"
        />
      </label>
      <label className="block">
        <span className="text-sm font-medium">Handle</span>
        <div className="mt-1 flex items-center rounded-lg border border-line bg-surface px-3">
          <span className="text-muted">@</span>
          <input
            value={handle}
            onChange={(e) => setHandle(e.target.value.toLowerCase())}
            maxLength={24}
            className="w-full bg-transparent py-2 pl-1 focus:outline-none"
          />
        </div>
        <span className="mt-1 block text-xs text-muted">
          3–24 characters: lowercase letters, numbers, underscores. Others tag you with @{handle}.
        </span>
      </label>
      <label className="block">
        <span className="text-sm font-medium">Bio</span>
        <textarea
          value={bio}
          onChange={(e) => setBio(e.target.value)}
          rows={3}
          maxLength={500}
          className="mt-1 w-full rounded-lg border border-line bg-surface px-3 py-2"
        />
      </label>

      {msg && <p className={msg.ok ? "text-sm text-brand" : "text-sm text-danger"}>{msg.text}</p>}

      <button
        type="submit"
        disabled={busy}
        className="rounded-full bg-brand px-6 py-2.5 font-medium text-white disabled:opacity-50"
      >
        {busy ? "Saving…" : "Save changes"}
      </button>
    </form>
  );
}
