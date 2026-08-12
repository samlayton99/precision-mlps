"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { createClient } from "@/lib/supabase/client";
import { setUserBan } from "@/app/actions/admin";
import { Avatar } from "@/components/ui/Avatar";

export interface MemberRow {
  id: string;
  handle: string;
  display_name: string;
  avatar_url: string | null;
  is_banned: boolean;
  banned_reason: string | null;
}

export function UserModeration({
  bannedMembers,
  adminIds,
}: {
  bannedMembers: MemberRow[];
  adminIds: string[];
}) {
  const router = useRouter();
  const adminSet = new Set(adminIds);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<MemberRow[]>([]);
  const [searching, setSearching] = useState(false);
  const [pending, startTransition] = useTransition();
  const [msg, setMsg] = useState<string | null>(null);

  async function search(e: React.FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setSearching(true);
    const supabase = createClient();
    const { data } = await supabase
      .from("profiles")
      .select("id,handle,display_name,avatar_url,is_banned,banned_reason")
      .or(`handle.ilike.%${q}%,display_name.ilike.%${q}%`)
      .limit(20)
      .returns<MemberRow[]>();
    setResults(data ?? []);
    setSearching(false);
  }

  function ban(m: MemberRow, banned: boolean) {
    setMsg(null);
    if (adminSet.has(m.id) && banned) {
      setMsg("Remove this member's admin privileges before banning them.");
      return;
    }
    const reason = banned
      ? window.prompt(`Reason for restricting ${m.display_name} (shown to them):`) ?? undefined
      : undefined;
    if (banned && !window.confirm(`Restrict @${m.handle}? They will be unable to post, comment, like, or chat.`)) return;
    startTransition(async () => {
      const res = await setUserBan({ userId: m.id, banned, reason: reason ?? "" });
      if (!res.ok) setMsg(res.error);
      else {
        setResults((prev) => prev.map((r) => (r.id === m.id ? { ...r, is_banned: banned } : r)));
        router.refresh();
      }
    });
  }

  function Row({ m }: { m: MemberRow }) {
    return (
      <li className="flex flex-wrap items-center gap-3 rounded-xl border border-line bg-surface p-4">
        <Avatar src={m.avatar_url} name={m.display_name} size={38} />
        <div className="min-w-0 flex-1">
          <Link href={`/u/${m.handle}`} className="font-medium hover:text-brand">
            {m.display_name}
          </Link>
          <p className="text-sm text-muted">
            @{m.handle}
            {adminSet.has(m.id) && <span className="ml-2 rounded bg-brand-soft px-1.5 py-0.5 text-xs text-brand">admin</span>}
            {m.is_banned && <span className="ml-2 rounded bg-danger/10 px-1.5 py-0.5 text-xs text-danger">restricted</span>}
          </p>
          {m.is_banned && m.banned_reason && (
            <p className="text-xs text-muted">Reason: {m.banned_reason}</p>
          )}
        </div>
        {m.is_banned ? (
          <button
            onClick={() => ban(m, false)}
            disabled={pending}
            className="rounded-full border border-line px-3 py-1.5 text-sm hover:bg-brand-soft"
          >
            Lift restriction
          </button>
        ) : (
          <button
            onClick={() => ban(m, true)}
            disabled={pending}
            className="rounded-full border border-danger px-3 py-1.5 text-sm text-danger hover:bg-danger hover:text-white"
          >
            Restrict
          </button>
        )}
      </li>
    );
  }

  return (
    <div className="space-y-8">
      {msg && <p className="text-sm text-danger">{msg}</p>}

      <section className="rounded-xl border border-line bg-surface p-5">
        <h2 className="font-serif text-lg font-semibold">Find a member</h2>
        <form onSubmit={search} className="mt-3 flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by handle or name…"
            className="flex-1 rounded-lg border border-line bg-canvas px-3 py-2 text-sm"
          />
          <button
            type="submit"
            disabled={searching}
            className="rounded-full bg-brand px-5 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {searching ? "…" : "Search"}
          </button>
        </form>
        {results.length > 0 && (
          <ul className="mt-4 space-y-2">
            {results.map((m) => (
              <Row key={m.id} m={m} />
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2 className="font-serif text-lg font-semibold">Restricted members ({bannedMembers.length})</h2>
        {bannedMembers.length === 0 ? (
          <p className="mt-2 text-sm text-muted">No members are currently restricted.</p>
        ) : (
          <ul className="mt-3 space-y-2">
            {bannedMembers.map((m) => (
              <Row key={m.id} m={m} />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
