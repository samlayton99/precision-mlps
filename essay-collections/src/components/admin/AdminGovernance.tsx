"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { Avatar } from "@/components/ui/Avatar";
import {
  inviteAdminByEmail,
  selfRevokeAdmin,
  requestAdminRemoval,
  cancelAdminRemoval,
} from "@/app/actions/admin";

export interface AdminEntry {
  userId: string;
  handle: string;
  displayName: string;
  avatarUrl: string | null;
  grantedAt: string;
  votes: number;
  viewerRequested: boolean;
  isViewer: boolean;
}

export function AdminGovernance({
  entries,
  requiredVotes,
  invites,
  atCap,
}: {
  entries: AdminEntry[];
  requiredVotes: number;
  invites: { email: string; createdAt: string }[];
  atCap: boolean;
}) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [pending, startTransition] = useTransition();

  function invite(e: React.FormEvent) {
    e.preventDefault();
    setMsg(null);
    startTransition(async () => {
      const res = await inviteAdminByEmail(email);
      if (res.ok) {
        setEmail("");
        setMsg({
          ok: true,
          text: res.status === "granted" ? "Admin granted." : "Invite recorded — they'll become admin on their next sign-in.",
        });
        router.refresh();
      } else {
        setMsg({ ok: false, text: res.error });
      }
    });
  }

  function revokeSelf() {
    if (!window.confirm("Revoke your OWN admin privileges? You cannot undo this yourself.")) return;
    startTransition(async () => {
      const res = await selfRevokeAdmin();
      if (res.ok) router.push("/");
      else setMsg({ ok: false, text: res.error });
    });
  }

  function requestRemoval(entry: AdminEntry) {
    const reason = window.prompt(`Reason for requesting removal of ${entry.displayName} (optional):`) ?? undefined;
    startTransition(async () => {
      const res = await requestAdminRemoval(entry.userId, reason);
      if (res.ok) {
        setMsg({
          ok: true,
          text: res.removed
            ? `${entry.displayName} was removed (${res.votes}/${res.required} votes reached).`
            : `Request recorded: ${res.votes}/${res.required} votes.`,
        });
        router.refresh();
      } else {
        setMsg({ ok: false, text: res.error });
      }
    });
  }

  function cancelRemoval(entry: AdminEntry) {
    startTransition(async () => {
      const res = await cancelAdminRemoval(entry.userId);
      if (res.ok) router.refresh();
      else setMsg({ ok: false, text: res.error });
    });
  }

  return (
    <div className="space-y-8">
      <section className="rounded-xl border border-line bg-surface p-5">
        <h2 className="font-serif text-lg font-semibold">Invite an admin</h2>
        <p className="mt-1 text-sm text-muted">
          {entries.length} / 20 admins. Grant admin by email — if they haven&rsquo;t signed in yet, the
          grant applies automatically on their next sign-in.
        </p>
        <form onSubmit={invite} className="mt-3 flex flex-wrap gap-2">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="person@example.com"
            required
            disabled={atCap}
            className="flex-1 rounded-lg border border-line bg-canvas px-3 py-2 text-sm"
          />
          <button
            type="submit"
            disabled={pending || atCap}
            className="rounded-full bg-brand px-5 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {atCap ? "At cap (20)" : "Invite"}
          </button>
        </form>
        {msg && <p className={msg.ok ? "mt-2 text-sm text-brand" : "mt-2 text-sm text-danger"}>{msg.text}</p>}

        {invites.length > 0 && (
          <div className="mt-4">
            <p className="text-xs font-medium uppercase tracking-wide text-muted">Pending invites</p>
            <ul className="mt-1 text-sm text-muted">
              {invites.map((i) => (
                <li key={i.email}>{i.email}</li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <section>
        <h2 className="font-serif text-lg font-semibold">Current admins</h2>
        <p className="mt-1 text-sm text-muted">
          Removing another admin requires <strong>{requiredVotes}</strong> distinct removal requests
          (a majority, capped at 10). You may only revoke your <em>own</em> privileges directly.
        </p>

        <ul className="mt-4 space-y-3">
          {entries.map((a) => (
            <li key={a.userId} className="flex flex-wrap items-center gap-3 rounded-xl border border-line bg-surface p-4">
              <Avatar src={a.avatarUrl} name={a.displayName} size={40} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{a.displayName}</span>
                  {a.isViewer && <span className="rounded bg-brand-soft px-1.5 py-0.5 text-xs text-brand">you</span>}
                </div>
                <p className="text-sm text-muted">@{a.handle}</p>
              </div>

              {a.votes > 0 && (
                <span className="text-sm text-muted">
                  {a.votes}/{requiredVotes} removal votes
                </span>
              )}

              {a.isViewer ? (
                <button
                  onClick={revokeSelf}
                  disabled={pending}
                  className="rounded-full border border-line px-3 py-1.5 text-sm hover:text-danger"
                >
                  Revoke my admin
                </button>
              ) : a.viewerRequested ? (
                <button
                  onClick={() => cancelRemoval(a)}
                  disabled={pending}
                  className="rounded-full border border-line px-3 py-1.5 text-sm hover:bg-brand-soft"
                >
                  Cancel my request
                </button>
              ) : (
                <button
                  onClick={() => requestRemoval(a)}
                  disabled={pending}
                  className="rounded-full border border-danger px-3 py-1.5 text-sm text-danger hover:bg-danger hover:text-white"
                >
                  Request removal
                </button>
              )}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
