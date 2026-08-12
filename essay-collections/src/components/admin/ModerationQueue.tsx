"use client";

import { useState, useTransition } from "react";
import Link from "next/link";
import { resolveModerationFlag } from "@/app/actions/admin";
import { timeAgo } from "@/lib/utils";
import { cn } from "@/lib/utils";

export interface EnrichedFlag {
  id: string;
  content_type: "post" | "comment" | "chat" | "profile";
  content_id: string;
  category: string | null;
  severity: "low" | "medium" | "high" | null;
  reason: string;
  flagged_by: "bot" | "user";
  created_at: string;
  reporter: { handle: string; display_name: string } | null;
  preview: string;
  href: string | null;
}

const severityColor: Record<string, string> = {
  high: "border-danger text-danger",
  medium: "border-brand text-brand",
  low: "border-line text-muted",
};

export function ModerationQueue({ flags }: { flags: EnrichedFlag[] }) {
  const [items, setItems] = useState(flags);
  const [pending, startTransition] = useTransition();

  function resolve(id: string, status: "dismissed" | "actioned") {
    setItems((prev) => prev.filter((f) => f.id !== id)); // optimistic
    startTransition(async () => {
      await resolveModerationFlag(id, status);
    });
  }

  return (
    <ul className="space-y-4">
      {items.map((f) => (
        <li key={f.id} className="rounded-xl border border-line bg-surface p-4">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded bg-brand-soft px-2 py-0.5 font-medium uppercase tracking-wide text-brand">
              {f.content_type}
            </span>
            {f.severity && (
              <span className={cn("rounded border px-2 py-0.5 font-medium uppercase", severityColor[f.severity])}>
                {f.severity}
              </span>
            )}
            {f.category && <span className="text-muted">{f.category}</span>}
            <span className="text-muted">·</span>
            <span className="text-muted">
              {f.flagged_by === "bot" ? "Automated scan" : `Reported by @${f.reporter?.handle ?? "member"}`}
            </span>
            <span className="text-muted">·</span>
            <span className="text-muted">{timeAgo(f.created_at)}</span>
          </div>

          <p className="mt-2 text-sm">
            <span className="font-medium">Reason: </span>
            {f.reason}
          </p>
          <p className="mt-1 line-clamp-3 rounded bg-canvas px-3 py-2 text-sm text-muted">
            “{f.preview}”
          </p>

          <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
            {f.href && (
              <Link href={f.href} className="rounded-full border border-line px-3 py-1.5 hover:bg-brand-soft" target="_blank">
                View content →
              </Link>
            )}
            <button
              onClick={() => resolve(f.id, "actioned")}
              disabled={pending}
              className="rounded-full border border-danger px-3 py-1.5 text-danger hover:bg-danger hover:text-white"
            >
              Mark actioned
            </button>
            <button
              onClick={() => resolve(f.id, "dismissed")}
              disabled={pending}
              className="rounded-full border border-line px-3 py-1.5 hover:bg-brand-soft"
            >
              Dismiss
            </button>
          </div>
          <p className="mt-2 text-xs text-muted">
            Use “View content” to hide, delete, or ban from the content itself; then mark this actioned.
          </p>
        </li>
      ))}
    </ul>
  );
}
