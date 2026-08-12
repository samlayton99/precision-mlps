import Link from "next/link";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

async function count(table: string, filter?: (q: any) => any) {
  const supabase = createClient();
  let q = supabase.from(table).select("*", { count: "exact", head: true });
  if (filter) q = filter(q);
  const { count } = await q;
  return count ?? 0;
}

export default async function AdminOverview() {
  const [openFlags, admins, boards, posts, members] = await Promise.all([
    count("moderation_flags", (q) => q.eq("status", "open")),
    count("admins"),
    count("boards"),
    count("posts"),
    count("profiles"),
  ]);

  const stats = [
    { label: "Open flags", value: openFlags, href: "/admin/moderation", highlight: openFlags > 0 },
    { label: "Admins", value: `${admins} / 20`, href: "/admin/admins" },
    { label: "Boards", value: boards, href: "/admin/boards" },
    { label: "Essays", value: posts, href: "/" },
    { label: "Members", value: members, href: "/admin/users" },
  ];

  return (
    <div>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        {stats.map((s) => (
          <Link
            key={s.label}
            href={s.href}
            className={`rounded-xl border p-4 ${s.highlight ? "border-danger bg-danger/5" : "border-line bg-surface"}`}
          >
            <div className="text-2xl font-semibold">{s.value}</div>
            <div className="text-sm text-muted">{s.label}</div>
          </Link>
        ))}
      </div>

      <div className="mt-8 rounded-xl border border-line bg-surface p-5 text-sm text-muted">
        <p className="font-medium text-ink">Admin responsibilities</p>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          <li>Review the moderation queue daily — the automated scan flags possible guideline violations for your judgment.</li>
          <li>Act with charity. Prefer hiding + a note over deletion when a member can correct course.</li>
          <li>Every privileged action is recorded in the audit log.</li>
        </ul>
      </div>
    </div>
  );
}
