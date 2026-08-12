import Link from "next/link";
import { requireAdmin } from "@/lib/auth";

export default async function AdminLayout({ children }: { children: React.ReactNode }) {
  await requireAdmin(); // guards every /admin/* route

  const tabs = [
    { href: "/admin", label: "Overview" },
    { href: "/admin/moderation", label: "Moderation" },
    { href: "/admin/admins", label: "Admins" },
    { href: "/admin/boards", label: "Boards" },
    { href: "/admin/users", label: "Members" },
  ];

  return (
    <div>
      <div className="mb-8 border-b border-line">
        <h1 className="font-serif text-2xl font-semibold">Admin</h1>
        <nav className="mt-3 flex flex-wrap gap-1 text-sm">
          {tabs.map((t) => (
            <Link key={t.href} href={t.href} className="rounded-t-lg px-3 py-2 text-muted hover:bg-brand-soft hover:text-ink">
              {t.label}
            </Link>
          ))}
        </nav>
      </div>
      {children}
    </div>
  );
}
