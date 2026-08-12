import Link from "next/link";
import { siteConfig } from "@/config/site";
import { Avatar } from "@/components/ui/Avatar";

interface NavViewer {
  handle: string;
  displayName: string;
  avatarUrl: string | null;
  isAdmin: boolean;
}

export function Nav({ viewer }: { viewer: NavViewer | null }) {
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-canvas/85 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <div className="flex items-baseline gap-6">
          <Link href="/" className="font-serif text-xl font-semibold tracking-tight">
            {siteConfig.name}
          </Link>
          <nav className="hidden gap-5 text-sm text-muted sm:flex">
            <Link href="/boards" className="hover:text-ink">Boards</Link>
            <Link href="/guidelines" className="hover:text-ink">Guidelines</Link>
          </nav>
        </div>

        <div className="flex items-center gap-3">
          {viewer ? (
            <>
              <Link
                href="/write"
                className="rounded-full bg-brand px-4 py-1.5 text-sm font-medium text-white hover:opacity-90"
              >
                Write
              </Link>
              {viewer.isAdmin && (
                <Link href="/admin" className="hidden text-sm text-muted hover:text-ink sm:block">
                  Admin
                </Link>
              )}
              <Link href={`/u/${viewer.handle}`} className="flex items-center gap-2" title={viewer.displayName}>
                <Avatar src={viewer.avatarUrl} name={viewer.displayName} size={30} />
              </Link>
              <form action="/auth/signout" method="post">
                <button type="submit" className="text-sm text-muted hover:text-ink">
                  Sign out
                </button>
              </form>
            </>
          ) : (
            <Link
              href="/login"
              className="rounded-full border border-line px-4 py-1.5 text-sm font-medium hover:bg-brand-soft"
            >
              Sign in
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
