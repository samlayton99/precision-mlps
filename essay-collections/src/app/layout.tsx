import type { Metadata } from "next";
import { Inter, Source_Serif_4 } from "next/font/google";
import "./globals.css";
import { siteConfig } from "@/config/site";
import { Nav } from "@/components/Nav";
import { getViewer } from "@/lib/auth";

const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const serif = Source_Serif_4({ subsets: ["latin"], variable: "--font-serif", display: "swap" });

export const metadata: Metadata = {
  title: {
    default: `${siteConfig.name} — ${siteConfig.tagline}`,
    template: `%s · ${siteConfig.name}`,
  },
  description: siteConfig.description,
  metadataBase: new URL(siteConfig.url),
  openGraph: {
    title: siteConfig.name,
    description: siteConfig.description,
    siteName: siteConfig.name,
    type: "website",
  },
  robots: { index: true, follow: true },
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const viewer = await getViewer();
  return (
    <html lang="en" className={`${sans.variable} ${serif.variable}`}>
      <body>
        <Nav
          viewer={
            viewer
              ? {
                  handle: viewer.profile.handle,
                  displayName: viewer.profile.display_name,
                  avatarUrl: viewer.profile.avatar_url,
                  isAdmin: viewer.isAdmin,
                }
              : null
          }
        />
        <main className="mx-auto w-full max-w-5xl px-4 pb-24 pt-8 sm:px-6">{children}</main>
        <footer className="border-t border-line">
          <div className="mx-auto flex max-w-5xl flex-col gap-2 px-4 py-8 text-sm text-muted sm:flex-row sm:items-center sm:justify-between sm:px-6">
            <p>
              © {new Date().getFullYear()} {siteConfig.name}. {siteConfig.tagline}
            </p>
            <nav className="flex gap-4">
              <a href="/guidelines" className="hover:text-ink">Community Guidelines</a>
              <a href="/boards" className="hover:text-ink">Boards</a>
            </nav>
          </div>
        </footer>
      </body>
    </html>
  );
}
