import { redirect } from "next/navigation";
import type { Metadata } from "next";
import { getViewer } from "@/lib/auth";
import { siteConfig } from "@/config/site";
import { SignInButton } from "@/components/SignInButton";

export const metadata: Metadata = { title: "Sign in" };

export default async function LoginPage({
  searchParams,
}: {
  searchParams: { next?: string };
}) {
  const viewer = await getViewer();
  const next = typeof searchParams.next === "string" ? searchParams.next : "/";
  if (viewer) redirect(next);

  return (
    <div className="mx-auto flex max-w-md flex-col items-center gap-6 py-16 text-center">
      <h1 className="font-serif text-3xl font-semibold">Welcome to {siteConfig.name}</h1>
      <p className="text-muted">
        {siteConfig.description}
      </p>
      <SignInButton next={next} />
      <p className="text-sm text-muted">
        By signing in you agree to our{" "}
        <a href="/guidelines" className="text-brand underline">
          Community Guidelines
        </a>
        . We use your Google account only to sign you in.
      </p>
    </div>
  );
}
