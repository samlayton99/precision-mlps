import type { Metadata } from "next";
import { getViewer } from "@/lib/auth";

export const metadata: Metadata = { title: "Account restricted" };

export default async function BannedPage() {
  const viewer = await getViewer();
  const reason = viewer?.profile.banned_reason;
  return (
    <div className="mx-auto max-w-md py-16 text-center">
      <h1 className="font-serif text-2xl font-semibold">Your account is restricted</h1>
      <p className="mt-4 text-muted">
        You can still read, but posting, commenting, liking, and chat are disabled while your
        account is under moderation review.
      </p>
      {reason && (
        <p className="mt-4 rounded-lg border border-line bg-surface p-4 text-sm">
          <span className="font-medium">Reason given: </span>
          {reason}
        </p>
      )}
      <p className="mt-6 text-sm text-muted">
        If you believe this was a mistake, please reach out to an administrator.
      </p>
    </div>
  );
}
