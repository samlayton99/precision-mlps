import type { Metadata } from "next";
import { requireViewer } from "@/lib/auth";
import { ProfileSettingsForm } from "@/components/ProfileSettingsForm";

export const metadata: Metadata = { title: "Settings" };

export default async function SettingsPage() {
  const viewer = await requireViewer("/settings");
  return (
    <div className="mx-auto max-w-lg">
      <h1 className="font-serif text-3xl font-semibold">Profile settings</h1>
      <p className="mt-2 text-muted">Signed in with Google as {viewer.email}.</p>
      <div className="mt-8">
        <ProfileSettingsForm
          initial={{
            displayName: viewer.profile.display_name,
            handle: viewer.profile.handle,
            bio: viewer.profile.bio ?? "",
          }}
        />
      </div>
    </div>
  );
}
