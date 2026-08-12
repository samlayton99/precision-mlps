import { createClient } from "@/lib/supabase/server";
import { getViewer } from "@/lib/auth";
import { AdminGovernance, type AdminEntry } from "@/components/admin/AdminGovernance";

export const dynamic = "force-dynamic";

interface AdminRow {
  user_id: string;
  granted_at: string;
  profile: { handle: string; display_name: string; avatar_url: string | null } | null;
}

export default async function AdminsPage() {
  const supabase = createClient();
  const viewer = await getViewer();

  const [{ data: admins }, { data: requests }, { data: invites }, { data: required }] = await Promise.all([
    supabase
      .from("admins")
      .select("user_id,granted_at,profile:profiles!admins_user_id_fkey(handle,display_name,avatar_url)")
      .order("granted_at")
      .returns<AdminRow[]>(),
    supabase.from("admin_removal_requests").select("target_admin_id,requested_by"),
    supabase.from("admin_invites").select("email,created_at").order("created_at"),
    supabase.rpc("required_removal_votes"),
  ]);

  const reqRows = requests ?? [];
  const entries: AdminEntry[] = (admins ?? []).map((a) => {
    const forTarget = reqRows.filter((r) => r.target_admin_id === a.user_id);
    return {
      userId: a.user_id,
      handle: a.profile?.handle ?? "",
      displayName: a.profile?.display_name ?? "Member",
      avatarUrl: a.profile?.avatar_url ?? null,
      grantedAt: a.granted_at,
      votes: forTarget.length,
      viewerRequested: forTarget.some((r) => r.requested_by === viewer?.id),
      isViewer: a.user_id === viewer?.id,
    };
  });

  return (
    <AdminGovernance
      entries={entries}
      requiredVotes={(required as number | null) ?? 0}
      invites={(invites ?? []).map((i) => ({ email: i.email as string, createdAt: i.created_at as string }))}
      atCap={entries.length >= 20}
    />
  );
}
