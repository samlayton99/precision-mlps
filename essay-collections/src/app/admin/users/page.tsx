import { createClient } from "@/lib/supabase/server";
import { UserModeration, type MemberRow } from "@/components/admin/UserModeration";

export const dynamic = "force-dynamic";

export default async function AdminUsersPage() {
  const supabase = createClient();

  const [{ data: banned }, { data: admins }] = await Promise.all([
    supabase
      .from("profiles")
      .select("id,handle,display_name,avatar_url,is_banned,banned_reason")
      .eq("is_banned", true)
      .order("banned_at", { ascending: false })
      .returns<MemberRow[]>(),
    supabase.from("admins").select("user_id"),
  ]);

  const adminIds = new Set((admins ?? []).map((a) => a.user_id));

  return (
    <UserModeration
      bannedMembers={banned ?? []}
      adminIds={[...adminIds]}
    />
  );
}
