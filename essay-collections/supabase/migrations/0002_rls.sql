-- ===========================================================================
-- 0002_rls.sql — Row Level Security
--
-- Security model (defense in depth):
--   * The database is the last line of defense. Even if application code has a
--     bug, RLS decides what each caller may read/write.
--   * `is_admin` / `is_banned` are SECURITY DEFINER so they can read the
--     admins/profiles tables without tripping their own RLS (no recursion).
--   * Normal user actions (own content CRUD) are governed by these policies.
--   * Privileged admin actions run through SECURITY DEFINER RPCs (0003), which
--     bypass RLS deliberately and write to the audit_log.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Helper predicates.
-- ---------------------------------------------------------------------------
create or replace function public.is_admin(uid uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (select 1 from public.admins a where a.user_id = uid);
$$;

create or replace function public.is_banned(uid uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (select 1 from public.profiles p where p.id = uid and p.is_banned);
$$;

-- A writer must be signed in and not banned.
create or replace function public.can_write(uid uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select uid is not null and not public.is_banned(uid);
$$;

grant execute on function public.is_admin(uuid)  to anon, authenticated;
grant execute on function public.is_banned(uuid) to anon, authenticated;
grant execute on function public.can_write(uuid) to anon, authenticated;

-- ---------------------------------------------------------------------------
-- Enable RLS everywhere. Default posture: deny. Policies below open the doors.
-- ---------------------------------------------------------------------------
alter table public.profiles              enable row level security;
alter table public.admins                enable row level security;
alter table public.boards                enable row level security;
alter table public.posts                 enable row level security;
alter table public.comments              enable row level security;
alter table public.post_likes            enable row level security;
alter table public.mentions              enable row level security;
alter table public.board_chat_messages   enable row level security;
alter table public.notifications         enable row level security;
alter table public.moderation_flags      enable row level security;
alter table public.admin_removal_requests enable row level security;
alter table public.audit_log             enable row level security;
alter table public.moderation_runs       enable row level security;

-- ===========================================================================
-- profiles
-- ===========================================================================
create policy profiles_read_all on public.profiles
  for select using (true);

create policy profiles_update_own on public.profiles
  for update using (id = auth.uid())
  with check (id = auth.uid());

-- Column-level defense: users may edit only their display fields. Ban state
-- (is_banned, banned_*) is untouchable from the client no matter what the row
-- policy allows — it changes only via the set_user_ban definer RPC.
revoke update on public.profiles from authenticated, anon;
grant update (handle, display_name, avatar_url, bio) on public.profiles to authenticated;

-- ===========================================================================
-- admins — readable by signed-in members (governance transparency inside the
-- community, not to the open internet); all writes via RPC only.
-- ===========================================================================
create policy admins_read_members on public.admins
  for select using (auth.uid() is not null);

-- ===========================================================================
-- boards — public read; writes via RPC (admin) only.
-- ===========================================================================
create policy boards_read_all on public.boards
  for select using (true);

-- ===========================================================================
-- posts
-- ===========================================================================
create policy posts_read on public.posts
  for select using (
    status = 'published'
    or author_id = auth.uid()
    or public.is_admin(auth.uid())
  );

create policy posts_insert_own on public.posts
  for insert with check (
    author_id = auth.uid()
    and public.can_write(auth.uid())
  );

create policy posts_update_own on public.posts
  for update using (author_id = auth.uid() and public.can_write(auth.uid()))
  with check (author_id = auth.uid());

create policy posts_delete_own on public.posts
  for delete using (author_id = auth.uid());

-- ===========================================================================
-- comments — visible when the parent post is visible to the caller.
-- ===========================================================================
create policy comments_read on public.comments
  for select using (
    exists (
      select 1 from public.posts p
      where p.id = comments.post_id
        and (p.status = 'published' or p.author_id = auth.uid() or public.is_admin(auth.uid()))
    )
  );

create policy comments_insert_own on public.comments
  for insert with check (
    author_id = auth.uid()
    and public.can_write(auth.uid())
    and exists (select 1 from public.posts p where p.id = post_id and p.status = 'published')
  );

create policy comments_update_own on public.comments
  for update using (author_id = auth.uid() and public.can_write(auth.uid()))
  with check (author_id = auth.uid());

create policy comments_delete_own on public.comments
  for delete using (author_id = auth.uid());

-- ===========================================================================
-- post_likes
-- ===========================================================================
create policy likes_read_all on public.post_likes
  for select using (true);

create policy likes_insert_own on public.post_likes
  for insert with check (
    user_id = auth.uid()
    and public.can_write(auth.uid())
    and exists (select 1 from public.posts p where p.id = post_id and p.status = 'published')
  );

create policy likes_delete_own on public.post_likes
  for delete using (user_id = auth.uid());

-- ===========================================================================
-- mentions — the mentioned user (and admins) can read; writes via server only.
-- ===========================================================================
create policy mentions_read on public.mentions
  for select using (mentioned_user_id = auth.uid() or author_id = auth.uid() or public.is_admin(auth.uid()));

-- ===========================================================================
-- board_chat_messages — members-only read; not-banned members write.
-- ===========================================================================
create policy chat_read_authed on public.board_chat_messages
  for select using (auth.uid() is not null);

create policy chat_insert_own on public.board_chat_messages
  for insert with check (author_id = auth.uid() and public.can_write(auth.uid()));

create policy chat_delete_own on public.board_chat_messages
  for delete using (author_id = auth.uid());

-- ===========================================================================
-- notifications — recipient only.
-- ===========================================================================
create policy notifications_read_own on public.notifications
  for select using (user_id = auth.uid());

create policy notifications_update_own on public.notifications
  for update using (user_id = auth.uid())
  with check (user_id = auth.uid());

create policy notifications_delete_own on public.notifications
  for delete using (user_id = auth.uid());

-- ===========================================================================
-- moderation_flags — admins read/manage; any member may file a user report.
-- ===========================================================================
create policy mod_flags_read_admin on public.moderation_flags
  for select using (public.is_admin(auth.uid()));

create policy mod_flags_report on public.moderation_flags
  for insert with check (
    flagged_by = 'user'
    and reporter_id = auth.uid()
    and status = 'open'
    and public.can_write(auth.uid())
  );

-- ===========================================================================
-- admin_removal_requests — admins read; writes via RPC only.
-- ===========================================================================
create policy removal_read_admin on public.admin_removal_requests
  for select using (public.is_admin(auth.uid()));

-- ===========================================================================
-- audit_log & moderation_runs — admin read only; writes via definer/service.
-- ===========================================================================
create policy audit_read_admin on public.audit_log
  for select using (public.is_admin(auth.uid()));

create policy mod_runs_read_admin on public.moderation_runs
  for select using (public.is_admin(auth.uid()));
