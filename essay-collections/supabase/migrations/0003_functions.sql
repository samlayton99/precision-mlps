-- ===========================================================================
-- 0003_functions.sql — privileged RPCs (governance, moderation, boards)
--
-- Every function here is SECURITY DEFINER: it runs with the table owner's
-- rights and therefore bypasses RLS *on purpose*. Each one re-checks the
-- caller's authority itself and writes an audit_log row. This is where all
-- admin power is concentrated, so it is deliberately small and readable.
--
-- Removal threshold: an admin is auto-removed when the number of *other*
-- admins requesting their removal reaches `required_removal_votes()`. Per the
-- spec ("at least 10, or a majority of current admins"), that is the smaller
-- of 10 and a strict majority. Change the one line in that function to make
-- removal stricter (e.g. GREATEST for "10 AND a majority").
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- admin_invites — lets an existing admin grant admin to an email that has not
-- signed in yet. The grant is applied automatically on that person's next
-- sign-in (see ensure_admin_from_invite, called by the app).
-- ---------------------------------------------------------------------------
create table if not exists public.admin_invites (
  email      citext primary key,
  invited_by uuid references auth.users (id),
  created_at timestamptz not null default now()
);
alter table public.admin_invites enable row level security;
create policy admin_invites_read_admin on public.admin_invites
  for select using (public.is_admin(auth.uid()));

-- ---------------------------------------------------------------------------
-- Internal helpers.
-- ---------------------------------------------------------------------------
create or replace function public._require_admin()
returns uuid
language plpgsql
stable
security definer
set search_path = public
as $$
declare uid uuid := auth.uid();
begin
  if uid is null then
    raise exception 'not authenticated' using errcode = '28000';
  end if;
  if not public.is_admin(uid) then
    raise exception 'admin privilege required' using errcode = '42501';
  end if;
  return uid;
end;
$$;

create or replace function public._audit(
  p_action text, p_target_type text, p_target_id text, p_meta jsonb default '{}'::jsonb)
returns void
language sql
security definer
set search_path = public
as $$
  insert into public.audit_log (actor_id, action, target_type, target_id, meta)
  values (auth.uid(), p_action, p_target_type, p_target_id, coalesce(p_meta, '{}'::jsonb));
$$;

create or replace function public.required_removal_votes()
returns int
language sql
stable
security definer
set search_path = public
as $$
  -- smaller of 10 and a strict majority of current admins.
  select least(10, ((select count(*) from public.admins) / 2)::int + 1);
$$;

-- ===========================================================================
-- Admin roster management
-- ===========================================================================

-- Grant admin to an existing user id (must be under the cap, target not banned).
create or replace function public.grant_admin(p_target uuid)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare caller uuid := public._require_admin();
begin
  if p_target is null or not exists (select 1 from public.profiles where id = p_target) then
    raise exception 'target user does not exist';
  end if;
  if public.is_banned(p_target) then
    raise exception 'cannot grant admin to a banned user';
  end if;
  if exists (select 1 from public.admins where user_id = p_target) then
    return; -- already admin, idempotent
  end if;
  insert into public.admins (user_id, granted_by) values (p_target, caller); -- cap enforced by trigger
  perform public._audit('admin.grant', 'user', p_target::text, '{}'::jsonb);
end;
$$;

-- Invite by email: grant now if the account exists, else record a pending
-- invite to be applied on their first sign-in.
create or replace function public.invite_admin_by_email(p_email citext)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  caller uuid := public._require_admin();
  target uuid;
begin
  if (select count(*) from public.admins) >= 20 then
    raise exception 'admin cap reached: a maximum of 20 admins is allowed';
  end if;
  select id into target from auth.users where email = p_email limit 1;
  if target is not null then
    perform public.grant_admin(target);
    return jsonb_build_object('status', 'granted', 'user_id', target);
  end if;
  insert into public.admin_invites (email, invited_by)
  values (p_email, caller)
  on conflict (email) do update set invited_by = excluded.invited_by, created_at = now();
  perform public._audit('admin.invite', 'email', p_email::text, '{}'::jsonb);
  return jsonb_build_object('status', 'invited');
end;
$$;

-- Applied on sign-in (app calls this). Grants admin if a pending invite exists
-- and the cap allows. Safe to call on every sign-in.
create or replace function public.ensure_admin_from_invite()
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  uid uuid := auth.uid();
  em  citext;
begin
  if uid is null then return false; end if;
  select email into em from auth.users where id = uid;
  if em is null then return false; end if;
  if not exists (select 1 from public.admin_invites where email = em) then
    return false;
  end if;
  if exists (select 1 from public.admins where user_id = uid) then
    delete from public.admin_invites where email = em;
    return true;
  end if;
  if (select count(*) from public.admins) >= 20 then
    return false; -- keep the invite; a slot may open later
  end if;
  insert into public.admins (user_id, granted_by)
    values (uid, (select invited_by from public.admin_invites where email = em));
  delete from public.admin_invites where email = em;
  perform public._audit('admin.invite.accept', 'user', uid::text, '{}'::jsonb);
  return true;
end;
$$;

-- Revoke your OWN admin (the only self-service revoke).
create or replace function public.self_revoke_admin()
returns void
language plpgsql
security definer
set search_path = public
as $$
declare uid uuid := auth.uid();
begin
  if uid is null then raise exception 'not authenticated'; end if;
  delete from public.admins where user_id = uid;
  delete from public.admin_removal_requests where requested_by = uid or target_admin_id = uid;
  perform public._audit('admin.self_revoke', 'user', uid::text, '{}'::jsonb);
end;
$$;

-- Request removal of ANOTHER admin. When distinct requesters reach the
-- threshold, the target is removed atomically. Returns current tally.
create or replace function public.request_admin_removal(p_target uuid, p_reason text default null)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  caller   uuid := public._require_admin();
  required int;
  tally    int;
  removed  boolean := false;
begin
  if p_target = caller then
    raise exception 'use self_revoke_admin to remove your own privileges';
  end if;
  if not exists (select 1 from public.admins where user_id = p_target) then
    raise exception 'target is not an admin';
  end if;

  insert into public.admin_removal_requests (target_admin_id, requested_by, reason)
  values (p_target, caller, p_reason)
  on conflict (target_admin_id, requested_by) do update set reason = excluded.reason;

  required := public.required_removal_votes();
  select count(*) into tally
    from public.admin_removal_requests
    where target_admin_id = p_target
      and requested_by in (select user_id from public.admins where user_id <> p_target);

  if tally >= required then
    delete from public.admins where user_id = p_target;
    delete from public.admin_removal_requests where target_admin_id = p_target or requested_by = p_target;
    removed := true;
    perform public._audit('admin.removed_by_vote', 'user', p_target::text,
                          jsonb_build_object('votes', tally, 'required', required));
  else
    perform public._audit('admin.removal_request', 'user', p_target::text,
                          jsonb_build_object('votes', tally, 'required', required));
  end if;

  return jsonb_build_object('removed', removed, 'votes', tally, 'required', required);
end;
$$;

-- Withdraw your own removal request.
create or replace function public.cancel_admin_removal(p_target uuid)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare caller uuid := public._require_admin();
begin
  delete from public.admin_removal_requests
    where target_admin_id = p_target and requested_by = caller;
end;
$$;

-- ===========================================================================
-- User bans
-- ===========================================================================
create or replace function public.set_user_ban(p_target uuid, p_banned boolean, p_reason text default null)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare caller uuid := public._require_admin();
begin
  if p_banned and exists (select 1 from public.admins where user_id = p_target) then
    raise exception 'remove admin privileges before banning an admin';
  end if;
  update public.profiles
    set is_banned = p_banned,
        banned_reason = case when p_banned then p_reason else null end,
        banned_at = case when p_banned then now() else null end,
        banned_by = case when p_banned then caller else null end
    where id = p_target;
  perform public._audit(case when p_banned then 'user.ban' else 'user.unban' end,
                        'user', p_target::text, jsonb_build_object('reason', p_reason));
end;
$$;

-- ===========================================================================
-- Content moderation (admin)
-- ===========================================================================
create or replace function public.admin_delete_post(p_post uuid, p_reason text default null)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare caller uuid := public._require_admin();
begin
  perform public._audit('post.delete', 'post', p_post::text, jsonb_build_object('reason', p_reason));
  delete from public.posts where id = p_post;
end;
$$;

-- Hide/unhide ("temporarily pause") a post without deleting it.
create or replace function public.admin_set_post_hidden(p_post uuid, p_hidden boolean, p_reason text default null)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare caller uuid := public._require_admin();
begin
  update public.posts
    set status = case when p_hidden then 'hidden' else 'published' end,
        hidden_reason = case when p_hidden then p_reason else null end,
        hidden_by = case when p_hidden then caller else null end,
        hidden_at = case when p_hidden then now() else null end
    where id = p_post;
  perform public._audit(case when p_hidden then 'post.hide' else 'post.unhide' end,
                        'post', p_post::text, jsonb_build_object('reason', p_reason));
end;
$$;

create or replace function public.admin_delete_comment(p_comment uuid, p_reason text default null)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare caller uuid := public._require_admin();
begin
  perform public._audit('comment.delete', 'comment', p_comment::text, jsonb_build_object('reason', p_reason));
  delete from public.comments where id = p_comment;
end;
$$;

create or replace function public.admin_delete_chat_message(p_msg uuid, p_reason text default null)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare caller uuid := public._require_admin();
begin
  perform public._audit('chat.delete', 'chat', p_msg::text, jsonb_build_object('reason', p_reason));
  delete from public.board_chat_messages where id = p_msg;
end;
$$;

-- ===========================================================================
-- Boards (admin)
-- ===========================================================================
create or replace function public.create_board(p_slug citext, p_name text, p_description text default null)
returns uuid
language plpgsql
security definer
set search_path = public
as $$
declare caller uuid := public._require_admin(); new_id uuid;
begin
  insert into public.boards (slug, name, description, created_by)
  values (p_slug, p_name, p_description, caller)
  returning id into new_id;
  perform public._audit('board.create', 'board', new_id::text, jsonb_build_object('slug', p_slug));
  return new_id;
end;
$$;

create or replace function public.delete_board(p_board uuid)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare caller uuid := public._require_admin();
begin
  perform public._audit('board.delete', 'board', p_board::text, '{}'::jsonb);
  delete from public.boards where id = p_board; -- cascades to posts/comments/chat
end;
$$;

-- ===========================================================================
-- Moderation flags (admin resolution)
-- ===========================================================================
create or replace function public.resolve_moderation_flag(p_flag uuid, p_status text)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare caller uuid := public._require_admin();
begin
  if p_status not in ('dismissed', 'actioned') then
    raise exception 'status must be dismissed or actioned';
  end if;
  update public.moderation_flags
    set status = p_status, resolved_by = caller, resolved_at = now()
    where id = p_flag;
  perform public._audit('moderation.resolve', 'flag', p_flag::text, jsonb_build_object('status', p_status));
end;
$$;

-- ---------------------------------------------------------------------------
-- Execute grants: authenticated users may call; each function re-checks
-- authority internally. anon cannot call any of these.
-- ---------------------------------------------------------------------------
grant execute on function
  public.grant_admin(uuid),
  public.invite_admin_by_email(citext),
  public.ensure_admin_from_invite(),
  public.self_revoke_admin(),
  public.request_admin_removal(uuid, text),
  public.cancel_admin_removal(uuid),
  public.required_removal_votes(),
  public.set_user_ban(uuid, boolean, text),
  public.admin_delete_post(uuid, text),
  public.admin_set_post_hidden(uuid, boolean, text),
  public.admin_delete_comment(uuid, text),
  public.admin_delete_chat_message(uuid, text),
  public.create_board(citext, text, text),
  public.delete_board(uuid),
  public.resolve_moderation_flag(uuid, text)
to authenticated;
