-- ===========================================================================
-- 0001_init.sql — schema, indexes, and maintenance triggers
--
-- Essay Collections data model. Everything lives in the `public` schema and keys
-- off Supabase's `auth.users`. Security (RLS policies) is applied in 0002;
-- privileged governance/admin operations are defined in 0003.
-- ===========================================================================

create extension if not exists pgcrypto;   -- gen_random_uuid()
create extension if not exists citext;      -- case-insensitive handles/slugs

-- ---------------------------------------------------------------------------
-- profiles — one row per authenticated user, created automatically on signup.
-- ---------------------------------------------------------------------------
create table public.profiles (
  id            uuid primary key references auth.users (id) on delete cascade,
  handle        citext unique not null,
  display_name  text not null,
  avatar_url    text,
  bio           text check (bio is null or char_length(bio) <= 500),
  is_banned     boolean not null default false,
  banned_reason text,
  banned_at     timestamptz,
  banned_by     uuid references auth.users (id),
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- admins — membership table. Hard cap of 20 enforced by trigger below.
-- Granting/revoking happens only through the SECURITY DEFINER RPCs in 0003.
-- ---------------------------------------------------------------------------
-- NOTE on foreign keys: author/user columns that the UI needs to join to a
-- profile reference public.profiles(id) (which is itself id -> auth.users.id).
-- This is equivalent to referencing auth.users but lets PostgREST embed the
-- author profile directly (e.g. posts?select=*,author:profiles(...)).
create table public.admins (
  user_id    uuid primary key references public.profiles (id) on delete cascade,
  granted_by uuid references public.profiles (id) on delete set null,
  granted_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- boards — "discussion boards": the top-level buckets essays are organized in.
-- ---------------------------------------------------------------------------
create table public.boards (
  id          uuid primary key default gen_random_uuid(),
  slug        citext unique not null check (slug ~ '^[a-z0-9]([a-z0-9-]{0,48}[a-z0-9])?$'),
  name        text not null check (char_length(name) between 1 and 80),
  description text check (description is null or char_length(description) <= 500),
  created_by  uuid references auth.users (id),
  is_archived boolean not null default false,
  sort_order  int not null default 0,
  created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- posts — the essays. `content_html` is sanitized server-side before insert.
-- like_count / comment_count are denormalized for cheap sorting and kept in
-- sync by triggers.
-- ---------------------------------------------------------------------------
create table public.posts (
  id            uuid primary key default gen_random_uuid(),
  board_id      uuid not null references public.boards (id) on delete cascade,
  author_id     uuid not null references public.profiles (id) on delete cascade,
  title         text not null check (char_length(title) between 1 and 200),
  subtitle      text check (subtitle is null or char_length(subtitle) <= 300),
  content_html  text not null,
  excerpt       text check (excerpt is null or char_length(excerpt) <= 400),
  cover_image_url text,
  status        text not null default 'published'
                  check (status in ('published', 'hidden', 'draft')),
  hidden_reason text,
  hidden_by     uuid references auth.users (id),
  hidden_at     timestamptz,
  like_count    int not null default 0,
  comment_count int not null default 0,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now(),
  published_at  timestamptz
);

-- ---------------------------------------------------------------------------
-- comments — text only, threaded flat under a post.
-- ---------------------------------------------------------------------------
create table public.comments (
  id         uuid primary key default gen_random_uuid(),
  post_id    uuid not null references public.posts (id) on delete cascade,
  author_id  uuid not null references public.profiles (id) on delete cascade,
  body       text not null check (char_length(body) between 1 and 10000),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- post_likes — one like per (post, user).
-- ---------------------------------------------------------------------------
create table public.post_likes (
  post_id    uuid not null references public.posts (id) on delete cascade,
  user_id    uuid not null references public.profiles (id) on delete cascade,
  created_at timestamptz not null default now(),
  primary key (post_id, user_id)
);

-- ---------------------------------------------------------------------------
-- mentions — "tag other users". Written by the server when it parses @handles
-- out of a post or comment. Drives notifications.
-- ---------------------------------------------------------------------------
create table public.mentions (
  id                uuid primary key default gen_random_uuid(),
  source_type       text not null check (source_type in ('post', 'comment')),
  source_id         uuid not null,
  mentioned_user_id uuid not null references public.profiles (id) on delete cascade,
  author_id         uuid not null references public.profiles (id) on delete cascade,
  created_at        timestamptz not null default now(),
  unique (source_type, source_id, mentioned_user_id)
);

-- ---------------------------------------------------------------------------
-- board_chat_messages — each board's own global chat. Text only, realtime.
-- ---------------------------------------------------------------------------
create table public.board_chat_messages (
  id         uuid primary key default gen_random_uuid(),
  board_id   uuid not null references public.boards (id) on delete cascade,
  author_id  uuid not null references public.profiles (id) on delete cascade,
  body       text not null check (char_length(body) between 1 and 2000),
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- notifications — mentions, replies, admin notices. Recipient = user_id.
-- ---------------------------------------------------------------------------
create table public.notifications (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references public.profiles (id) on delete cascade,
  type       text not null check (type in ('mention', 'comment', 'admin')),
  actor_id   uuid references public.profiles (id) on delete set null,
  post_id    uuid references public.posts (id) on delete cascade,
  comment_id uuid references public.comments (id) on delete cascade,
  body       text,
  is_read    boolean not null default false,
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- moderation_flags — raised by the daily bot ('bot') or by users ('user').
-- Reviewed by admins in the moderation dashboard.
-- ---------------------------------------------------------------------------
create table public.moderation_flags (
  id           uuid primary key default gen_random_uuid(),
  content_type text not null check (content_type in ('post', 'comment', 'chat', 'profile')),
  content_id   uuid not null,
  board_id     uuid references public.boards (id) on delete set null,
  category     text,
  severity     text check (severity in ('low', 'medium', 'high')),
  reason       text not null,
  flagged_by   text not null default 'bot' check (flagged_by in ('bot', 'user')),
  reporter_id  uuid references public.profiles (id) on delete set null,
  status       text not null default 'open' check (status in ('open', 'dismissed', 'actioned')),
  resolved_by  uuid references auth.users (id),
  resolved_at  timestamptz,
  created_at   timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- admin_removal_requests — governance. A majority of current admins requesting
-- removal of the same target triggers automatic removal (logic in 0003).
-- ---------------------------------------------------------------------------
create table public.admin_removal_requests (
  id              uuid primary key default gen_random_uuid(),
  target_admin_id uuid not null references public.profiles (id) on delete cascade,
  requested_by    uuid not null references public.profiles (id) on delete cascade,
  reason          text,
  created_at      timestamptz not null default now(),
  unique (target_admin_id, requested_by)
);

-- ---------------------------------------------------------------------------
-- audit_log — append-only record of every privileged action. Admin-readable.
-- ---------------------------------------------------------------------------
create table public.audit_log (
  id          bigint generated always as identity primary key,
  actor_id    uuid references auth.users (id),
  action      text not null,
  target_type text,
  target_id   text,
  meta        jsonb not null default '{}'::jsonb,
  created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- moderation_runs — one row per daily bot run, for observability.
-- ---------------------------------------------------------------------------
create table public.moderation_runs (
  id            uuid primary key default gen_random_uuid(),
  started_at    timestamptz not null default now(),
  finished_at   timestamptz,
  items_scanned int not null default 0,
  flags_created int not null default 0,
  model         text,
  error         text
);

-- ---------------------------------------------------------------------------
-- Indexes for the hot read paths.
-- ---------------------------------------------------------------------------
create index posts_board_created_idx   on public.posts (board_id, created_at desc);
create index posts_author_idx          on public.posts (author_id);
create index posts_status_idx          on public.posts (status);
create index posts_published_idx       on public.posts (published_at desc) where status = 'published';
create index comments_post_created_idx on public.comments (post_id, created_at);
create index comments_author_idx       on public.comments (author_id);
create index post_likes_user_idx       on public.post_likes (user_id);
create index chat_board_created_idx    on public.board_chat_messages (board_id, created_at desc);
create index notifications_user_idx    on public.notifications (user_id, is_read, created_at desc);
create index mentions_user_idx         on public.mentions (mentioned_user_id);
create index mod_flags_status_idx      on public.moderation_flags (status, created_at desc);
create index removal_req_target_idx    on public.admin_removal_requests (target_admin_id);

-- ===========================================================================
-- Maintenance triggers
-- ===========================================================================

-- keep updated_at fresh -------------------------------------------------------
create or replace function public.touch_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

create trigger profiles_touch before update on public.profiles
  for each row execute function public.touch_updated_at();
create trigger posts_touch before update on public.posts
  for each row execute function public.touch_updated_at();
create trigger comments_touch before update on public.comments
  for each row execute function public.touch_updated_at();

-- stamp published_at when a post first becomes published ----------------------
create or replace function public.stamp_published_at()
returns trigger language plpgsql as $$
begin
  if new.status = 'published' and new.published_at is null then
    new.published_at := now();
  end if;
  return new;
end;
$$;

create trigger posts_stamp_published before insert or update on public.posts
  for each row execute function public.stamp_published_at();

-- denormalized like_count -----------------------------------------------------
create or replace function public.sync_like_count()
returns trigger language plpgsql as $$
begin
  if tg_op = 'INSERT' then
    update public.posts set like_count = like_count + 1 where id = new.post_id;
  elsif tg_op = 'DELETE' then
    update public.posts set like_count = greatest(like_count - 1, 0) where id = old.post_id;
  end if;
  return null;
end;
$$;

create trigger post_likes_count after insert or delete on public.post_likes
  for each row execute function public.sync_like_count();

-- denormalized comment_count --------------------------------------------------
create or replace function public.sync_comment_count()
returns trigger language plpgsql as $$
begin
  if tg_op = 'INSERT' then
    update public.posts set comment_count = comment_count + 1 where id = new.post_id;
  elsif tg_op = 'DELETE' then
    update public.posts set comment_count = greatest(comment_count - 1, 0) where id = old.post_id;
  end if;
  return null;
end;
$$;

create trigger comments_count after insert or delete on public.comments
  for each row execute function public.sync_comment_count();

-- enforce the hard cap of 20 admins ------------------------------------------
create or replace function public.enforce_admin_cap()
returns trigger language plpgsql as $$
begin
  if (select count(*) from public.admins) >= 20 then
    raise exception 'admin cap reached: a maximum of 20 admins is allowed'
      using errcode = 'check_violation';
  end if;
  return new;
end;
$$;

create trigger admins_cap before insert on public.admins
  for each row execute function public.enforce_admin_cap();

-- ---------------------------------------------------------------------------
-- New-user bootstrap: create a profile row for every new auth user. A unique,
-- URL-safe @handle is derived from the email local-part, deduped with a
-- counter. Admin bootstrap happens app-side (needs the env allowlist).
-- ---------------------------------------------------------------------------
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  base_handle text;
  candidate   text;
  n           int := 0;
begin
  base_handle := lower(regexp_replace(split_part(coalesce(new.email, 'saint'), '@', 1),
                                       '[^a-z0-9_]', '_', 'g'));
  base_handle := left(nullif(base_handle, ''), 24);
  base_handle := coalesce(base_handle, 'saint');
  candidate := base_handle;

  while exists (select 1 from public.profiles where handle = candidate) loop
    n := n + 1;
    candidate := left(base_handle, 20) || '_' || n::text;
  end loop;

  insert into public.profiles (id, handle, display_name, avatar_url)
  values (
    new.id,
    candidate,
    coalesce(nullif(new.raw_user_meta_data ->> 'full_name', ''),
             nullif(new.raw_user_meta_data ->> 'name', ''),
             candidate),
    new.raw_user_meta_data ->> 'avatar_url'
  );
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ---------------------------------------------------------------------------
-- Realtime: board chat streams to subscribed clients.
-- ---------------------------------------------------------------------------
alter publication supabase_realtime add table public.board_chat_messages;
alter publication supabase_realtime add table public.comments;
