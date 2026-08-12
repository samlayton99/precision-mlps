-- ===========================================================================
-- 0004_seed.sql — starter discussion boards
--
-- Inserted directly (migrations run as owner, bypassing RLS). Admins can
-- rename, add, archive, or delete these from the admin dashboard after launch.
-- ===========================================================================

insert into public.boards (slug, name, description, sort_order) values
  ('doctrine',        'Doctrine & Scripture',
     'Careful essays on scripture, doctrine, and the teachings of the Restoration.', 10),
  ('faith-and-reason', 'Faith & Reason',
     'Where the life of the mind meets the life of faith — philosophy, epistemology, and honest inquiry.', 20),
  ('discipleship',    'Discipleship & Christlike Living',
     'Coming unto Christ in practice: repentance, covenants, charity, and daily discipleship.', 30),
  ('church-history',  'Church History',
     'Reflections on the Restoration, its people, and its unfolding story.', 40),
  ('science-and-faith', 'Science & Faith',
     'Truth from every source: essays at the intersection of scientific understanding and revealed truth.', 50),
  ('building-zion',   'Building Zion',
     'Society, service, family, and community — the practical work of establishing Zion.', 60)
on conflict (slug) do nothing;
