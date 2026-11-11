-- Run this migration in the Supabase SQL editor.
-- Supabase owns auth.users and auth.identities. These tables store app data.

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  first_name text,
  last_name text,
  avatar_url text,
  height numeric,
  weight numeric,
  handedness text not null default 'right'
    check (handedness in ('left', 'right')),
  backhand_style text
    check (backhand_style in ('one-handed', 'two-handed')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, display_name, avatar_url)
  values (
    new.id,
    coalesce(new.raw_user_meta_data ->> 'full_name', new.raw_user_meta_data ->> 'name'),
    coalesce(new.raw_user_meta_data ->> 'avatar_url', new.raw_user_meta_data ->> 'picture')
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

create table if not exists public.analysis_jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  original_filename text not null,
  stored_filename text not null,
  dominant_arm text not null
    check (dominant_arm in ('left', 'right')),
  backhand text
    check (backhand in ('one-handed', 'two-handed')),
  status text not null default 'queued'
    check (status in ('queued', 'processing', 'complete', 'failed')),
  error_message text,
  created_at timestamptz not null default now(),
  completed_at timestamptz
);

create index if not exists analysis_jobs_user_id_created_at_idx
  on public.analysis_jobs (user_id, created_at desc);

create table if not exists public.analysis_results (
  id uuid primary key default gen_random_uuid(),
  job_id uuid not null unique references public.analysis_jobs(id) on delete cascade,
  duration_seconds double precision,
  strokes jsonb not null default '[]'::jsonb,
  counts jsonb not null default '{}'::jsonb,
  comments jsonb not null default '[]'::jsonb,
  overlay_path text,
  strokes_csv_path text,
  features_path text,
  created_at timestamptz not null default now()
);

alter table public.profiles enable row level security;
alter table public.analysis_jobs enable row level security;
alter table public.analysis_results enable row level security;

create policy "Users can read their own profile"
  on public.profiles for select
  using (auth.uid() = id);

create policy "Users can update their own profile"
  on public.profiles for update
  using (auth.uid() = id)
  with check (auth.uid() = id);

create policy "Users can read their own jobs"
  on public.analysis_jobs for select
  using (auth.uid() = user_id);

create policy "Users can read results for their own jobs"
  on public.analysis_results for select
  using (
    exists (
      select 1
      from public.analysis_jobs
      where analysis_jobs.id = analysis_results.job_id
        and analysis_jobs.user_id = auth.uid()
    )
  );
