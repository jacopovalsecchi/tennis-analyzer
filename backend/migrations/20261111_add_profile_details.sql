-- Add personal and tennis profile data to an existing Supabase project.
-- height is stored in centimetres; weight is stored in kilograms.

begin;

do $$
begin
  if exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'profiles'
      and column_name = 'default_arm'
  ) and not exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'profiles'
      and column_name = 'handedness'
  ) then
    alter table public.profiles rename column default_arm to handedness;
  end if;

  if exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'profiles'
      and column_name = 'default_backhand'
  ) and not exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'profiles'
      and column_name = 'backhand_style'
  ) then
    alter table public.profiles rename column default_backhand to backhand_style;
  end if;
end
$$;

alter table public.profiles
  add column if not exists first_name text,
  add column if not exists last_name text,
  add column if not exists height numeric,
  add column if not exists weight numeric,
  add column if not exists handedness text not null default 'right',
  add column if not exists backhand_style text;

comment on column public.profiles.height is 'Height in centimetres';
comment on column public.profiles.weight is 'Weight in kilograms';

commit;
