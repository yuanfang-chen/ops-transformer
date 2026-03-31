-- Run in Supabase SQL Editor

create table if not exists public.machines (
  id text primary key,
  name text not null,
  model text not null,
  location text not null,
  remark text default '',
  created_at timestamptz default now()
);

create table if not exists public.bookings (
  id text primary key,
  machine_id text not null references public.machines(id) on delete cascade,
  date text not null,
  start_hour int not null check (start_hour >= 0 and start_hour <= 23),
  end_hour int not null check (end_hour >= 1 and end_hour <= 24),
  user_name text not null,
  purpose text not null,
  created_at timestamptz default now()
);

create index if not exists idx_bookings_machine_date on public.bookings(machine_id, date);

alter table public.machines enable row level security;
alter table public.bookings enable row level security;

drop policy if exists "machines_read_write" on public.machines;
create policy "machines_read_write" on public.machines
for all
to anon, authenticated
using (true)
with check (true);

drop policy if exists "bookings_read_write" on public.bookings;
create policy "bookings_read_write" on public.bookings
for all
to anon, authenticated
using (true)
with check (true);
