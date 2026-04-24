-- Run in Supabase SQL editor (PostgreSQL)
-- Storage: create bucket `audio-files` (public or signed URLs per your policy)

create extension if not exists "uuid-ossp";

create table if not exists public.audio_logs (
  id uuid primary key default uuid_generate_v4(),
  audio_url text,
  transcript text,
  sounds jsonb default '[]'::jsonb,
  emotion text,
  answer text,
  question text,
  created_at timestamptz default now()
);

alter table public.audio_logs enable row level security;

-- Example: allow anon insert/read for demo (tighten for production)
create policy "Allow insert for authenticated service role"
  on public.audio_logs for insert
  with check (true);

create policy "Allow select for anon"
  on public.audio_logs for select
  using (true);
