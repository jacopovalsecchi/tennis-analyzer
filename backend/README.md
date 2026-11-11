# Backend setup

## Supabase

1. Create a Supabase project.
2. In the SQL editor, run [`supabase.sql`](supabase.sql).
3. Copy `.env.example` to `.env`.
4. Fill in `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `SUPABASE_SERVICE_ROLE_KEY`.
5. Set `SUPABASE_AUTH_REQUIRED=true` when the frontend is ready to send Supabase access tokens.

The service role key is server-only. Never expose it to the browser or commit it.

## Endpoints

- `GET /health` — reports whether Supabase auth and persistence are configured.
- `POST /upload` — accepts a `video` multipart field and creates a user-owned analysis job.
- `GET /jobs` — lists the authenticated user's jobs.
- `GET /jobs/:id` — returns one authenticated user's job and result.

Authenticated requests must send:

```text
Authorization: Bearer <supabase-access-token>
```

When `SUPABASE_AUTH_REQUIRED=false`, the local prototype still accepts unauthenticated uploads and uses the existing `latest.json` compatibility path.
