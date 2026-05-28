# Safety Guide (Security + Operational Hardening)

This guide defines safe operating practices for local, staging, and production usage.

## 1) Security Baseline

- Never run production with `DEMO_MODE=true`.
- Never keep default `SECRET_KEY` or `JWT_SECRET`.
- Keep `DEBUG=false` in production.
- Restrict `CORS_ORIGINS` to trusted frontend domains only.
- Do not commit `.env` or service-account JSON files.

## 2) Environment Profiles

### Local Development

- `APP_ENV=development`
- `DEBUG=true`
- `DEMO_MODE=true` (allowed only for local demos)
- `FIREBASE_AUTH_ENABLED=false` (unless explicitly testing Firebase auth)

### Production

- `APP_ENV=production`
- `DEBUG=false`
- `DEMO_MODE=false`
- `FIREBASE_AUTH_ENABLED=true`
- Strong secrets for `SECRET_KEY` and `JWT_SECRET`

## 3) Auth and Access Safety

- If `FIREBASE_AUTH_ENABLED=true`, backend expects Firebase ID tokens.
- In this mode, `/auth/login` and `/auth/register` are intentionally disabled.
- Keep frontend and backend auth modes aligned to avoid unintended bypasses.

## 4) Secret Management

- Store secrets in environment variables, not source code.
- Use a secrets manager in cloud deployments.
- Rotate credentials immediately if exposed:
  - Firebase service-account key
  - `SECRET_KEY`
  - `JWT_SECRET`

## 5) Firebase Safety

- Keep Firestore security rules strict before production.
- Restrict service-account file permissions (`chmod 600`).
- Use least-privilege service accounts where possible.
- Audit Firebase Auth and Firestore usage periodically.

## 6) Data and Model Safety

- Trained models (`*.pt`, `*.zip`) are artifacts; avoid storing sensitive data in them.
- Validate incoming API payloads and symbol inputs.
- Treat market-data providers as untrusted external dependencies; handle failures gracefully.

## 7) Operational Checks Before Go-Live

1. Backend starts with production env and no runtime security error.
2. Frontend build passes (`npm run build`).
3. API health endpoint is stable.
4. Auth flow works with Firebase token validation.
5. CORS allows only intended domains.
6. No secret files are tracked by git.
7. Logging does not print secrets/tokens.

## 8) Incident Response Quick Actions

- Revoke exposed keys immediately.
- Rotate app secrets and restart services.
- Invalidate active tokens if compromise is suspected.
- Review recent auth and API activity logs.
- Re-run smoke tests after mitigation.

## 9) References

- Setup runbook: `setup.md`
- Demo behavior details: `docs/DEMO_MODE.md`
- Architecture and runtime flow: `docs/ARCHITECTURE.md`
