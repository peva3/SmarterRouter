# Security Policy

## Supported Versions

We actively support the following versions of SmarterRouter with security updates:

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in SmarterRouter, please report it responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via:

1. **GitHub Security Advisories** (preferred):
   - Go to [Security > Advisories](https://github.com/peva3/SmarterRouter/security/advisories)
   - Click "Report a vulnerability"
   - Fill out the form with details

2. **Email** (alternative):
   - Send details to the project maintainers
   - Include "SECURITY" in the subject line

### What to Include

Please include the following information:

- Type of vulnerability (e.g., injection, XSS, authentication bypass)
- Full paths of source files related to the vulnerability
- Step-by-step instructions to reproduce
- Proof-of-concept or exploit code (if available)
- Potential impact and severity
- Suggested fix (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Depends on severity and complexity

### Disclosure Policy

- We follow responsible disclosure
- Vulnerabilities will be disclosed after a fix is released
- We will credit reporters who follow responsible disclosure (unless they prefer anonymity)

## Security Best Practices

When deploying SmarterRouter, follow these security guidelines:

### API Keys and Secrets

1. **Never commit secrets to the repository**
   - Use environment variables
   - The `.env` file is gitignored for a reason
   - Use `ENV_DEFAULT` as a template

2. **Set a strong admin API key in production**:
   ```bash
   ROUTER_ADMIN_API_KEY=$(openssl rand -hex 32)
   ```

3. **Protect judge API keys**:
   - If using LLM-as-Judge, ensure `ROUTER_JUDGE_API_KEY` is secured
   - Never log or expose this key

### Network Security

1. **Bind to localhost only** if not exposing externally:
   ```bash
   ROUTER_HOST=127.0.0.1
   ```

2. **Use HTTPS** in production with a reverse proxy (nginx, Caddy, etc.)

3. **Enable rate limiting**:
   ```bash
   ROUTER_RATE_LIMIT_ENABLED=true
   ROUTER_RATE_LIMIT_REQUESTS_PER_MINUTE=60
   ```

### Docker Security

1. **Run as non-root user** (the Dockerfile creates an unprivileged user)

2. **Use read-only filesystem** when possible:
   ```yaml
   read_only: true
   ```

3. **Limit capabilities**:
   ```yaml
   security_opt:
     - no-new-privileges:true
   ```

4. **Don't expose unnecessary ports**

### Input Validation

SmarterRouter includes several security measures:

- **SQL Injection Prevention**: All database operations use SQLAlchemy ORM
- **Input Sanitization**: Prompts are sanitized for control characters
- **Length Limits**: Enforced on prompts (10k chars) and messages (100 max)
- **Content-Type Validation**: POST endpoints require `application/json`

### Known Security Considerations

1. **Admin Endpoints**: Without `ROUTER_ADMIN_API_KEY`, admin endpoints are publicly accessible. **Always set this in production.**

2. **In-Memory Rate Limiting**: Rate limits are stored in memory and reset on restart. For production with multiple instances, consider external rate limiting (nginx, Kong, etc.).

3. **Logging Sanitization**: API keys and secrets are automatically redacted from logs, but be careful when adding custom logging.

4. **VRAM Monitoring**: Requires `nvidia-smi` access. In Docker, this requires GPU passthrough which has security implications.

## Security Updates

Security updates will be:

1. Announced in GitHub Releases
2. Tagged with security labels
3. Documented in CHANGELOG.md

Subscribe to [GitHub Releases](https://github.com/peva3/SmarterRouter/releases) to be notified of security updates.

## Contact

For security concerns, please use the reporting channels above. For general questions, open a GitHub Discussion.

Thank you for helping keep SmarterRouter secure!
