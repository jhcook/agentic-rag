<!-- 241134d4-5976-47e8-b937-846772349ed7 fcfa00ca-10a2-408e-a5d9-a93fd44d7494 -->
# Code Compliance Review and Mitigation Plan

## Executive Summary

This plan addresses violations of SOC 2, GDPR, and Lean Code rules identified in the codebase. Key areas include missing type hints, PII logging, documentation gaps, and code quality issues.

## Identified Violations

### 1. Missing Type Hints (Python Rules Violation)

**Files Affected:**

- `src/core/rag_core.py` - `search()` function (line 974) missing return type annotation
- `src/servers/rest_server.py` - Multiple functions missing return types:
- `_proxy_to_mcp()` (line 242)
- `api_upsert()` (line 728)
- `api_index_url()` (line 769)
- `api_index_path()` (line 784)
- `api_search()` (line 797)
- `api_load()` (line 850)
- `metrics_endpoint()` (line 861)
- `api_grounded_answer()` (line 881)
- `api_rerank()` (line 903)
- `api_verify()` (line 916)
- `api_verify_simple()` (line 926)
- And ~20+ more endpoint functions

**Impact:** Violates Python type hinting requirement, reduces code clarity and IDE support.

### 2. PII Logging (GDPR Violation)

**Files Affected:**

- `src/servers/mcp_app/logging_config.py` - Logs client IP addresses (line 140)
- `src/core/google_backend.py` - Logs email IDs (line 602)
- `src/servers/rest_server.py` - Access logs may contain IP addresses

**Impact:** GDPR violation - personal data (IP addresses, email IDs) logged in plaintext without redaction or justification.

### 3. Secrets Handling (SOC 2 Security)

**Files Affected:**

- `src/core/openai_assistants_backend.py` - API keys loaded from files (acceptable) but need verification they're not logged
- `src/servers/rest_server.py` - API key masking in responses (line 1468) - good practice, but need audit

**Impact:** Potential secret exposure if error messages or logs contain API keys.

### 4. HTTP vs HTTPS (SOC 2 Transport Security)

**Files Affected:**

- Multiple files use hardcoded `http://` URLs for localhost (acceptable for localhost)
- No explicit HTTPS enforcement for production deployments

**Impact:** Acceptable for localhost, but production deployments should enforce HTTPS.

### 5. Missing Documentation (SOC 2 + GDPR Evidence)

**Gaps:**

- No documented GDPR lawful basis for IP address collection
- No documented retention/deletion policy for access logs
- No documented purpose for email ID logging in Google backend
- Missing docstrings on some functions (need comprehensive audit)

**Impact:** Violates SOC 2 auditability and GDPR documentation requirements.

### 6. Test Coverage Gaps

**Areas Needing Tests:**

- Google backend authentication flows
- OpenAI Assistants backend error handling
- PII redaction functions (if implemented)
- Access log middleware

**Impact:** SOC 2 processing integrity - non-trivial logic without tests.

## Mitigation Plan

### Phase 1: Critical Security & Privacy Fixes

1. **PII Redaction in Logs**

- Add IP address redaction utility function
- Update `AccessLoggingMiddleware` in `src/servers/mcp_app/logging_config.py` to redact IP addresses
- Update `google_backend.py` to redact email IDs in logs
- Add configuration flag to enable/disable PII logging for debugging

2. **Secrets Audit**

- Audit all error logging to ensure API keys are never logged
- Add secret detection in CI/CD pipeline
- Verify `.gitignore` properly excludes `secrets/` directory

### Phase 2: Type Hints & Code Quality

3. **Add Missing Type Hints**

- Add return type annotations to all functions in `src/core/rag_core.py`
- Add return type annotations to all REST API endpoints in `src/servers/rest_server.py`
- Run `mypy` or `pyright` to identify remaining type issues
- Update `scripts/check_rules_compliance.py` to check for missing type hints

4. **Docstring Audit**

- Run pylint with docstring checks to identify missing docstrings
- Add docstrings to all public functions and classes
- Ensure docstrings follow project conventions

### Phase 3: Documentation & Compliance

5. **GDPR Documentation**

- Create `docs/gdpr-compliance.md` documenting:
- Lawful basis for IP address collection (legitimate interest for security)
- Retention period for access logs (30 days recommended)
- Deletion mechanism for logs
- Purpose of email ID logging (Google Drive integration)
- Update README to reference GDPR documentation

6. **SOC 2 Documentation**

- Document HTTPS enforcement for production deployments
- Document secret management practices
- Add security considerations to API documentation

### Phase 4: Testing & Validation

7. **Test Coverage**

- Add tests for PII redaction functions
- Add tests for access log middleware
- Add tests for Google backend authentication error paths
- Add tests for OpenAI backend error handling

8. **Compliance Validation**

- Update `scripts/check_rules_compliance.py` to include:
- Type hint checking
- PII detection in logs
- Secret detection
- Docstring coverage
- Run compliance checks in CI/CD

## Implementation Priority

**High Priority (Blocking):**

- PII redaction in logs (GDPR violation)
- Secrets audit (SOC 2 security)

**Medium Priority:**

- Type hints (code quality)
- Documentation (compliance evidence)

**Low Priority:**

- Test coverage improvements
- HTTPS enforcement documentation

## Files to Modify

1. `src/servers/mcp_app/logging_config.py` - Add PII redaction
2. `src/core/google_backend.py` - Redact email IDs in logs
3. `src/core/rag_core.py` - Add type hints
4. `src/servers/rest_server.py` - Add type hints to all endpoints
5. `docs/gdpr-compliance.md` - Create GDPR documentation (new file)
6. `scripts/check_rules_compliance.py` - Add compliance checks
7. `README.md` - Reference GDPR documentation

## Success Criteria

- All functions have type hints
- No PII logged in plaintext (IP addresses, emails redacted)
- GDPR documentation exists with lawful basis and retention policies
- Compliance script passes all checks
- No secrets in code or logs