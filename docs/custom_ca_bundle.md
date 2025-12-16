# Custom CA Bundle Configuration

This guide explains how to create and use a custom Certificate Authority (CA) bundle with the Agentic RAG application. This is useful when running the application behind a corporate proxy or when using self-signed certificates.

## 1. Finding Your Certificates

You'll need the PEM-formatted certificate(s) for your custom CA or self-signed server. These often have extensions like `.pem`, `.crt`, or `.cer`.

## 2. Creating a Combined Bundle

A CA bundle is simply a text file containing multiple certificates concatenated together. All certificates should be in PEM format (Base64 encoded ASCII starting with `-----BEGIN CERTIFICATE-----`).

### Method 1: Python One-Liner (Cross-Platform)

If you have `certifi` installed (usually available in Python environments), you can easily create a bundle that includes both standard trusted CAs and your custom certificate.

Run this command in your terminal:

```bash
python -c "import certifi, sys; sys.stdout.write(open(certifi.where()).read() + open('PATH_TO_YOUR_CUSTOM.pem').read())" > custom-ca-bundle.pem
```

Replace `PATH_TO_YOUR_CUSTOM.pem` with the path to your custom certificate file. This command reads the default `certifi` bundle, appends your custom certificate, and writes the result to `custom-ca-bundle.pem`.

### Method 2: OpenSSL (Linux / macOS / Windows with Git Bash)

If you have `openssl` installed, you can convert and combine files.

**Converting DER (.cer/.crt) to PEM if needed:**
```bash
openssl x509 -inform DER -in certificate.cer -out certificate.pem
```

**Combining PEM files:**
```bash
# Combine specific files
cat cert1.pem cert2.pem > custom-ca-bundle.pem

# Or append to system certs (example for Debian/Ubuntu)
cat /etc/ssl/certs/ca-certificates.crt custom-cert.pem > custom-ca-bundle.pem
```

### Method 3: PowerShell (Windows)

You can use PowerShell to combine files comfortably.

```powershell
# Get all cert files in current directory
$certs = Get-ChildItem -Include *.pem,*.crt,*.cer -File

# Combine content
$certs | Get-Content | Set-Content custom-ca-bundle.pem -Encoding Ascii

Write-Host "Combined $($certs.Count) certificates into custom-ca-bundle.pem"
```

**Note:** If your `.cer` or `.crt` files are binary (DER format) and not Base64 (PEM format), PowerShell's `Get-Content` will not work correctly. You should export them as Base64/PEM first (e.g., via Windows Certificate Manager) or use OpenSSL.

The resulting bundle file should look like this:

```text
-----BEGIN CERTIFICATE-----
(Base64 encoded certificate data)
-----END CERTIFICATE-----
-----BEGIN CERTIFICATE-----
(Base64 encoded certificate data)
-----END CERTIFICATE-----
...
```

## 3. Configuration

You can configure the application to use your custom bundle in one of two ways.

### Option A: settings.json (Recommended)

Add the absolute path to your bundle in `config/settings.json`:

```json
{
  "caBundlePath": "/absolute/path/to/custom-ca-bundle.pem"
}
```

### Option B: Environment Variables

Set one of the following environment variables before starting the application:

- `CA_BUNDLE`
- `REQUESTS_CA_BUNDLE`
- `SSL_CERT_FILE`

Example:

```bash
export CA_BUNDLE="/absolute/path/to/custom-ca-bundle.pem"
python start.py
```

## 4. Verification

When properly configured, the application will use this bundle for:
- All external HTTP requests (e.g., to LLM providers)
- Internal MCP server communication
- Vector database connections (depending on driver support)
