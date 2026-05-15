/**
 * HTTP transport: native fetch + bearer auth + exponential backoff.
 *
 * Retries 429 / 5xx with full jitter. Honors Retry-After headers when present.
 * Throws typed ExtremisError subclasses on non-retryable failures.
 */

import {
  ExtremisAuthError,
  ExtremisError,
  ExtremisNetworkError,
  ExtremisRateLimitError,
} from "./errors.js";

export interface TransportOptions {
  apiKey: string;
  baseUrl: string;
  timeoutMs: number;
  maxRetries: number;
  baseDelayMs: number;
  maxDelayMs: number;
  fetch: typeof fetch;
}

export class Transport {
  private readonly opts: TransportOptions;

  constructor(opts: TransportOptions) {
    this.opts = opts;
  }

  async post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>("POST", path, body);
  }

  async postEmpty(path: string, body: unknown): Promise<void> {
    await this.request<unknown>("POST", path, body, true);
  }

  async get<T>(path: string, query?: Record<string, string | undefined>): Promise<T> {
    const qs = query ? buildQuery(query) : "";
    return this.request<T>("GET", `${path}${qs}`, undefined);
  }

  /**
   * Server defines POST /v1/attention/score with query params (not body).
   * Use this for that one endpoint.
   */
  async postWithQuery<T>(path: string, query: Record<string, string | undefined>): Promise<T> {
    return this.request<T>("POST", `${path}${buildQuery(query)}`, undefined);
  }

  private async request<T>(
    method: string,
    path: string,
    body: unknown,
    allowEmpty = false,
  ): Promise<T> {
    const url = `${this.opts.baseUrl}${path}`;
    const headers: Record<string, string> = {
      Authorization: `Bearer ${this.opts.apiKey}`,
    };
    let payload: BodyInit | undefined;
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      payload = JSON.stringify(body);
    }

    let lastError: unknown = null;
    const maxAttempts = this.opts.maxRetries + 1;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.opts.timeoutMs);
      let response: Response;
      try {
        response = await this.opts.fetch(url, {
          method,
          headers,
          body: payload,
          signal: controller.signal,
        });
      } catch (err) {
        clearTimeout(timer);
        lastError = err;
        // Network/abort error — retry if budget allows
        if (attempt < maxAttempts - 1) {
          await sleep(this.backoffDelay(attempt, null));
          continue;
        }
        throw new ExtremisNetworkError(
          `Network error calling ${method} ${path}: ${(err as Error)?.message ?? err}`,
          { cause: err, method, path },
        );
      }
      clearTimeout(timer);

      if (response.ok) {
        if (response.status === 204 || allowEmpty) {
          // 204 No Content — drain and return undefined
          await response.text().catch(() => "");
          return undefined as T;
        }
        const text = await response.text();
        if (!text) return undefined as T;
        try {
          return JSON.parse(text) as T;
        } catch {
          throw new ExtremisError(`Invalid JSON in response from ${method} ${path}`, {
            status: response.status,
            body: text,
            method,
            path,
          });
        }
      }

      // Error path — decide retry vs throw
      const errBody = await parseBody(response);
      const retryable = response.status === 429 || response.status >= 500;
      if (retryable && attempt < maxAttempts - 1) {
        const retryAfter = parseRetryAfter(response.headers.get("Retry-After"));
        await sleep(this.backoffDelay(attempt, retryAfter));
        continue;
      }

      const message = errorMessage(response.status, errBody);
      if (response.status === 401 || response.status === 403) {
        throw new ExtremisAuthError(message, {
          status: response.status,
          body: errBody,
          method,
          path,
        });
      }
      if (response.status === 429) {
        throw new ExtremisRateLimitError(message, {
          status: response.status,
          body: errBody,
          method,
          path,
          retryAfterMs: parseRetryAfter(response.headers.get("Retry-After")),
        });
      }
      throw new ExtremisError(message, {
        status: response.status,
        body: errBody,
        method,
        path,
      });
    }

    // Unreachable in practice — loop exits via return/throw
    throw lastError ?? new ExtremisError(`Exhausted retries calling ${method} ${path}`, {
      status: 0,
      body: null,
      method,
      path,
    });
  }

  private backoffDelay(attempt: number, retryAfterMs: number | null): number {
    if (retryAfterMs !== null && retryAfterMs > 0) {
      return Math.min(retryAfterMs, this.opts.maxDelayMs);
    }
    // Full jitter: random uniform in [0, base * 2^attempt], capped at maxDelayMs
    const expBackoff = Math.min(
      this.opts.baseDelayMs * 2 ** attempt,
      this.opts.maxDelayMs,
    );
    return Math.floor(Math.random() * expBackoff);
  }
}

function buildQuery(params: Record<string, string | undefined>): string {
  const entries = Object.entries(params).filter(
    (pair): pair is [string, string] => pair[1] !== undefined && pair[1] !== "",
  );
  if (entries.length === 0) return "";
  const usp = new URLSearchParams(entries);
  return `?${usp.toString()}`;
}

async function parseBody(response: Response): Promise<unknown> {
  const text = await response.text().catch(() => "");
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function parseRetryAfter(header: string | null): number | null {
  if (!header) return null;
  const asNumber = Number(header);
  if (!Number.isNaN(asNumber)) return asNumber * 1000;
  const asDate = Date.parse(header);
  if (!Number.isNaN(asDate)) return Math.max(0, asDate - Date.now());
  return null;
}

function errorMessage(status: number, body: unknown): string {
  if (body && typeof body === "object" && body !== null) {
    const detail = (body as Record<string, unknown>).detail;
    if (typeof detail === "string") return `${status}: ${detail}`;
  }
  if (typeof body === "string" && body) return `${status}: ${body.slice(0, 200)}`;
  return `HTTP ${status}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
