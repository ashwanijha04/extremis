/**
 * Typed errors thrown by the SDK. Catch ExtremisError to inspect status
 * code, parsed body, and request context.
 */

export class ExtremisError extends Error {
  readonly status: number;
  readonly body: unknown;
  readonly method: string;
  readonly path: string;

  constructor(
    message: string,
    init: { status: number; body: unknown; method: string; path: string },
  ) {
    super(message);
    this.name = "ExtremisError";
    this.status = init.status;
    this.body = init.body;
    this.method = init.method;
    this.path = init.path;
  }
}

export class ExtremisAuthError extends ExtremisError {
  constructor(
    message: string,
    init: { status: number; body: unknown; method: string; path: string },
  ) {
    super(message, init);
    this.name = "ExtremisAuthError";
  }
}

export class ExtremisRateLimitError extends ExtremisError {
  readonly retryAfterMs: number | null;

  constructor(
    message: string,
    init: {
      status: number;
      body: unknown;
      method: string;
      path: string;
      retryAfterMs: number | null;
    },
  ) {
    super(message, init);
    this.name = "ExtremisRateLimitError";
    this.retryAfterMs = init.retryAfterMs;
  }
}

export class ExtremisNetworkError extends Error {
  override readonly cause: unknown;
  readonly method: string;
  readonly path: string;

  constructor(message: string, init: { cause: unknown; method: string; path: string }) {
    super(message);
    this.name = "ExtremisNetworkError";
    this.cause = init.cause;
    this.method = init.method;
    this.path = init.path;
  }
}
