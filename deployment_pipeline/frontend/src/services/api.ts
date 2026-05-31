/**
 * SavVio backend API client (proxied via Vite `/api` → FastAPI).
 */

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export interface PredictRequestBody {
  user_query: string;
  user_id: string;
  product_id?: string | null;
}

/** Financial inputs: user profile ratios + affordability features. */
export interface FinancialFeaturesView {
  discretionary_income?: number | null;
  debt_to_income_ratio?: number | null;
  saving_to_income_ratio?: number | null;
  monthly_expense_burden_ratio?: number | null;
  emergency_fund_months?: number | null;
  affordability_score?: number | null;
  price_to_income_ratio?: number | null;
  residual_utility_score?: number | null;
  savings_to_price_ratio?: number | null;
  net_worth_indicator?: number | null;
  credit_risk_indicator?: number | null;
}

export interface PredictResponse {
  recommendation: string;
  /** ML model confidence; null when the model is not loaded or did not score. */
  confidence: number | null;
  /** When confidence is null: no_model | no_pipeline | scoring_error */
  ml_unavailable_reason?: string | null;
  explanation: string;
  product_name?: string | null;
  product_price?: number | null;
  /** "catalog" | "hypothetical" | "none" */
  evaluation_mode?: string;
  /** Discretionary income minus price; may be clamped when unreliable. */
  affordability_score?: number | null;
  affordability_score_unreliable?: boolean;
  /** Months of expenses covered by liquid savings. */
  emergency_fund_months?: number | null;
  /** Debt payments / income, 0-1. */
  debt_to_income_ratio?: number | null;
  financial_features?: FinancialFeaturesView | null;
  /** ML classifier predicted label (GREEN/YELLOW/RED). */
  ml_predicted_label?: string | null;
  /** Display name for ML model. */
  ml_model_name?: string;
}

export interface UserProfileResponse {
  user_id: string;
  monthly_income: number | null;
  monthly_expenses: number | null;
  savings_balance: number | null;
  has_loan: number | null;
  loan_amount: number | null;
  monthly_emi: number | null;
  loan_interest_rate: number | null;
  loan_term_months: number | null;
  credit_score: number | null;
  employment_status: string | null;
  region: string | null;
  liquid_savings: number | null;
  discretionary_income: number | null;
  debt_to_income_ratio: number | null;
  saving_to_income_ratio: number | null;
  monthly_expense_burden_ratio: number | null;
  emergency_fund_months: number | null;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  db_connected: boolean;
  llm_provider: string;
  version: string;
}

export interface ProductListItem {
  product_id: string;
  product_name: string;
  price: number | null;
  average_rating: number | null;
  rating_number: number | null;
}

export interface ProductListResponse {
  items: ProductListItem[];
  total: number;
  price_min_applied: number;
  price_max_applied: number;
  limit: number;
  offset: number;
}

export interface FetchProductsParams {
  q?: string;
  price_min?: number;
  price_max?: number;
  limit?: number;
  offset?: number;
}

/** Human-readable fallback when the response has no JSON body. */
function statusFallbackMessage(status: number, statusText: string): string {
  switch (status) {
    case 502:
    case 503:
    case 504:
      return `Service temporarily unavailable (HTTP ${status}). Try again in a moment.`;
    case 401:
      return "Unauthorized — check API credentials if required.";
    case 403:
      return "Access denied for this resource.";
    case 408:
      return "Request timed out. Try again.";
    case 429:
      return "Too many requests. Wait briefly and retry.";
    default:
      return statusText?.trim() || `Request failed (HTTP ${status}).`;
  }
}

/**
 * Parse error payload from a non-OK response.
 * Supports SavVio `ErrorResponse` `{ error, detail }`, FastAPI `detail` string or list, and plain text.
 */
async function readHttpErrorMessage(res: Response): Promise<string> {
  let text = "";
  try {
    text = await res.text();
  } catch {
    return statusFallbackMessage(res.status, res.statusText);
  }
  const trimmed = text.trim();
  if (!trimmed) {
    return statusFallbackMessage(res.status, res.statusText);
  }
  try {
    const j = JSON.parse(trimmed) as Record<string, unknown>;
    if (typeof j.detail === "string" && j.detail.length > 0) {
      return j.detail;
    }
    if (Array.isArray(j.detail)) {
      const msgs = j.detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            return String((item as { msg: unknown }).msg);
          }
          return null;
        })
        .filter((x): x is string => Boolean(x?.trim()));
      if (msgs.length) return msgs.join("; ");
    }
    if (typeof j.message === "string" && j.message.length > 0) {
      return j.message;
    }
  } catch {
    return trimmed.length > 400 ? `${trimmed.slice(0, 397)}…` : trimmed;
  }
  return statusFallbackMessage(res.status, res.statusText);
}

async function assertOk(res: Response): Promise<void> {
  if (res.ok) return;
  const detail = await readHttpErrorMessage(res);
  throw new Error(detail);
}

/** `fetch` wrapper: turns connection failures into clear user-facing errors. */
async function apiFetch(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (e) {
    if (e instanceof TypeError) {
      throw new Error(
        "Cannot reach the SavVio API (network error or server not running). Check your connection, proxy/Vite `VITE_API_BASE`, and that the backend is up.",
      );
    }
    throw e instanceof Error ? e : new Error("Network request failed.");
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await apiFetch(`${API_BASE}/health`);
  await assertOk(res);
  return res.json() as Promise<HealthResponse>;
}

export async function fetchUserProfile(userId: string): Promise<UserProfileResponse> {
  const trimmed = userId.trim();
  const id = encodeURIComponent(trimmed);
  const res = await apiFetch(`${API_BASE}/user/${id}/profile`);
  if (!res.ok) {
    if (res.status === 404) {
      throw new Error(
        `No profile for “${trimmed}”. This user ID is not in the database—check the ID or use an account that exists in SavVio.`,
      );
    }
    await assertOk(res);
  }
  return res.json() as Promise<UserProfileResponse>;
}

export async function fetchProducts(params?: FetchProductsParams): Promise<ProductListResponse> {
  const sp = new URLSearchParams();
  if (params?.q != null && params.q.trim()) sp.set("q", params.q.trim());
  if (params?.price_min != null) sp.set("price_min", String(params.price_min));
  if (params?.price_max != null) sp.set("price_max", String(params.price_max));
  if (params?.limit != null) sp.set("limit", String(params.limit));
  if (params?.offset != null) sp.set("offset", String(params.offset));
  const qs = sp.toString();
  const url = qs ? `${API_BASE}/products?${qs}` : `${API_BASE}/products`;
  const res = await apiFetch(url);
  await assertOk(res);
  return res.json() as Promise<ProductListResponse>;
}

export async function sendPredict(body: PredictRequestBody): Promise<PredictResponse> {
  const res = await apiFetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_query: body.user_query,
      user_id: body.user_id,
      product_id: body.product_id ?? undefined,
    }),
  });
  if (!res.ok) {
    const detail = await readHttpErrorMessage(res);
    if (res.status === 422) {
      throw new Error(`Invalid request: ${detail}`);
    }
    if (res.status >= 500) {
      throw new Error(`${detail} If this keeps happening, check API logs and database connectivity.`);
    }
    throw new Error(detail);
  }
  return res.json() as Promise<PredictResponse>;
}
