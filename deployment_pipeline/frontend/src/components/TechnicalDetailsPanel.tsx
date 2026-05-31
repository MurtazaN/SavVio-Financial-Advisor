/**
 * Collapsible technical details panel for the AiChat assistant messages.
 *
 * Displays financial features, evaluation context, and ML layer
 * information from a PredictResponse.
 */

import { useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "./ui/collapsible";
import { fmtMoney, fmtPct, fmtPctRounded, fmtNum, fmtStirMultiple } from "../lib/formatters";
import type { FinancialFeaturesView, PredictResponse } from "../services/api";

// ---------------------------------------------------------------------------
// Severity helpers
// ---------------------------------------------------------------------------

type SeverityLevel = "ok" | "warn" | "critical" | "na";

function SeverityDot({ level }: { level: SeverityLevel }) {
  const cls =
    level === "critical"
      ? "bg-destructive"
      : level === "warn"
        ? "bg-amber-500"
        : level === "ok"
          ? "bg-emerald-500/85"
          : "bg-muted-foreground/35";
  return (
    <span
      className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${cls}`}
      title={level}
      aria-hidden
    />
  );
}

/**
 * CRI = (credit_score - 299) / 550. FICO-style UI bands:
 * <580 -> red, 580-669 -> yellow, >=670 -> green.
 */
const CRI_CREDIT_FAIR = (580 - 299) / 550;
const CRI_CREDIT_GOOD = (670 - 299) / 550;

function severityNeutral(): SeverityLevel {
  return "na";
}

/** DTI: <28% ok; 28-30% warn; >30% critical. */
function severityDTI(v: number): SeverityLevel {
  if (v < 0.28) return "ok";
  if (v <= 0.3) return "warn";
  return "critical";
}

/** MEB: >70% critical; otherwise ok. */
function severityMEB(v: number): SeverityLevel {
  return v > 0.7 ? "critical" : "ok";
}

/** EFM: <3 critical; 3-6 warn; >6 ok. */
function severityEFM(v: number): SeverityLevel {
  if (v < 3) return "critical";
  if (v <= 6) return "warn";
  return "ok";
}

/** AFS: negative -> critical; positive -> ok. */
function severityAFS(v: number, unreliable: boolean): SeverityLevel {
  if (unreliable) return "critical";
  if (v < 0) return "critical";
  if (v > 0) return "ok";
  return "warn";
}

/** FICO bands via inverted CRI. */
function severityCRI(cri: number | null | undefined): SeverityLevel {
  if (cri == null || Number.isNaN(cri)) return "na";
  if (cri < CRI_CREDIT_FAIR) return "critical";
  if (cri < CRI_CREDIT_GOOD) return "warn";
  return "ok";
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

/** Strip catalog noise after the first vertical bar. */
export function truncateProductNameAtPipe(
  raw: string | null | undefined,
  emptyFallback: string,
): string {
  const trimmed = (raw ?? "").trim();
  if (!trimmed) return emptyFallback;
  const beforeBar = trimmed.split(/[|｜]/)[0]?.trim() ?? "";
  return beforeBar || emptyFallback;
}

function resolveFinancialFeatures(res: PredictResponse): FinancialFeaturesView {
  if (res.financial_features) return res.financial_features;
  return {
    discretionary_income: null,
    debt_to_income_ratio: res.debt_to_income_ratio ?? null,
    saving_to_income_ratio: null,
    monthly_expense_burden_ratio: null,
    emergency_fund_months: res.emergency_fund_months ?? null,
    affordability_score: res.affordability_score ?? null,
    price_to_income_ratio: null,
    residual_utility_score: null,
    savings_to_price_ratio: null,
    net_worth_indicator: null,
    credit_risk_indicator: null,
  };
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function TechnicalFeatureRow({
  abbrev,
  fullLabel,
  value,
  severity,
}: {
  abbrev: string;
  fullLabel: string;
  value: ReactNode;
  severity: SeverityLevel;
}) {
  return (
    <div className="grid grid-cols-[4.5rem_minmax(0,1fr)_auto] items-center gap-x-2 border-b border-border/20 py-1 text-[11px] last:border-b-0">
      <span className="font-mono text-[10px] text-muted-foreground">{abbrev}</span>
      <span className="min-w-0 text-muted-foreground" title={fullLabel}>
        {fullLabel}
      </span>
      <span className="inline-flex max-w-full flex-wrap items-baseline justify-end gap-x-1.5 gap-y-0.5 text-right font-mono tabular-nums text-foreground/90">
        <SeverityDot level={severity} />
        {value}
      </span>
    </div>
  );
}

function SectionBlock({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-md border border-border/30 bg-background/40 px-2.5 py-2">
      <h4 className="mb-2 border-b border-border/25 pb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </h4>
      {children}
    </section>
  );
}

function Subheading({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <p className={`mb-1 text-[10px] font-medium text-muted-foreground/90 ${className ?? ""}`}>
      {children}
    </p>
  );
}

// ---------------------------------------------------------------------------
// TechnicalDetailsBody — the full debug view
// ---------------------------------------------------------------------------

function TechnicalDetailsBody({ res }: { res: PredictResponse }) {
  const ff = resolveFinancialFeatures(res);
  const di = ff.discretionary_income ?? 0;
  const dti = ff.debt_to_income_ratio ?? 0;
  const stirRaw = ff.saving_to_income_ratio;
  const meb = ff.monthly_expense_burden_ratio ?? 0;
  const efm = ff.emergency_fund_months ?? 0;
  const afs = ff.affordability_score ?? 0;
  const pir = ff.price_to_income_ratio;
  const rus = ff.residual_utility_score;
  const spr = ff.savings_to_price_ratio;
  const nwi = ff.net_worth_indicator;
  const cri = ff.credit_risk_indicator;

  const finalL = (res.recommendation ?? "—").toUpperCase();

  const mode =
    res.evaluation_mode === "catalog"
      ? "catalog"
      : res.evaluation_mode === "hypothetical"
        ? "hypothetical"
        : res.evaluation_mode === "none"
          ? "none"
          : String(res.evaluation_mode ?? "—");

  const mlReady =
    res.confidence != null && !Number.isNaN(res.confidence) && res.ml_unavailable_reason == null;
  const modelName = res.ml_model_name ?? "SavVio classifier";

  return (
    <div className="space-y-3 pt-2">
      {/* 1 — Financial features */}
      <SectionBlock title="1 · Financial features">
        <Subheading>User profile</Subheading>
        <TechnicalFeatureRow abbrev="DI" fullLabel="Discretionary income" value={fmtMoney(di)} severity={severityNeutral()} />
        <TechnicalFeatureRow abbrev="DTI" fullLabel="Debt-to-income ratio" value={fmtPct(dti)} severity={severityDTI(dti)} />
        <TechnicalFeatureRow
          abbrev="STIR"
          fullLabel="Liquid savings ÷ monthly income"
          value={stirRaw != null && Number.isFinite(stirRaw) ? fmtStirMultiple(stirRaw) : "—"}
          severity={stirRaw != null && Number.isFinite(stirRaw) ? severityNeutral() : "na"}
        />
        <TechnicalFeatureRow abbrev="MEB" fullLabel="Monthly expense burden" value={fmtPct(meb)} severity={severityMEB(meb)} />
        <TechnicalFeatureRow abbrev="EFM" fullLabel="Emergency fund (months)" value={fmtNum(efm, 2)} severity={severityEFM(efm)} />

        <Subheading className="mt-2">Purchase pair</Subheading>
        <TechnicalFeatureRow
          abbrev="AFS"
          fullLabel="Affordability score"
          value={
            res.affordability_score_unreliable ? (
              "unreliable"
            ) : afs < 0 ? (
              <>
                <span className="text-destructive">{fmtNum(afs, 2)}</span>
                <span className="font-sans text-[9px] font-normal text-muted-foreground">
                  below affordability threshold
                </span>
              </>
            ) : (
              fmtNum(afs, 2)
            )
          }
          severity={severityAFS(afs, !!res.affordability_score_unreliable)}
        />
        <TechnicalFeatureRow abbrev="PIR" fullLabel="Price-to-income ratio" value={pir != null && !Number.isNaN(pir) ? fmtNum(pir, 4) : "—"} severity={pir != null && !Number.isNaN(pir) ? severityNeutral() : "na"} />
        <TechnicalFeatureRow abbrev="RUS" fullLabel="Residual utility score" value={rus != null && !Number.isNaN(rus) ? fmtNum(rus, 4) : "—"} severity={rus != null && !Number.isNaN(rus) ? severityNeutral() : "na"} />
        <TechnicalFeatureRow abbrev="SPR" fullLabel="Savings-to-price ratio" value={spr != null && !Number.isNaN(spr) ? fmtNum(spr, 2) : "—"} severity={spr != null && !Number.isNaN(spr) ? severityNeutral() : "na"} />
        <TechnicalFeatureRow abbrev="NWI" fullLabel="Net worth indicator" value={nwi != null && !Number.isNaN(nwi) ? fmtNum(nwi, 2) : "—"} severity={nwi != null && !Number.isNaN(nwi) ? severityNeutral() : "na"} />
        <TechnicalFeatureRow abbrev="CRI" fullLabel="Credit risk indicator" value={cri != null && !Number.isNaN(cri) ? fmtNum(cri, 4) : "—"} severity={cri != null && !Number.isNaN(cri) ? severityCRI(cri) : "na"} />
      </SectionBlock>

      {/* 2 — Evaluation context */}
      <SectionBlock title="2 · Evaluation context">
        <div className="space-y-1 text-[11px]">
          <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2 border-b border-border/15 py-1 font-mono tabular-nums">
            <span className="text-muted-foreground">Product</span>
            <span className="text-right text-foreground/90 break-words">{truncateProductNameAtPipe(res.product_name, "—")}</span>
          </div>
          <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2 border-b border-border/15 py-1 font-mono tabular-nums">
            <span className="text-muted-foreground">Price</span>
            <span className="text-right">
              {res.product_price != null && !Number.isNaN(res.product_price)
                ? `$${res.product_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`
                : "—"}
            </span>
          </div>
          <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2 border-b border-border/15 py-1 font-mono tabular-nums">
            <span className="text-muted-foreground">Evaluation mode</span>
            <span className="text-right uppercase">{mode}</span>
          </div>
          <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2 py-1">
            <span className="text-muted-foreground">Recommendation</span>
            <span className="text-right font-mono font-semibold text-foreground">{finalL}</span>
          </div>
        </div>
      </SectionBlock>

      {/* 3 — ML layer */}
      <SectionBlock title="3 · ML layer">
        {mlReady ? (
          <div className="space-y-1 text-[11px]">
            <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2 border-b border-border/15 py-1">
              <span className="text-muted-foreground">Confidence</span>
              <span className="text-right font-mono tabular-nums">{(res.confidence! * 100).toFixed(1)}%</span>
            </div>
            <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2 py-1">
              <span className="text-muted-foreground">Predicted label</span>
              <span className="text-right font-mono">{(res.ml_predicted_label ?? "—").toUpperCase()}</span>
            </div>
          </div>
        ) : (
          <p className="font-mono text-[11px] text-muted-foreground/95">
            {modelName}: pending integration
          </p>
        )}
      </SectionBlock>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Exported collapsible wrapper
// ---------------------------------------------------------------------------

export function TechnicalDetailsPanel({ res }: { res: PredictResponse }) {
  const [open, setOpen] = useState(false);
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mt-3 rounded-lg border border-border/35 bg-muted/15 px-3 py-2">
      <CollapsibleTrigger className="flex w-full items-center gap-2 text-left text-xs font-medium text-muted-foreground hover:text-foreground/75">
        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden
        />
        <span>{open ? "Technical details \u2191" : "Technical details \u2193"}</span>
      </CollapsibleTrigger>
      <CollapsibleContent className="data-[state=closed]:animate-none">
        <TechnicalDetailsBody res={res} />
      </CollapsibleContent>
    </Collapsible>
  );
}
