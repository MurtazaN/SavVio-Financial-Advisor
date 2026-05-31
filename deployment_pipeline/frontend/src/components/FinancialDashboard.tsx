import { motion } from "framer-motion";
import {
  TrendingUp,
  TrendingDown,
  Wallet,
  PiggyBank,
  CreditCard,
  ShieldCheck,
  DollarSign,
  AlertTriangle,
} from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { useUser } from "../context/UserContext";
import { fmtMoney, fmtPct, ratioHealth } from "../lib/formatters";

const container = { hidden: {}, show: { transition: { staggerChildren: 0.06 } } };
const item = { hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0 } };

const CustomTooltipStyle = {
  backgroundColor: "hsl(222,40%,10%)",
  border: "1px solid hsl(222,30%,22%)",
  borderRadius: "8px",
  fontSize: 12,
};

export const FinancialDashboard = () => {
  const { userId, userProfile, profileError, isLoadingProfile } = useUser();

  if (!userId.trim()) {
    return (
      <div className="space-y-4 max-w-2xl">
        <h1 className="font-heading text-xl font-semibold">Financial Health Dashboard</h1>
        <p className="text-sm text-muted-foreground">Enter your **User ID** in the header to load your profile.</p>
      </div>
    );
  }

  if (isLoadingProfile) {
    return (
      <div className="space-y-4">
        <h1 className="font-heading text-xl font-semibold">Financial Health Dashboard</h1>
        <p className="text-sm text-muted-foreground">Loading your profile…</p>
      </div>
    );
  }

  if (profileError || !userProfile) {
    return (
      <div className="space-y-4 max-w-2xl">
        <h1 className="font-heading text-xl font-semibold">Financial Health Dashboard</h1>
        <div className="glass-card p-4 border-destructive/30 flex gap-3">
          <AlertTriangle className="w-5 h-5 text-destructive shrink-0" />
          <p className="text-sm text-destructive">{profileError ?? "Profile unavailable."}</p>
        </div>
      </div>
    );
  }

  const p = userProfile;
  const income = p.monthly_income ?? 0;
  const expenses = p.monthly_expenses ?? 0;
  const emi = p.monthly_emi ?? 0;
  const discretionaryRaw = p.discretionary_income;
  const discretionary =
    discretionaryRaw != null && !Number.isNaN(discretionaryRaw)
      ? Math.max(0, discretionaryRaw)
      : Math.max(0, income - expenses - emi);
  const safeDaily = Math.max(0, discretionary / 30);
  const safeWeekly = safeDaily * 7;
  const liquidSavings = p.liquid_savings ?? p.savings_balance;

  /** Monthly savings *flow* (surplus vs income), not liquid_savings ÷ income (STIR in DB). */
  const monthlyFlowSurplus =
    discretionaryRaw != null && !Number.isNaN(discretionaryRaw)
      ? discretionaryRaw
      : income - expenses - emi;
  const savingsRateMonthly =
    income > 0 && monthlyFlowSurplus != null && !Number.isNaN(monthlyFlowSurplus)
      ? monthlyFlowSurplus / income
      : null;

  const dti = p.debt_to_income_ratio;
  const dtiHealth = ratioHealth(dti, 0.28, 0.36);
  const expenseBurden = p.monthly_expense_burden_ratio;
  const ebHealth = ratioHealth(expenseBurden, 0.5, 0.65);
  const savingsRate = savingsRateMonthly;
  const srHealth =
    savingsRate == null || Number.isNaN(savingsRate)
      ? "unknown"
      : savingsRate >= 0.15
        ? "good"
        : savingsRate >= 0.08
          ? "warn"
          : "bad";
  const emer = p.emergency_fund_months;
  const emerHealth =
    emer == null || Number.isNaN(emer) ? "unknown" : emer >= 6 ? "good" : emer >= 3 ? "warn" : "bad";

  const healthColor = (h: string) =>
    h === "good" ? "text-success" : h === "warn" ? "text-caution" : h === "bad" ? "text-destructive" : "text-muted-foreground";

  const barCompare = [
    { name: "Income", value: income, fill: "hsl(160,72%,40%)" },
    { name: "Expenses", value: expenses, fill: "hsl(215,20%,55%)" },
  ];

  const stats = [
    {
      label: "Monthly Income",
      value: fmtMoney(p.monthly_income),
      change: p.region ? p.region : "—",
      up: true,
      icon: Wallet,
      sub: true,
    },
    {
      label: "Liquid savings",
      value: fmtMoney(liquidSavings),
      change: fmtPct(savingsRateMonthly) + " savings rate (monthly flow)",
      up: (savingsRateMonthly ?? 0) >= 0.1,
      icon: PiggyBank,
      sub: false,
    },
    {
      label: "Debt / Loan",
      value: p.has_loan ? fmtMoney(p.loan_amount) : "None",
      change: p.monthly_emi ? `${fmtMoney(p.monthly_emi)}/mo EMI` : "",
      up: false,
      icon: CreditCard,
      sub: false,
    },
    {
      label: "Credit Score",
      value: p.credit_score != null ? String(p.credit_score) : "—",
      change: p.employment_status ?? "",
      up: true,
      icon: ShieldCheck,
      sub: true,
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-heading text-xl font-semibold mb-1">Financial Health Dashboard</h1>
        <p className="text-sm text-muted-foreground">
          Snapshot from your profile — income, expenses, and key health ratios.
        </p>
      </div>

      {/* Safe to Spend Banner */}
      <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} className="glass-card p-5 border-primary/20">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-2xl bg-primary/15 flex items-center justify-center">
              <DollarSign className="w-7 h-7 text-primary" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">Est. safe to spend / day</p>
              <p className="font-heading text-3xl font-bold text-primary">${Math.round(safeDaily)}</p>
              <p className="text-[11px] text-muted-foreground mt-1">
                From monthly discretionary ({fmtMoney(discretionary)}) ÷ 30
              </p>
            </div>
          </div>
          <div className="flex gap-6 flex-wrap">
            <div className="text-center">
              <p className="text-xs text-muted-foreground">This Week (est.)</p>
              <p className="font-heading text-lg font-semibold">${Math.round(safeWeekly)}</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Discretionary (mo)</p>
              <p className="font-heading text-lg font-semibold">{fmtMoney(discretionary)}</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Monthly income</p>
              <p className="font-heading text-lg font-semibold">{fmtMoney(income)}</p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Stats */}
      <motion.div variants={container} initial="hidden" animate="show" className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s) => (
          <motion.div key={s.label} variants={item} className="glass-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground font-medium">{s.label}</span>
              <s.icon className="w-4 h-4 text-muted-foreground" />
            </div>
            <p className="font-heading text-2xl font-semibold">{s.value}</p>
            {s.change ? (
              <span
                className={`inline-flex items-center gap-1 text-xs font-medium ${
                  s.sub ? "text-muted-foreground" : s.up ? "text-success" : "text-destructive"
                }`}
              >
                {!s.sub && (s.up ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />)}
                {s.change}
              </span>
            ) : null}
          </motion.div>
        ))}
      </motion.div>

      {/* Health gauges */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card p-5">
        <h2 className="font-heading text-sm font-semibold mb-4">Financial health indicators</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="space-y-1">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Debt-to-income</p>
            <p className={`text-lg font-semibold ${healthColor(dtiHealth)}`}>
              {dti != null ? fmtPct(dti) : "—"}
            </p>
            <p className="text-[11px] text-muted-foreground">Lower is better (typ. &lt;28% good)</p>
          </div>
          <div className="space-y-1">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Emergency fund</p>
            <p className={`text-lg font-semibold ${healthColor(emerHealth)}`}>
              {emer != null ? `${emer.toFixed(1)} mo` : "—"}
            </p>
            <p className="text-[11px] text-muted-foreground">Target often 3–6+ months</p>
          </div>
          <div className="space-y-1">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Savings rate</p>
            <p className={`text-lg font-semibold ${healthColor(srHealth)}`}>{fmtPct(savingsRate)}</p>
            <p className="text-[11px] text-muted-foreground">Monthly surplus ÷ income (not liquid balance)</p>
          </div>
          <div className="space-y-1">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Expense burden</p>
            <p className={`text-lg font-semibold ${healthColor(ebHealth)}`}>
              {expenseBurden != null ? fmtPct(expenseBurden) : "—"}
            </p>
            <p className="text-[11px] text-muted-foreground">Expenses ÷ income</p>
          </div>
        </div>
      </motion.div>

      {/* Income vs expenses — single period */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }} className="glass-card p-5">
        <h2 className="font-heading text-sm font-semibold mb-1">Income vs. expenses (monthly)</h2>
        <p className="text-xs text-muted-foreground mb-4">From your profile</p>
        <div className="h-52">
          <ResponsiveContainer>
            <BarChart data={barCompare} barGap={8}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(222,30%,18%)" />
              <XAxis dataKey="name" stroke="hsl(215,20%,55%)" fontSize={12} />
              <YAxis stroke="hsl(215,20%,55%)" fontSize={12} tickFormatter={(v) => `$${v / 1000}k`} />
              <Tooltip contentStyle={CustomTooltipStyle} formatter={(value: number) => [`$${value.toLocaleString()}`, ""]} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]} barSize={40} name="Amount">
                {barCompare.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Loan summary */}
      {p.has_loan ? (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-5 border-border/60"
        >
          <h2 className="font-heading text-sm font-semibold mb-3">Loan summary</h2>
          <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-[10px] uppercase text-muted-foreground">Outstanding</p>
              <p className="font-semibold">{fmtMoney(p.loan_amount)}</p>
            </div>
            <div>
              <p className="text-[10px] uppercase text-muted-foreground">Monthly EMI</p>
              <p className="font-semibold">{fmtMoney(p.monthly_emi)}</p>
            </div>
            <div>
              <p className="text-[10px] uppercase text-muted-foreground">Interest rate</p>
              <p className="font-semibold">{p.loan_interest_rate != null ? `${p.loan_interest_rate}%` : "—"}</p>
            </div>
            <div>
              <p className="text-[10px] uppercase text-muted-foreground">Term (months)</p>
              <p className="font-semibold">{p.loan_term_months != null ? Math.round(p.loan_term_months) : "—"}</p>
            </div>
          </div>
        </motion.div>
      ) : null}
    </div>
  );
};
