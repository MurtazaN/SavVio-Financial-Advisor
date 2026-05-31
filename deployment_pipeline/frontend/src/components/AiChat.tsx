/**
 * AiChat — Purchase Advisor conversational interface.
 *
 * Orchestrates the chat experience: message state, API calls, and layout.
 * Heavy sub-components are extracted into focused files:
 *   - TechnicalDetailsPanel  → debug/signal detail view
 *   - CatalogSearch          → product picker popover
 *   - MessageInput           → input bar
 *   - lib/formatters         → shared formatting (DRY)
 */

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Sparkles,
  Link2,
  AlertTriangle,
  TrendingDown,
  ShieldCheck,
} from "lucide-react";
import { useUser } from "../context/UserContext";
import {
  sendPredict,
  type PredictResponse,
  type ProductListItem,
} from "../services/api";

import { TechnicalDetailsPanel, truncateProductNameAtPipe } from "./TechnicalDetailsPanel";
import { CatalogSearch } from "./CatalogSearch";
import { MessageInput } from "./MessageInput";

// ---------------------------------------------------------------------------
// Types & constants
// ---------------------------------------------------------------------------

type Signal = "green" | "yellow" | "red";

interface AssistantLayer1 {
  leadParagraph: string;
  summaryLines: [string, string, string];
  closingLine: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  signal?: Signal;
  layer1?: AssistantLayer1;
  predictResponse?: PredictResponse;
}

const signalConfig: Record<
  Signal,
  { color: string; bg: string; border: string; label: string; icon: typeof ShieldCheck }
> = {
  green: { color: "text-success", bg: "bg-success/10", border: "border-success/25", label: "Safe to Buy", icon: ShieldCheck },
  yellow: { color: "text-caution", bg: "bg-caution/10", border: "border-caution/25", label: "Proceed with Caution", icon: AlertTriangle },
  red: { color: "text-destructive", bg: "bg-destructive/10", border: "border-destructive/25", label: "Not Recommended", icon: TrendingDown },
};

function mapRecommendationToSignal(rec: string): Signal | undefined {
  const u = rec.toUpperCase();
  if (u === "GREEN") return "green";
  if (u === "YELLOW") return "yellow";
  if (u === "RED") return "red";
  return undefined;
}

// ---------------------------------------------------------------------------
// Response formatting helpers
// ---------------------------------------------------------------------------

function InlineBold({ text }: { text: string }) {
  return (
    <>
      {text.split(/(\*\*.*?\*\*)/).map((part, i) =>
        part.startsWith("**") && part.endsWith("**") ? (
          <strong key={i} className="font-semibold text-foreground">
            {part.slice(2, -2)}
          </strong>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

function buildLeadParagraph(res: PredictResponse, signal: Signal): string {
  const name = truncateProductNameAtPipe(res.product_name, "this purchase");
  const price = res.product_price;
  const pricePart =
    price != null && !Number.isNaN(price) ? ` at **$${price.toFixed(2)}**` : "";
  const hypo =
    res.evaluation_mode === "hypothetical"
      ? " We're using **the price you stated** (no catalog match for this item)."
      : "";
  if (signal === "green") {
    return `Based on your financial profile, **${name}**${pricePart}.${hypo} This purchase looks **manageable** for your current budget.`.trim();
  }
  if (signal === "yellow") {
    return `Based on your financial profile, **${name}**${pricePart}.${hypo} This purchase is **borderline** — you may want to pause or adjust before buying.`.trim();
  }
  return `Based on your financial profile, **${name}**${pricePart}.${hypo} This purchase is **not recommended** given your current savings, debt, and cash flow.`.trim();
}

function formatEmergencySavingsLine(res: PredictResponse): string {
  const e = res.emergency_fund_months;
  if (e == null || Number.isNaN(e)) return "Emergency savings: — (target: 3–6 months)";
  return `Emergency savings: ${e.toFixed(1)} months (target: 3–6 months)`;
}

function formatDebtLoadLine(res: PredictResponse): string {
  const d = res.debt_to_income_ratio;
  if (d == null || Number.isNaN(d)) return "Monthly debt load: —";
  if (d <= 0.0001) return "No current debt";
  return `Monthly debt load: ${(d * 100).toFixed(1)}% of income`;
}

function formatPurchaseConfidenceLine(res: PredictResponse): string {
  const mlReady =
    res.confidence != null && !Number.isNaN(res.confidence) && res.ml_unavailable_reason == null;
  if (mlReady) return `Purchase confidence: ${(res.confidence * 100).toFixed(0)}%`;
  if (res.ml_unavailable_reason === "scoring_error") {
    return "Purchase confidence: Unavailable (model error — see logs)";
  }
  return "Purchase confidence: Unavailable (ML layer pending)";
}

const CLOSING_BY_SIGNAL: Record<Signal, string> = {
  green: "If this remains within your monthly plan, you can proceed — still track discretionary spend.",
  yellow: "Consider waiting a pay cycle, comparing alternatives, or trimming other discretionary spend first.",
  red: "Prioritize essentials, debt minimums, and emergency savings before this purchase.",
};

function buildLayer1(res: PredictResponse, signal: Signal): AssistantLayer1 {
  return {
    leadParagraph: buildLeadParagraph(res, signal),
    summaryLines: [
      formatEmergencySavingsLine(res),
      formatDebtLoadLine(res),
      formatPurchaseConfidenceLine(res),
    ],
    closingLine: CLOSING_BY_SIGNAL[signal],
  };
}

// ---------------------------------------------------------------------------
// AiChat component
// ---------------------------------------------------------------------------

const quickPrompts = [
  "Should I buy a $1,200 MacBook?",
  "Can I afford $350 headphones?",
  "Is a $2,500 Peloton worth it?",
];

const CHAT_STORAGE_KEY = "savvio_chat_history";

const WELCOME_MESSAGE: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "Welcome to **SavVio**. I'm your AI Financial Fiduciary — I'm here to help you make purchase decisions that align with your real financial health.\n\nEnter your **User ID** in the header. Turn on **Use catalog product** to pick a real SKU from our database (enables review + quality signals), or describe a purchase in your own words (may use **stated price only** if no catalog match).",
};

function loadChatHistory(uid: string): Message[] {
  try {
    const key = uid ? `${CHAT_STORAGE_KEY}_${uid}` : CHAT_STORAGE_KEY;
    const raw = localStorage.getItem(key);
    if (raw) {
      const parsed = JSON.parse(raw) as Message[];
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch { /* corrupt data — start fresh */ }
  return [WELCOME_MESSAGE];
}

function saveChatHistory(uid: string, messages: Message[]) {
  try {
    const key = uid ? `${CHAT_STORAGE_KEY}_${uid}` : CHAT_STORAGE_KEY;
    localStorage.setItem(key, JSON.stringify(messages));
  } catch { /* storage full or unavailable */ }
}

export const AiChat = () => {
  const { userId, profileError } = useUser();
  const [messages, setMessages] = useState<Message[]>(() => loadChatHistory(userId));
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [useCatalog, setUseCatalog] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState<ProductListItem | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevUserIdRef = useRef(userId);

  useEffect(() => {
    if (prevUserIdRef.current !== userId) {
      setMessages(loadChatHistory(userId));
      prevUserIdRef.current = userId;
    }
  }, [userId]);

  const persistMessages = useCallback(
    (msgs: Message[]) => saveChatHistory(userId, msgs),
    [userId],
  );

  useEffect(() => {
    persistMessages(messages);
  }, [messages, persistMessages]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isTyping]);

  const handleSend = async (text?: string) => {
    const msg = text || input;
    if (!msg.trim() && !useCatalog) return;
    if (useCatalog && !selectedProduct) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content:
            "Select a **product from the catalog** first, or turn off **Use catalog product** to ask in free text (no SKU).",
        },
      ]);
      return;
    }
    if (!msg.trim() && useCatalog && selectedProduct) {
      void handleSend(`Should I buy ${selectedProduct.product_name}?`);
      return;
    }
    if (!msg.trim()) return;
    if (!userId.trim()) {
      setMessages((prev) => [
        ...prev,
        { id: Date.now().toString(), role: "user", content: msg },
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content:
            "Please set your **User ID** in the header so I can load your financial profile and run the recommendation engine.",
        },
      ]);
      setInput("");
      return;
    }
    if (profileError) {
      setMessages((prev) => [
        ...prev,
        { id: Date.now().toString(), role: "user", content: msg },
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: `**Cannot run advice.** ${profileError}`,
        },
      ]);
      setInput("");
      return;
    }

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: msg };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsTyping(true);

    try {
      const res = await sendPredict({
        user_query: msg,
        user_id: userId.trim(),
        product_id: useCatalog && selectedProduct ? selectedProduct.product_id : undefined,
      });

      const signal = mapRecommendationToSignal(res.recommendation);

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: res.explanation,
        signal,
        layer1: signal ? buildLayer1(res, signal) : undefined,
        predictResponse: signal ? res : undefined,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e) {
      const errText = e instanceof Error ? e.message : "Request failed.";
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 2).toString(),
          role: "assistant",
          content: `**Something went wrong.** ${errText}`,
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const hasLink = (text: string) => /https?:\/\//.test(text);

  const canSend = !!(input.trim() || (useCatalog && selectedProduct));

  return (
    <div className="max-w-3xl mx-auto flex flex-col h-[calc(100vh-7rem)]">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-primary" />
        <div>
          <h1 className="font-heading text-xl font-semibold">Purchase Advisor</h1>
          <p className="text-xs text-muted-foreground">Your fiduciary — always on your side</p>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-4 pb-4 pr-1 scrollbar-thin">
        {messages.map((msg) => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[88%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-primary text-primary-foreground rounded-br-md"
                  : "glass-card rounded-bl-md"
              }`}
            >
              {/* Signal Badge */}
              {msg.signal &&
                (() => {
                  const cfg = signalConfig[msg.signal!];
                  const Icon = cfg.icon;
                  return (
                    <div
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold mb-3 ${cfg.bg} ${cfg.color} border ${cfg.border}`}
                    >
                      <Icon className="w-3.5 h-3.5" />
                      {cfg.label}
                    </div>
                  );
                })()}

              {/* LLM explanation */}
              {msg.content && (
                <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground/95">
                  <InlineBold text={msg.content} />
                </p>
              )}

              {/* Quick-stats strip */}
              {msg.layer1 && (
                <ul className="space-y-1 text-sm text-foreground/80 list-none pl-0 mt-2 border-t border-border/20 pt-2">
                  {msg.layer1.summaryLines.map((line, i) => (
                    <li key={i} className="leading-snug">{line}</li>
                  ))}
                </ul>
              )}

              {/* Link indicator */}
              {msg.role === "user" && hasLink(msg.content) && (
                <div className="flex items-center gap-1 mt-2 text-xs opacity-70">
                  <Link2 className="w-3 h-3" />
                  Product link detected
                </div>
              )}

              {/* Technical details (collapsed) */}
              {msg.predictResponse && <TechnicalDetailsPanel res={msg.predictResponse} />}
            </div>
          </motion.div>
        ))}

        {/* Typing indicator */}
        <AnimatePresence>
          {isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex justify-start"
            >
              <div className="glass-card rounded-2xl rounded-bl-md px-4 py-3 flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse-glow" />
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse-glow" style={{ animationDelay: "0.3s" }} />
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse-glow" style={{ animationDelay: "0.6s" }} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Quick Prompts */}
      {messages.length <= 1 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {quickPrompts.map((prompt) => (
            <button
              key={prompt}
              onClick={() => handleSend(prompt)}
              className="text-xs px-3 py-1.5 rounded-full border border-border/60 text-muted-foreground hover:text-foreground hover:border-primary/40 transition-colors"
            >
              {prompt}
            </button>
          ))}
        </div>
      )}

      {/* Catalog product picker */}
      <CatalogSearch
        useCatalog={useCatalog}
        onUseCatalogChange={setUseCatalog}
        selectedProduct={selectedProduct}
        onSelectProduct={setSelectedProduct}
        disabled={isTyping}
      />

      {/* Input bar */}
      <MessageInput
        value={input}
        onChange={setInput}
        onSend={() => void handleSend()}
        disabled={isTyping}
        canSend={canSend}
      />
    </div>
  );
};
