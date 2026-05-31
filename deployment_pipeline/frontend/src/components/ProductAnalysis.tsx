import { motion } from "framer-motion";
import { BarChart3, Construction } from "lucide-react";

/**
 * Placeholder — full product utility analysis can be wired to backend later.
 */
export const ProductAnalysis = () => (
  <div className="max-w-xl mx-auto space-y-6 text-center py-16">
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass-card p-10 border-dashed border-border/60"
    >
      <div className="flex justify-center mb-4">
        <div className="w-14 h-14 rounded-2xl bg-primary/15 flex items-center justify-center">
          <Construction className="w-7 h-7 text-primary" />
        </div>
      </div>
      <div className="flex items-center justify-center gap-2 mb-2">
        <BarChart3 className="w-5 h-5 text-muted-foreground" />
        <h1 className="font-heading text-xl font-semibold">Product analysis</h1>
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed">
        This section is <strong className="text-foreground">coming soon</strong>. Use <strong className="text-foreground">AI Advisor</strong> for purchase
        recommendations with your financial profile, or check the <strong className="text-foreground">Dashboard</strong> for your snapshot.
      </p>
    </motion.div>
  </div>
);
