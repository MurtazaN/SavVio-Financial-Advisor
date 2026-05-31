/**
 * Chat message input bar for the AiChat component.
 */

import { Send } from "lucide-react";
import { Button } from "./ui/button";

interface MessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled: boolean;
  canSend: boolean;
}

export function MessageInput({
  value,
  onChange,
  onSend,
  disabled,
  canSend,
}: MessageInputProps) {
  return (
    <div className="glass-card p-2 flex items-center gap-2">
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) =>
          e.key === "Enter" && !e.shiftKey && canSend && onSend()
        }
        placeholder="Paste a product link or ask 'Should I buy...?'"
        className="flex-1 bg-transparent px-3 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none"
        disabled={disabled}
      />
      <Button
        size="icon"
        onClick={onSend}
        disabled={disabled || !canSend}
        className="shrink-0 rounded-lg"
      >
        <Send className="w-4 h-4" />
      </Button>
    </div>
  );
}
