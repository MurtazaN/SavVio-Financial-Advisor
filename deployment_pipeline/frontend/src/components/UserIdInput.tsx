import { useEffect, useState } from "react";
import { User } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useUser } from "../context/UserContext";

export function UserIdInput() {
  const { userId, setUserId, profileError, isLoadingProfile } = useUser();
  const [draft, setDraft] = useState(userId);

  useEffect(() => {
    setDraft(userId);
  }, [userId]);

  const apply = () => {
    setUserId(draft.trim());
  };

  const hasError = Boolean(profileError && userId);

  return (
    <div className="flex flex-col items-end gap-1 max-w-[min(100%,28rem)]">
      <div className="flex items-center gap-2 w-full min-w-0">
        <User className="w-4 h-4 text-muted-foreground shrink-0 hidden sm:block" />
        <Input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && apply()}
          placeholder="User ID (e.g. U00047)"
          className={`h-9 text-sm bg-secondary/40 min-w-0 ${
            hasError ? "border-destructive focus-visible:ring-destructive/40" : "border-border/60"
          }`}
          aria-label="User ID"
          aria-invalid={hasError}
          aria-describedby={hasError ? "user-id-profile-error" : undefined}
        />
        <Button type="button" size="sm" variant="secondary" className="shrink-0" onClick={apply}>
          Set
        </Button>
        {isLoadingProfile && userId && (
          <span className="text-[10px] text-muted-foreground shrink-0">Loading…</span>
        )}
        {userId && !profileError && !isLoadingProfile && (
          <span className="text-[10px] text-primary font-medium shrink-0 hidden sm:inline" title={userId}>
            {userId}
          </span>
        )}
      </div>
      {hasError && (
        <p
          id="user-id-profile-error"
          className="text-xs text-destructive text-right leading-snug max-w-full pl-6 sm:pl-0"
          role="alert"
        >
          {profileError}
        </p>
      )}
    </div>
  );
}
