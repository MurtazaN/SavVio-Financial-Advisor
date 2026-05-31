import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { fetchUserProfile, type UserProfileResponse } from "../services/api";

const STORAGE_KEY = "savvio_user_id";

interface UserContextValue {
  userId: string;
  setUserId: (id: string) => void;
  userProfile: UserProfileResponse | null;
  profileError: string | null;
  isLoadingProfile: boolean;
  refetchProfile: () => Promise<void>;
}

const UserContext = createContext<UserContextValue | null>(null);

export function UserProvider({ children }: { children: ReactNode }) {
  const [userId, setUserIdState] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) ?? "";
    } catch {
      return "";
    }
  });
  const [userProfile, setUserProfile] = useState<UserProfileResponse | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [isLoadingProfile, setIsLoadingProfile] = useState(false);

  const setUserId = useCallback((id: string) => {
    const trimmed = id.trim();
    setUserIdState(trimmed);
    try {
      if (trimmed) localStorage.setItem(STORAGE_KEY, trimmed);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
    if (!trimmed) {
      setUserProfile(null);
      setProfileError(null);
    }
  }, []);

  const refetchProfile = useCallback(async () => {
    if (!userId.trim()) {
      setUserProfile(null);
      setProfileError(null);
      return;
    }
    setIsLoadingProfile(true);
    setProfileError(null);
    try {
      const p = await fetchUserProfile(userId.trim());
      setUserProfile(p);
    } catch (e) {
      setUserProfile(null);
      setProfileError(
        e instanceof Error ? e.message : typeof e === "string" ? e : "Failed to load profile",
      );
    } finally {
      setIsLoadingProfile(false);
    }
  }, [userId]);

  useEffect(() => {
    void refetchProfile();
  }, [refetchProfile]);

  const value = useMemo(
    () => ({
      userId,
      setUserId,
      userProfile,
      profileError,
      isLoadingProfile,
      refetchProfile,
    }),
    [userId, setUserId, userProfile, profileError, isLoadingProfile, refetchProfile],
  );

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
}

export function useUser(): UserContextValue {
  const ctx = useContext(UserContext);
  if (!ctx) throw new Error("useUser must be used within UserProvider");
  return ctx;
}
