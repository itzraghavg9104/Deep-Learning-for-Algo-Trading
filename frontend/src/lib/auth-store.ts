"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import axios from "axios";
import api from "./api";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
} from "firebase/auth";
import { firebaseAuth, googleProvider } from "./firebase";

interface User {
  id: number | string;
  email: string;
  is_active: boolean;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  loginWithGoogle: () => Promise<void>;
  logout: () => void;
  fetchUser: () => Promise<void>;
  clearError: () => void;
}

const setAuthCookie = (token: string) => {
  if (typeof document === "undefined") return;
  document.cookie = `auth_token=${token}; Path=/; Max-Age=${60 * 60 * 24}; SameSite=Lax`;
};

const clearAuthCookie = () => {
  if (typeof document === "undefined") return;
  document.cookie = "auth_token=; Path=/; Max-Age=0; SameSite=Lax";
};

const getErrorMessage = (error: unknown, fallback: string) => {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") {
      return detail;
    }
  }
  return fallback;
};

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          const credential = await signInWithEmailAndPassword(firebaseAuth, email, password);
          const idToken = await credential.user.getIdToken();
          set({ token: idToken, isAuthenticated: true });
          setAuthCookie(idToken);

          // Update API client default headers
          api.defaults.headers.common["Authorization"] = `Bearer ${idToken}`;

          // Fetch user details
          await get().fetchUser();
        } catch (error: unknown) {
          const message = getErrorMessage(error, "Login failed. Please try again.");
          set({ error: message, isAuthenticated: false });
          throw new Error(message);
        } finally {
          set({ isLoading: false });
        }
      },

      register: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          const credential = await createUserWithEmailAndPassword(firebaseAuth, email, password);
          const idToken = await credential.user.getIdToken();
          set({ token: idToken, isAuthenticated: true });
          setAuthCookie(idToken);
          api.defaults.headers.common["Authorization"] = `Bearer ${idToken}`;
          await get().fetchUser();
        } catch (error: unknown) {
          const message = getErrorMessage(error, "Registration failed. Please try again.");
          set({ error: message });
          throw new Error(message);
        } finally {
          set({ isLoading: false });
        }
      },

      loginWithGoogle: async () => {
        set({ isLoading: true, error: null });
        try {
          const credential = await signInWithPopup(firebaseAuth, googleProvider);
          const idToken = await credential.user.getIdToken();
          set({ token: idToken, isAuthenticated: true });
          setAuthCookie(idToken);
          api.defaults.headers.common["Authorization"] = `Bearer ${idToken}`;
          await get().fetchUser();
        } catch (error: unknown) {
          const message = getErrorMessage(error, "Google sign-in failed. Please try again.");
          set({ error: message, isAuthenticated: false });
          throw new Error(message);
        } finally {
          set({ isLoading: false });
        }
      },

      logout: () => {
        // Clear auth header
        delete api.defaults.headers.common["Authorization"];
        signOut(firebaseAuth).catch(() => undefined);
        clearAuthCookie();
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          error: null,
        });
      },

      fetchUser: async () => {
        try {
          const response = await api.get("/auth/me");
          set({ user: response.data });
        } catch (error: unknown) {
          // If fetching user fails, logout
          get().logout();
          throw error;
        }
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: "auth-storage",
      partialize: (state) => ({ token: state.token, isAuthenticated: state.isAuthenticated }),
    }
  )
);

// Initialize auth header from stored token
export const initializeAuth = () => {
  const state = useAuthStore.getState();
  if (state.token) {
    setAuthCookie(state.token);
    api.defaults.headers.common["Authorization"] = `Bearer ${state.token}`;
    // Try to fetch user on app load
    state.fetchUser().catch(() => {
      // If fetching fails, token is invalid
      state.logout();
    });
  }
};
