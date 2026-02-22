"use client";

import { useEffect } from "react";
import { initializeAuth } from "@/lib/auth-store";

export function AuthInitializer() {
  useEffect(() => {
    initializeAuth();
  }, []);

  return null;
}
