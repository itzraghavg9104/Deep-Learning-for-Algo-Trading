import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

const PROTECTED_PREFIXES = ["/dashboard", "/profile", "/backtest", "/trades"];
const AUTH_PAGES = ["/auth/login", "/auth/register"];

const isProtectedRoute = (pathname: string) =>
  PROTECTED_PREFIXES.some((prefix) => pathname.startsWith(prefix));

const isAuthRoute = (pathname: string) =>
  AUTH_PAGES.some((path) => pathname.startsWith(path));

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const token = request.cookies.get("auth_token")?.value;

  if (isProtectedRoute(pathname) && !token) {
    const loginUrl = new URL("/auth/login", request.url);
    loginUrl.searchParams.set("next", pathname);
    return NextResponse.redirect(loginUrl);
  }

  if (isAuthRoute(pathname) && token) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard/:path*", "/profile/:path*", "/backtest/:path*", "/trades/:path*", "/auth/:path*"],
};
