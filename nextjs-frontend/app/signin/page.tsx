"use client";

import { signIn } from "next-auth/react";

export default function SignInPage() {
  return (
    <main style={{ fontFamily: "Arial", margin: "2rem" }}>
      <h2>Sign in</h2>
      <p>Use your company Google account to access attrition dashboards.</p>
      <button onClick={() => signIn("google")} style={{ padding: "0.6rem 1rem" }}>
        Sign in with Google
      </button>
    </main>
  );
}
