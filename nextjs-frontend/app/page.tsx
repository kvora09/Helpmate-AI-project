import Link from "next/link";

export default function Home() {
  return (
    <main style={{ fontFamily: "Arial", margin: "2rem" }}>
      <h1>Attrition Early Warning (Starter)</h1>
      <p>
        This React/Next starter is designed to consume scored employee risk data from the Python
        model service.
      </p>
      <ul>
        <li>Google authentication via NextAuth starter.</li>
        <li>Ready to plug API routes for GPTW survey and CEO chat signals.</li>
        <li>Can display retained-employee scoring population and intervention queue.</li>
      </ul>
      <Link href="/signin">Go to Sign In</Link>
    </main>
  );
}
