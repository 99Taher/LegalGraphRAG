import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Legal GraphRAG — EU Treaty Intelligence",
  description:
    "AI-powered legal research on EU Treaties using Graph RAG with Neo4j and semantic search.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="fr" className="dark">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
