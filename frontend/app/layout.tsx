import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Veloitt RAG",
  description: "Streaming RAG chat interface powered by Veloitt knowledge base",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
