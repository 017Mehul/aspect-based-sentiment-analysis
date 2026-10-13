import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aspectly — Aspect-Based Sentiment Analysis",
  description:
    "Explore review sentiment by aspect with a transparent, Python-powered NLP pipeline.",
  applicationName: "Aspectly",
};

export const viewport: Viewport = {
  themeColor: "#f8faf4",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
