import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* Enable Tailwind v4 via turbopack */
  experimental: {
    turbopackRules: {},
  },

  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
      {
        source: "/health",
        destination: "http://localhost:8000/health",
      },
    ];
  },
};

export default nextConfig;
