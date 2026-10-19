import type { NextConfig } from "next";

const HOST = "const-ellation.com";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      {
        source: "/:path*",
        has: [{ type: "host", value: `www.${HOST}` }],
        destination: `https://${HOST}/:path*`,
        permanent: true,
      },
    ];
  },
  async rewrites() {
    return [{ source: "/", has: [{ type: "host", value: HOST }], destination: "/constellation" }];
  },
};

export default nextConfig;
