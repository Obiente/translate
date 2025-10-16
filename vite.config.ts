import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import fs from "fs";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    // Clean-URL static pages for SPA: serve /about -> public/about.html, etc.
    ((): any => {
      const pages: Record<string, string> = {
        "/about": "about.html",
        "/about/": "about.html",
        "/privacy": "privacy.html",
        "/privacy/": "privacy.html",
        "/terms": "terms.html",
        "/terms/": "terms.html",
      };
      const sendFile = (res: any, filePath: string) => {
        try {
          const html = fs.readFileSync(filePath, "utf-8");
          res.setHeader("Content-Type", "text/html");
          res.statusCode = 200;
          res.end(html);
        } catch (e) {
          res.statusCode = 404;
          res.end("Not Found");
        }
      };
      const middleware = (req: any, res: any, next: any) => {
        if (req.method !== "GET") return next();
        // Allow direct serving of robots/sitemap
        if (req.url === "/robots.txt" || req.url === "/sitemap.xml") return next();
        const accept = String(req.headers?.accept || "");
        if (!accept.includes("text/html")) return next();
        const url = new URL(req.url, "http://localhost");
        const pathname = url.pathname;
        const match = pages[pathname];
        if (!match) return next();
        const filePath = path.resolve(process.cwd(), "public", match);
        return sendFile(res, filePath);
      };
      const plugin: any = {
        name: "spa-static-pages",
        enforce: "pre",
        configureServer(server: any) {
          server.middlewares.use(middleware);
        },
      };
      // Support vite preview if available
      (plugin as any).configurePreviewServer = (server: any) => {
        server.middlewares.use(middleware);
      };
      return plugin;
    })(),
  ],
  server: {
    port: 3000,
    host: true,
    allowedHosts: true,
  },
  define: {
    // Make env variables available in the app
    "process.env": process.env,
  },
});
