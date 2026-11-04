import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import importer from "./importer-plugin.js";

// relative base, so the built site also works from a sub-path (GitHub Pages);
// the importer adds upload endpoints to the dev server only
export default defineConfig({ plugins: [react(), importer()], base: "./" });
