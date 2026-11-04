import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// relative base, so the built site also works from a sub-path (GitHub Pages)
export default defineConfig({ plugins: [react()], base: "./" });
