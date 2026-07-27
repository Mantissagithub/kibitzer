import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ThemeProvider } from "@/components/ThemeProvider";
import { RouterProvider } from "@/lib/router";
import App from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </RouterProvider>
  </StrictMode>,
);
