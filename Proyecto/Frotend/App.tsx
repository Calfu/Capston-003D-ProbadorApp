import React from "react";
import { ThemeProvider } from "./src/theme/ThemeContext";
import RootNavigation from "./src/navigation";

export default function App() {
  return (
    <ThemeProvider>
      <RootNavigation />
    </ThemeProvider>
  );
}