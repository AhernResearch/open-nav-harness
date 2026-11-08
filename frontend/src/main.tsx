import { createRoot } from "react-dom/client";
import { Workbench } from "./Workbench";
import { HarnessPage } from "./harness/HarnessPage";
import "./style.css";

// Two independent views; `/` and legacy `/?attempt=` links land on the Harness.
if (location.pathname === "/") history.replaceState(null, "", "/harness" + location.search);
const view = location.pathname.startsWith("/workbench") ? <Workbench /> : <HarnessPage />;
createRoot(document.getElementById("root")!).render(view);
