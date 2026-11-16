import { createRoot } from "react-dom/client";
import { Workbench } from "./Workbench";
import { HarnessPage } from "./harness/HarnessPage";
import { installFavicon } from "./Logo";
import "./style.css";

if (location.pathname === "/chat") history.replaceState(null, "", "/harness" + location.search);

// Two independent views; `/` and legacy `/?attempt=` links land on the Harness.
if (location.pathname === "/") history.replaceState(null, "", "/harness" + location.search);
const view = location.pathname.startsWith("/workbench") ? <Workbench /> : <HarnessPage />;
installFavicon();
createRoot(document.getElementById("root")!).render(view);
