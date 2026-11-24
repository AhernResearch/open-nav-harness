import type { Row } from "./api";
export interface WorkspaceProps {
  services: Row[];
  templates: Row[];
  perform: (fn: () => Promise<unknown>) => Promise<void>;
  refresh: () => Promise<void>;
  setNotice: (text: string) => void;
}
