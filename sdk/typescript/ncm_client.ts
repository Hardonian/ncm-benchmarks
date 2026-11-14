/** Thin TypeScript SDK client for Next-Concept Modeling sovereign microservice. */
export interface TriageRequest {
  text: string;
  labels?: string[];
  max_level?: number;
  best_of_k?: number;
  redact_pii?: boolean;
  restore_pii?: boolean;
  audit_receipt?: boolean;
}

export interface TriageResponse {
  category: string;
  confidence: number;
  escalate_to_frontier: boolean;
  summary: string;
  schema_valid: boolean;
  latency_ms: number;
}

export class NCMClientTS {
  private baseUrl: string;
  private apiKey?: string;

  constructor(baseUrl = "http://127.0.0.1:8031", apiKey?: string) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.apiKey = apiKey;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) h["Authorization"] = `Bearer ${this.apiKey}`;
    return h;
  }

  async summarize(text: string): Promise<{ summary: string; latency_ms: number }> {
    const res = await fetch(`${this.baseUrl}/summarize`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async triage(req: TriageRequest): Promise<TriageResponse> {
    const res = await fetch(`${this.baseUrl}/triage`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async triageFeedback(text: string, label: string, maxLevel = 2): Promise<Record<string, unknown>> {
    const res = await fetch(`${this.baseUrl}/triage/feedback`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ text, label, max_level: maxLevel }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }
}
