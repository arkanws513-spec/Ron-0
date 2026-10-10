(function (root) {
  "use strict";

  async function chat(endpoint, messages, fetchImpl) {
    const base = String(endpoint || "").trim().replace(/\/+$/, "");
    const fetcher = fetchImpl || root.fetch;
    if (!base || typeof fetcher !== "function") return null;

    const safeMessages = (Array.isArray(messages) ? messages : [])
      .filter((item) => item && ["system", "user", "assistant"].includes(item.role))
      .slice(-12)
      .map((item) => ({
        role: item.role,
        content: String(item.content || "").trim().slice(-4000)
      }))
      .filter((item) => item.content.length > 0);
    if (!safeMessages.length) return null;

    try {
      const response = await fetcher(base + "/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: safeMessages })
      });
      if (!response || !response.ok) return null;
      const payload = await response.json();
      if (!payload || typeof payload.content !== "string" || !payload.content.trim()) return null;
      return {
        content: payload.content.trim(),
        model: String(payload.model || "ron-native-checkpoint"),
        metadata: payload.metadata && typeof payload.metadata === "object" ? payload.metadata : {}
      };
    } catch (error) {
      return null;
    }
  }

  root.RonInferenceClient = Object.freeze({ chat });
})(globalThis);
