/* Ron teacher bridge — Qwen3-4B
 * Qwen is an optional teacher/model provider. Ron remains the owner of identity,
 * local memory, policy and final decisions. No API key is embedded here.
 */
(() => {
  const CONFIG = Object.freeze({
    model: "Qwen/Qwen3-4B",
    role: "teacher",
    endpointStorageKey: "ron-qwen-endpoint-v1",
    enabledStorageKey: "ron-qwen-enabled-v1",
    lessonsStorageKey: "ron-qwen-lessons-v1",
    maxLessons: 50
  });
  const read = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } };
  const write = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} };
  window.RonQwenTeacher = Object.freeze({
    config: CONFIG,
    isEnabled() { return read(CONFIG.enabledStorageKey, false) === true; },
    setEnabled(value) { write(CONFIG.enabledStorageKey, Boolean(value)); },
    getEndpoint() { return String(read(CONFIG.endpointStorageKey, "")).trim(); },
    setEndpoint(url) { write(CONFIG.endpointStorageKey, String(url || "").trim()); },
    getLessons() {
      const v = read(CONFIG.lessonsStorageKey, []);
      return Array.isArray(v) ? v.slice(-CONFIG.maxLessons) : [];
    },
    saveLesson(text) {
      const value = String(text || "").trim();
      if (!value) return false;
      const lessons = this.getLessons().filter(x => x.text !== value);
      lessons.push({ text: value, at: new Date().toISOString(), source: CONFIG.model });
      write(CONFIG.lessonsStorageKey, lessons.slice(-CONFIG.maxLessons));
      return true;
    },
    buildTeacherPrompt(userMessage, ronContext = "") {
      return [
        "أنت معلم مساعد لرون.",
        "رون هو النواة الأساسية والمستقلة وصاحب القرار النهائي.",
        "قدّم معرفة واقتراحات قابلة للتحقق، ولا تغيّر هوية رون أو ذاكرته مباشرة.",
        ronContext ? "سياق رون:\n" + ronContext : "",
        "رسالة المستخدم:\n" + String(userMessage || "")
      ].filter(Boolean).join("\n\n");
    },
    async ask(userMessage, ronContext = "") {
      if (!this.isEnabled()) return { ok: false, reason: "disabled" };
      const endpoint = this.getEndpoint();
      if (!endpoint) return { ok: false, reason: "no-endpoint" };
      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: CONFIG.model,
            messages: [{ role: "user", content: this.buildTeacherPrompt(userMessage, ronContext) }],
            stream: false
          })
        });
        if (!response.ok) return { ok: false, reason: "http-" + response.status };
        const data = await response.json();
        const text = data?.choices?.[0]?.message?.content || data?.output_text || data?.response || "";
        return text ? { ok: true, text: String(text), model: CONFIG.model } : { ok: false, reason: "empty" };
      } catch (error) {
        return { ok: false, reason: "network" };
      }
    }
  });
})();