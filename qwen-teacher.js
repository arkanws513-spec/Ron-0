/* Ron teacher bridge — Qwen3-4B
 * Qwen is a teacher component only. Ron owns identity, local memory,
 * policy and final decisions. No API key is embedded in the browser.
 */
(() => {
  const CONFIG = Object.freeze({
    model: "qwen3-4b",
    role: "teacher",
    defaultEndpoint: "https://ron-qwen-teacher-production.up.railway.app/v1/chat/completions",
    endpointStorageKey: "ron-qwen-endpoint-v1",
    enabledStorageKey: "ron-qwen-enabled-v1",
    lessonsStorageKey: "ron-qwen-lessons-v1",
    maxLessons: 50
  });
  const read = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } };
  const write = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} };
  window.RonQwenTeacher = Object.freeze({
    config: CONFIG,
    isEnabled() { return read(CONFIG.enabledStorageKey, true) === true; },
    setEnabled(value) { write(CONFIG.enabledStorageKey, Boolean(value)); },
    getEndpoint() { return String(read(CONFIG.endpointStorageKey, CONFIG.defaultEndpoint)).trim(); },
    setEndpoint(url) { write(CONFIG.endpointStorageKey, String(url || CONFIG.defaultEndpoint).trim()); },
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
        "قدّم معرفة واقتراحات قابلة للتحقق، ولا تغيّر هوية رون أو ذاكرته مباشرة. إذا كانت الرسالة تعلّم رون معلومة، فاقترح طريقة لفهمها أو حفظها دون الادعاء أنك غيّرت ذاكرته.",
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
        const raw = data?.choices?.[0]?.message?.content ?? data?.output_text ?? data?.response ?? "";
        const text = Array.isArray(raw) ? raw.map(x => typeof x === "string" ? x : (x?.text || x?.content || "")).join("").trim() : String(raw || "").trim();
        const apiError = data?.error?.message || data?.message || "";
        return text ? { ok: true, text, model: CONFIG.model } : { ok: false, reason: apiError ? "upstream: " + apiError : "empty" };
      } catch {
        return { ok: false, reason: "network" };
      }
    }
  });
})();