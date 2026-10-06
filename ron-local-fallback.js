/* Ron local fallback — keeps basic conversation alive when Qwen is unavailable. */
(() => {
  const waitForTeacher = () => {
    const teacher = globalThis.RonQwenTeacher;
    if (!teacher || teacher.__ronFallbackWrapped) return Boolean(teacher);

    const originalAsk = teacher.ask.bind(teacher);
    const fallback = (message) => {
      const n = String(message || "")
        .toLowerCase()
        .replace(/[ًٌٍَُِّْـ]/g, "")
        .replace(/[أإآ]/g, "ا")
        .replace(/ى/g, "ي")
        .replace(/\s+/g, " ")
        .trim();

      if (/^(مرحبا|اهلا|اهلا وسهلا|السلام عليكم|السلام عليكم ورحمة الله وبركاته)[؟?!.،]*$/.test(n))
        return "مرحبًا! أنا رون. كيف يمكنني مساعدتك؟";
      if (/^(كيف حالك|عامل ايه|اخبارك ايه|كيفك)[؟?!.،]*$/.test(n))
        return "أنا بخير وجاهز للعمل معك. أنا رون.";
      if (/^(ما اسمك|ما هو اسمك|ايه اسمك|اسمك ايه)[؟?!.،]*$/.test(n))
        return "اسمي رون.";
      if (/^(وانت|وانت؟)[؟?!.،]*$/.test(n))
        return "أنا رون.";
      return null;
    };

    const proxy = new Proxy(teacher, {
      get(target, prop, receiver) {
        if (prop === "__ronFallbackWrapped") return true;
        if (prop === "ask") {
          return async (message, context, history) => {
            const local = fallback(message);
            if (local) return {ok:true, text:local, model:"ron-local"};
            return originalAsk(message, context, history);
          };
        }
        return Reflect.get(target, prop, receiver);
      }
    });

    globalThis.RonQwenTeacher = proxy;
    return true;
  };

  if (!waitForTeacher()) {
    let attempts = 0;
    const timer = setInterval(() => {
      attempts++;
      if (waitForTeacher() || attempts > 100) clearInterval(timer);
    }, 10);
  }
})();