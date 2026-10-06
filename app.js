(() => {
  const chat = document.getElementById("chat");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const key = "ron-local-memory-v1";
  let memory = JSON.parse(localStorage.getItem(key) || "[]");

  const add = (role, text) => {
    const box = document.createElement("div");
    box.className = "message " + role;
    box.innerHTML = "<b>" + (role === "ron" ? "رون" : "أنت") + "</b><p></p>";
    box.querySelector("p").textContent = text;
    chat.appendChild(box);
    chat.scrollTop = chat.scrollHeight;
  };

  const save = () => localStorage.setItem(key, JSON.stringify(memory.slice(-100)));

  const answer = (text) => {
    const lower = text.toLowerCase();
    if (lower.startsWith("علّم رون:") || lower.startsWith("علم رون:")) {
      const lesson = text.replace(/^علّم رون:\s*/i, "").replace(/^علم رون:\s*/i, "").trim();
      if (lesson) {
        memory.push({ type: "lesson", text: lesson, at: Date.now() });
        save();
        return "تم حفظ تعليمك محليًا على هذا الجهاز.";
      }
    }
    if (lower.includes("ماذا تعلمت") || lower.includes("ماذا تعرف")) {
      return memory.length ? "أتذكر حاليًا " + memory.length + " معلومة محلية.\n\n" + memory.slice(-5).map((m, i) => (i + 1) + ". " + m.text).join("\n") : "لم تعلّمني شيئًا محفوظًا بعد.";
    }
    if (lower.includes("امسح الذاكرة")) {
      memory = []; save(); return "تم مسح الذاكرة المحلية.";
    }
    if (/^(مرحبا|أهلا|اهلا|السلام عليكم)/.test(lower)) return "أهلًا بك. أنا رون، وما زلنا نبني قدراتي الأساسية خطوة بخطوة.";
    return "وصلت رسالتك. النواة المحلية تعمل، لكن نموذج الاستدلال الحقيقي لم يتم توصيله بعد. يمكنك تعليمي بكتابة: علّم رون: ثم المعلومة.";
  };

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    add("user", text);
    input.value = "";
    input.style.height = "auto";
    setTimeout(() => add("ron", answer(text)), 120);
  });

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 140) + "px";
  });
})();