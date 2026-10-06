(() => {
  const STORAGE = { chat: "ron-chat-v2", lessons: "ron-lessons-v2", settings: "ron-settings-v1" };
  const $ = (id) => document.getElementById(id);
  const chat = $("chat"), form = $("composer"), input = $("input"), send = $("send");
  const menu = $("menu"), settings = $("settings"), closeSettings = $("close-settings");
  const exportBtn = $("export"), importInput = $("import"), clearBtn = $("clear-data"), newChat = $("new-chat");

  const read = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } };
  let messages = read(STORAGE.chat, [{role:"ron", text:"مرحبًا. أنا رون. النواة المحلية تعمل، وذاكرتي محفوظة على هذا الجهاز."}]);
  let lessons = read(STORAGE.lessons, []);

  const save = () => {
    localStorage.setItem(STORAGE.chat, JSON.stringify(messages.slice(-200)));
    localStorage.setItem(STORAGE.lessons, JSON.stringify(lessons.slice(-500)));
  };

  const bubble = (role, text) => {
    const el = document.createElement("article");
    el.className = "message " + role;
    const name = document.createElement("b");
    name.textContent = role === "ron" ? "رون" : "أنت";
    const p = document.createElement("p");
    p.textContent = text;
    el.append(name, p);
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
  };

  const render = () => {
    chat.replaceChildren();
    messages.forEach(m => bubble(m.role, m.text));
  };

  const normalize = (s) => s.toLowerCase().replace(/[ًٌٍَُِّْـ]/g, "").trim();
  const teach = (text) => {
    const lesson = text.replace(/^\s*عل[ّ]?م\s+رون\s*:\s*/i, "").trim();
    if (!lesson) return "اكتب المعلومة بعد «علّم رون:».";
    lessons.push({text: lesson, at: new Date().toISOString()}); save();
    return "تم حفظ التعليم في ذاكرة رون المحلية.";
  };

  const answer = (text) => {
    const n = normalize(text);
    if (/^عل[ّ]?م رون\s*:/.test(n)) return teach(text);
    if (n.includes("ماذا تعلمت") || n.includes("ما الذي تعلمته")) {
      return lessons.length ? "هذه آخر تعليماتي المحفوظة:\n\n" + lessons.slice(-10).map((x,i) => (i+1)+". "+x.text).join("\n") : "لم تعلّمني شيئًا بعد.";
    }
    if (n.includes("امسح الذاكره") || n.includes("امسح الذاكرة")) {
      lessons = []; save(); return "تم مسح الذاكرة التي علّمتني إياها.";
    }
    if (/^(مرحبا|اهلا|أهلا|السلام عليكم)/.test(n)) return "أهلًا بك. أنا رون. ابدأ بتعليمي أو اكتب أي شيء لتجربة النواة.";
    if (n.includes("من انت") || n.includes("من أنت")) return "أنا رون، مشروع مساعد مستقل. هذه النسخة تعمل محليًا بدون مفتاح API، بينما طبقة النموذج المتقدم ما زالت قيد البناء.";
    const hit = lessons.slice().reverse().find(x => n.includes(normalize(x.text).slice(0, Math.min(30, normalize(x.text).length))));
    if (hit) return "أتذكر تعليمك: " + hit.text;
    return "وصلتني رسالتك. النواة المحلية تعمل، لكن نموذج الاستدلال المتقدم لم يُوصل بعد. يمكنك تعليمي بكتابة «علّم رون: ...».";
  };

  const sendMessage = () => {
    const text = input.value.trim(); if (!text) return;
    messages.push({role:"user", text}); bubble("user", text);
    input.value = ""; input.style.height = "auto"; send.disabled = true;
    setTimeout(() => { const text = answer(messages[messages.length-1].text); messages.push({role:"ron", text}); bubble("ron", text); save(); send.disabled = false; input.focus(); }, 100);
  };

  form.addEventListener("submit", e => { e.preventDefault(); sendMessage(); });
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 140)+"px"; });
  input.addEventListener("keydown", e => { if(e.key==="Enter" && !e.shiftKey){e.preventDefault(); sendMessage();} });

  menu.addEventListener("click", () => settings.classList.add("open"));
  closeSettings.addEventListener("click", () => settings.classList.remove("open"));
  newChat.addEventListener("click", () => { messages = [{role:"ron", text:"بدأنا محادثة جديدة. كيف يمكنني مساعدتك؟"}]; save(); render(); settings.classList.remove("open"); });
  clearBtn.addEventListener("click", () => { if(confirm("مسح كل بيانات رون المحلية؟")) { localStorage.clear(); location.reload(); } });
  exportBtn.addEventListener("click", () => {
    const data = JSON.stringify({version:1, exportedAt:new Date().toISOString(), messages, lessons}, null, 2);
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([data], {type:"application/json"})); a.download="ron-backup.json"; a.click(); URL.revokeObjectURL(a.href);
  });
  importInput.addEventListener("change", async () => {
    const file = importInput.files[0]; if(!file) return;
    try { const data = JSON.parse(await file.text()); if(!Array.isArray(data.messages)||!Array.isArray(data.lessons)) throw new Error(); messages=data.messages; lessons=data.lessons; save(); render(); alert("تم استيراد ذاكرة رون."); } catch { alert("ملف النسخ الاحتياطي غير صالح."); } importInput.value="";
  });
  render();
})();