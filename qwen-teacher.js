/* Ron teacher bridge — Qwen3
 * Qwen is a teacher/reasoning component only. Ron owns identity, memory,
 * conversation state, policy and final decisions.
 */
(() => {
  const CONFIG = Object.freeze({
    model: "qwen3-4b",
    role: "teacher",
    defaultEndpoint: "https://ron-qwen-teacher-production.up.railway.app/v1/chat/completions",
    endpointStorageKey: "ron-qwen-endpoint-v2",
    enabledStorageKey: "ron-qwen-enabled-v2",
    lessonsStorageKey: "ron-qwen-lessons-v2",
    maxLessons: 80,
    maxHistory: 14
  });
  const read=(key,fallback)=>{try{return JSON.parse(localStorage.getItem(key))??fallback}catch{return fallback}};
  const write=(key,value)=>{try{localStorage.setItem(key,JSON.stringify(value))}catch{}};
  const normalizeMessages=history=>Array.isArray(history)?history.filter(x=>x&&(x.role==="user"||x.role==="assistant")&&typeof x.content==="string").slice(-CONFIG.maxHistory):[];
  window.RonQwenTeacher=Object.freeze({
    config:CONFIG,
    isEnabled(){return read(CONFIG.enabledStorageKey,true)===true},
    setEnabled(value){write(CONFIG.enabledStorageKey,Boolean(value))},
    getEndpoint(){return String(read(CONFIG.endpointStorageKey,CONFIG.defaultEndpoint)).trim()},
    setEndpoint(url){write(CONFIG.endpointStorageKey,String(url||CONFIG.defaultEndpoint).trim())},
    getLessons(){const value=read(CONFIG.lessonsStorageKey,[]);return Array.isArray(value)?value.slice(-CONFIG.maxLessons):[]},
    saveLesson(text){const value=String(text||"").trim();if(!value)return false;const lessons=this.getLessons().filter(x=>x.text!==value);lessons.push({text:value,at:new Date().toISOString(),source:CONFIG.model});write(CONFIG.lessonsStorageKey,lessons.slice(-CONFIG.maxLessons));return true},
    buildMessages(userMessage,ronContext="",history=[]){
      const system=[
        "أنت Qwen، معلم ومكوّن تفكير مساعد لرون.",
        "رون هو النواة الأساسية والمستقلة وصاحب القرار النهائي، وليس تابعًا لـ Qwen أو OpenAI أو Google أو أي جهة أخرى.",
        "افهم الرسالة الحالية بالاعتماد على سياق المحادثة والذاكرة المقدمة.",
        "حافظ على اتساق الأسماء والحقائق والموضوع الجاري، وافهم الأسئلة المختصرة مثل: طيب؟ وماذا عنه؟ بالرجوع إلى السياق.",
        "أجب المستخدم مباشرة وبالعربية الواضحة ما لم يطلب لغة أخرى.",
        "لا تدّع أنك عدّلت ذاكرة رون. إذا كان المستخدم يعلّم رون، أعطِ صياغة معرفة مفيدة يمكن لرون حفظها.",
        ronContext?"ذاكرة رون ذات الصلة:\n"+ronContext:""
      ].filter(Boolean).join("\n\n");
      const messages=[{role:"system",content:system}];
      messages.push(...normalizeMessages(history));
      messages.push({role:"user",content:String(userMessage||"")});
      return messages;
    },
    async ask(userMessage,ronContext="",history=[]){
      if(!this.isEnabled())return{ok:false,reason:"disabled"};
      const endpoint=this.getEndpoint();
      if(!endpoint)return{ok:false,reason:"no-endpoint"};
      try{
        const response=await fetch(endpoint,{method:"POST",headers:{"Content-Type":"application/json","Accept":"application/json"},body:JSON.stringify({model:CONFIG.model,messages:this.buildMessages(userMessage,ronContext,history),temperature:.35,stream:false})});
        let data=null;try{data=await response.json()}catch{data=null}
        if(!response.ok){const message=data?.error?.message||data?.message||("HTTP "+response.status);return{ok:false,reason:"upstream: "+message}}
        const raw=data?.choices?.[0]?.message?.content??data?.output_text??data?.response??"";
        const text=Array.isArray(raw)?raw.map(x=>typeof x==="string"?x:(x?.text||x?.content||"")).join("").trim():String(raw||"").trim();
        if(!text){const message=data?.error?.message||data?.message||"";return{ok:false,reason:message?"upstream: "+message:"empty"}}
        return{ok:true,text,model:String(data?.model||CONFIG.model)}
      }catch(error){return{ok:false,reason:"network: "+(error?.message||"request failed")}}
    }
  });
})();