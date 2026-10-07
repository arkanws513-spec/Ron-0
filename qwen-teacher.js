/* Ron base-model bridge — Qwen3
 * Qwen is Ron's base language/reasoning component. Ron owns identity, memory,
 * conversation state, policy and final decisions.
 */
(() => {
  const CONFIG = Object.freeze({
    model: "Qwen/Qwen3-0.6B",
    role: "base-model",
    defaultEndpoint: "",
    endpointStorageKey: "ron-qwen-endpoint-v2",
    enabledStorageKey: "ron-qwen-enabled-v2",
    lessonsStorageKey: "ron-qwen-lessons-v2",
    maxLessons: 80,
    maxHistory: 18,
    trainingStorageKey: "ron-training-candidates-v1",
    maxTrainingExamples: 200,
    maxConcepts: 300
  });
  const read=(key,fallback)=>{try{return JSON.parse(localStorage.getItem(key))??fallback}catch{return fallback}};
  const DEFAULT_ENDPOINT=CONFIG.defaultEndpoint;
  const endpointIsValid=url=>{const value=String(url||"").trim();return value===""||/^https:\/\/.+\/v1\/chat\/completions\/?$/.test(value)};
  const write=(key,value)=>{try{localStorage.setItem(key,JSON.stringify(value))}catch{}};
  const normalizeMessages=history=>Array.isArray(history)?history.filter(x=>x&&(x.role==="user"||x.role==="assistant")&&typeof x.content==="string").slice(-CONFIG.maxHistory):[];
  window.RonQwenTeacher=Object.freeze({
    config:CONFIG,
    isEnabled(){return read(CONFIG.enabledStorageKey,true)!==false},
    setEnabled(value){write(CONFIG.enabledStorageKey,Boolean(value))},
    getEndpoint(){const saved=String(read(CONFIG.endpointStorageKey,DEFAULT_ENDPOINT)).trim();if(/railway\.app/i.test(saved)){try{localStorage.removeItem(CONFIG.endpointStorageKey)}catch{};return DEFAULT_ENDPOINT}return endpointIsValid(saved)?saved:DEFAULT_ENDPOINT},
    setEndpoint(url){write(CONFIG.endpointStorageKey,String(url||CONFIG.defaultEndpoint).trim())},
    getLessons(){const value=read(CONFIG.lessonsStorageKey,[]);return Array.isArray(value)?value.slice(-CONFIG.maxLessons):[]},
    saveLesson(text){const value=String(text||"").trim();if(!value)return false;const lessons=this.getLessons().filter(x=>x.text!==value);lessons.push({text:value,at:new Date().toISOString(),source:CONFIG.model});write(CONFIG.lessonsStorageKey,lessons.slice(-CONFIG.maxLessons));return true},
    getConcepts(){const value=read("ron-concepts-v1",{});return value&&typeof value==="object"?value:{}} ,
    learnConcept(key,value,source="qwen3"){const k=String(key||"").trim(),v=String(value||"").trim();if(!k||!v)return false;const concepts=this.getConcepts();concepts[k]={value:v,source,at:new Date().toISOString()};const entries=Object.entries(concepts).slice(-CONFIG.maxConcepts);write("ron-concepts-v1",Object.fromEntries(entries));return true},
    getTrainingExamples(){const value=read(CONFIG.trainingStorageKey,[]);return Array.isArray(value)?value.slice(-CONFIG.maxTrainingExamples):[]},
    setTrainingExampleStatus(index,status){
      const allowed=new Set(["candidate","approved","rejected"]);
      if(!allowed.has(status))return false;
      const examples=this.getTrainingExamples();
      if(!Number.isInteger(index)||!examples[index])return false;
      examples[index]={...examples[index],status};
      write(CONFIG.trainingStorageKey,examples);
      return true;
    },
    exportTrainingCandidatesJSONL(){return this.getTrainingExamples().map(x=>JSON.stringify({messages:[{role:"user",content:x.user},{role:"assistant",content:x.assistant}],status:x.status,source:x.source,at:x.at})).join("\n");},
    exportApprovedTrainingJSONL(){return this.exportTrainingDataset().map(x=>JSON.stringify(x)).join("\n");},
    learnFromQwen(user,assistant,context=""){const u=String(user||"").trim(),a=String(assistant||"").trim();if(!u||!a)return false;this.recordTrainingExample(u,a,context,"candidate",this.lastProvider||CONFIG.model);return true},
    recordTrainingExample(user,assistant,context,status="candidate",source=CONFIG.model){
      const u=String(user||"").trim(),a=String(assistant||"").trim();if(!u||!a)return false;
      const examples=this.getTrainingExamples().filter(x=>!(x.user===u&&x.assistant===a));
      examples.push({user:u,assistant:a,context:String(context||""),status,source:String(source||CONFIG.model),at:new Date().toISOString()});
      write(CONFIG.trainingStorageKey,examples.slice(-CONFIG.maxTrainingExamples));return true;
    },
    exportTrainingDataset(){return this.getTrainingExamples().filter(x=>x.status==="approved").map(x=>({messages:[{role:"user",content:x.user},{role:"assistant",content:x.assistant}]}));},
    buildMessages(userMessage,ronContext="",history=[]){
      const system=[
        "أنت Qwen، النموذج الأساسي للغة والاستدلال داخل رون.",
        "رون هو النواة الأساسية والمستقلة وصاحب القرار النهائي، وليس تابعًا لـ Qwen أو OpenAI أو Google أو أي جهة أخرى.",
        "افهم الرسالة الحالية بالاعتماد على سياق المحادثة والذاكرة المقدمة.",
        "حافظ على اتساق الأسماء والحقائق والموضوع الجاري، وافهم الأسئلة المختصرة مثل: طيب؟ وماذا عنه؟ بالرجوع إلى السياق.",
        "أجب المستخدم مباشرة وبالعربية الواضحة ما لم يطلب لغة أخرى.",
        "لا تدّع أنك عدّلت ذاكرة رون. إذا كان المستخدم يعلّم رون، أعطِ صياغة معرفة مفيدة يمكن لرون حفظها.",
        ronContext?"ذاكرة رون ذات الصلة:\n"+ronContext:""
      ].filter(Boolean).join("\n\n");
      const concepts=Object.entries(this.getConcepts()).slice(-40).map(([k,v])=>`- ${k}: ${v.value}`).join("\n");
      const learned=globalThis.RonLearning?.search?.(userMessage,12)||[];
      const learnedText=learned.map(x=>`- ${x.text} (confidence ${Number(x.confidence??.5).toFixed(2)})`).join("\n");
      const systemWithLearning=system+(concepts?"\n\nمعرفة متراكمة من رون:\n"+concepts:"")+(learnedText?"\n\nمعرفة مسترجعة من ذاكرة التعلّم:\n"+learnedText:"");
      const messages=[{role:"system",content:systemWithLearning}];messages.push(...normalizeMessages(history));messages.push({role:"user",content:String(userMessage||"")});return messages;
    },
    async ask(userMessage,ronContext="",history=[]){
      if(!this.isEnabled())return{ok:false,reason:"disabled"};
      const endpoint=this.getEndpoint();if(!endpoint)return{ok:false,reason:"no-endpoint"};
      try{
        const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),60000);
        const response=await fetch(endpoint,{method:"POST",headers:{"Content-Type":"application/json","Accept":"application/json"},signal:controller.signal,body:JSON.stringify({model:CONFIG.model,messages:this.buildMessages(userMessage,ronContext,history),temperature:.35,stream:false})});
        clearTimeout(timeout);
        let data=null;try{data=await response.json()}catch{data=null}
        if(!response.ok){const message=data?.error?.message||data?.message||("HTTP "+response.status);return{ok:false,reason:"upstream: "+message}}
        const raw=data?.choices?.[0]?.message?.content??data?.output_text??data?.response??"";
        const text=Array.isArray(raw)?raw.map(x=>typeof x==="string"?x:(x?.text||x?.content||"")).join("").trim():String(raw||"").trim();
        if(!text){const message=data?.error?.message||data?.message||"";return{ok:false,reason:message?"upstream: "+message:"empty"}}
        this.lastProvider=String(data?.provider||"qwen");
        return{ok:true,text,model:String(data?.model||CONFIG.model),provider:this.lastProvider,fallbackUsed:Boolean(data?.fallback_used)}
      }catch(error){return{ok:false,reason:error?.name==="AbortError"?"network: request timeout":"network: "+(error?.message||"request failed")}}
    }
  });
})();