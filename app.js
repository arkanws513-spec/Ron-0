(() => {
const S={chat:"ron-chat-v5",lessons:"ron-lessons-v5",convos:"ron-conversations-v2",started:"ron-started-v1"},LEGACY={chat:"ron-chat-v4",lessons:"ron-lessons-v4",convos:"ron-conversations-v1"},$=id=>document.getElementById(id),chat=$("chat"),form=$("composer"),input=$("input"),send=$("send");
const norm=s=>s.toLowerCase().replace(/[ًٌٍَُِّْـ]/g,"").replace(/[أإآ]/g,"ا").replace(/ى/g,"ي").replace(/\s+/g," ").trim();
const read=(k,d)=>{try{return JSON.parse(localStorage.getItem(k))??d}catch{return d}};
let preferences=read("ron-preferences-v1",{});if(!preferences||typeof preferences!=="object"||Array.isArray(preferences))preferences={};
const savePreferences=()=>{try{localStorage.setItem("ron-preferences-v1",JSON.stringify(preferences))}catch{}};
const asMessages=v=>Array.isArray(v)?v.filter(x=>x&&typeof x.role==="string"&&typeof x.text==="string"):[];
const asLessons=v=>Array.isArray(v)?v.filter(x=>x&&typeof x.key==="string"&&typeof x.text==="string"):[];
let messages=asMessages(read(S.chat,null)),lessons=asLessons(read(S.lessons,null)),conversations=Array.isArray(read(S.convos,[]))?read(S.convos,[]):[];
// One-time migration: keep user facts/explicit lessons, but discard stale model-generated replies from the old runtime.
if(localStorage.getItem("ron-runtime-clean-v1")!=="1"){
  const oldChat=asMessages(read("ron-chat-v4",[]));
  const oldLessons=asLessons(read("ron-lessons-v4",[]));
  if(!messages.length)messages=oldChat;
  lessons=lessons.filter(x=>x.kind!=="response"&&x.kind!=="model-response");
  const explicit=oldLessons.filter(x=>x.kind==="relation"||x.kind==="lesson"||x.key==="user.name"||x.key==="user.age"||x.key==="ron.name"||x.key==="ron.age");
  lessons=[...lessons,...explicit.filter(x=>!lessons.some(y=>y.key===x.key&&y.text===x.text))].slice(-500);
  localStorage.removeItem("ron-chat-v3");localStorage.removeItem("ron-lessons-v3");
  localStorage.setItem("ron-runtime-clean-v1","1");
  try{localStorage.setItem(S.chat,JSON.stringify(messages.slice(-300)));localStorage.setItem(S.lessons,JSON.stringify(lessons.slice(-500)))}catch{}
}
const LEARNING_CLEAN="ron-learning-clean-v1";
if(localStorage.getItem(LEARNING_CLEAN)!=="1"){try{const raw=JSON.parse(localStorage.getItem("ron-learning-core-v1")||"null");if(raw&&Array.isArray(raw.knowledge)){raw.knowledge=raw.knowledge.filter(x=>x&&x.kind!=="response");localStorage.setItem("ron-learning-core-v1",JSON.stringify(raw));}}catch{}localStorage.setItem(LEARNING_CLEAN,"1");}
const FACTS_MIGRATION="ron-facts-migration-v1";
if(localStorage.getItem(FACTS_MIGRATION)!=="1"){lessons=lessons.filter(x=>x.key!=="user.name"&&x.key!=="user.age");localStorage.setItem(FACTS_MIGRATION,"1");}
const save=()=>{try{localStorage.setItem(S.chat,JSON.stringify(messages.slice(-300)));localStorage.setItem(S.lessons,JSON.stringify(lessons.slice(-500)));localStorage.setItem(S.convos,JSON.stringify(conversations.slice(-50)))}catch(err){console.error("Ron storage error",err)}};
if(localStorage.getItem("ron-name-clean-v2")!=="1"){lessons=lessons.filter(x=>!(x.key==="ron.name"&&/^(?:اي|انت رون|اسمك|اسمك انت رون)$/i.test(norm(x.text))));localStorage.setItem("ron-name-clean-v2","1");save();}

const PROFILE_CLEAN="ron-profile-clean-v3";
if(localStorage.getItem(PROFILE_CLEAN)!=="1"){
 const badName=x=>x?.key==="user.name"&&/^(?:اسالك عن اسمك|اسالك|ما اسمي|ما اسمك|اسمي وما اسمك)$/i.test(norm(x.text));
 lessons=lessons.filter(x=>!badName(x));
 try{const raw=JSON.parse(localStorage.getItem("ron-core-data-v2")||"null");if(raw&&Array.isArray(raw.facts)){raw.facts=raw.facts.filter(x=>!(x?.s==="$user"&&norm(x.p)==="اسم"&&/اسالك عن اسمك|اسالك|ما اسمي|ما اسمك|اسمي وما اسمك/i.test(norm(x.o))));localStorage.setItem("ron-core-data-v2",JSON.stringify(raw));}}catch{}
 try{const raw=JSON.parse(localStorage.getItem("ron-learning-core-v1")||"null");if(raw&&Array.isArray(raw.knowledge)){raw.knowledge=raw.knowledge.filter(x=>!/اسمك اسالك عن اسمك|اسمك ما اسمي/i.test(norm(x?.text||"")));localStorage.setItem("ron-learning-core-v1",JSON.stringify(raw));}}catch{}
 localStorage.setItem(PROFILE_CLEAN,"1");
 try{localStorage.setItem(S.lessons,JSON.stringify(lessons.slice(-500)))}catch{}
}

// One-time cleanup for names corrupted by the older relation parser.
const PROFILE_CLEAN_V4="ron-profile-clean-v4";
if(localStorage.getItem(PROFILE_CLEAN_V4)!=="1"){
 const badUserName=x=>x?.key==="user.name"&&(/(?:بسالك عن|اسالك عن|ليس اسمك|وليس اسمك|اسمك بسالك)/i.test(norm(x.text))||String(x.text||"").length>40);
 const badRonName=x=>x?.key==="ron.name"&&(/(?:بسالك عن|اسالك عن|اسمك)/i.test(norm(x.text))||/^(?:رون فعلا|رون حقا|رون بالفعل)$/.test(norm(x.text)));
 lessons=lessons.filter(x=>!badUserName(x)&&!badRonName(x));
 try{
  const raw=JSON.parse(localStorage.getItem("ron-core-data-v2")||"null");
  if(raw&&Array.isArray(raw.facts)){
   raw.facts=raw.facts.filter(x=>{
    if(!x)return false;
    const value=norm(x.o||"");
    if(x.s==="$user"&&norm(x.p)==="اسم")return !/(?:بسالك عن|اسالك عن|ليس اسمك|وليس اسمك|اسمك بسالك)/i.test(value)&&value.length<=40;
    if(x.s==="$self"&&norm(x.p)==="اسم")return !/(?:بسالك عن|اسالك عن|اسمك)/i.test(value)&&!/^(?:رون فعلا|رون حقا|رون بالفعل)$/.test(value);
    return true;
   });
   localStorage.setItem("ron-core-data-v2",JSON.stringify(raw));
  }
 }catch{}
 try{
  const raw=JSON.parse(localStorage.getItem("ron-learning-core-v1")||"null");
  if(raw&&Array.isArray(raw.knowledge)){
   raw.knowledge=raw.knowledge.filter(x=>!/اسمك\s+(?:بسالك عن|اسالك عن)|(?:ليس اسمك|وليس اسمك)|اسمك\s+رون فعلا/i.test(norm(x?.text||"")));
   localStorage.setItem("ron-learning-core-v1",JSON.stringify(raw));
  }
 }catch{}
 localStorage.setItem(PROFILE_CLEAN_V4,"1");
 try{localStorage.setItem(S.lessons,JSON.stringify(lessons.slice(-500)))}catch{}
}
const makeId=()=>Date.now().toString(36)+Math.random().toString(36).slice(2,7);const titleOf=ms=>{const m=ms.find(x=>x.role==="user");return m?m.text.slice(0,42):"محادثة جديدة"};const saveCurrent=()=>{const u=messages.find(x=>x.role==="user");if(!u)return;const id=globalThis.ronConversationId||makeId();conversations=conversations.filter(x=>x.id!==id);conversations.push({id,title:titleOf(messages),messages:messages.slice(-300),updatedAt:new Date().toISOString()});globalThis.ronConversationId=id;save()};const downloadText=(name,text,mime="application/jsonl")=>{const blob=new Blob([text],{type:mime+";charset=utf-8"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)};
const analyzeIntent=text=>{const n=norm(text);if(/^(طيب|طب|وبعدين|وماذا عنه|وماذا عنها|وهل|وضح|اشرح اكثر|كمل|تابع|ابحث عنها|ابحث عن ذلك|دور عليها|دور عن ذلك|اذا ابحث عنها|اذا ابحث عن ذلك|نعم علق عليه|ايوه علق عليه|علق عليه|اشرح كلامي|حلل كلامي|ليه|لماذا|ازاي|كيف ذلك|لماذا ذلك|ما السبب|ما معنى ذلك|ماذا تقصد|هل هذا صحيح)(?:\s*[؟?!.،]*)?$/.test(n))return "follow_up";if(/^(مرحبا|اهلا|السلام عليكم)/.test(n))return "greeting";if(isNameQuestion(n)||isAgeQuestion(n))return "profile";if(/^(ليه|لماذا|ازاي|كيف|ماذا|ما هو|ما هي|ما |ايه |اي |هل|هل يمكن|عايز|اريد|ممكن|من |اين |فين |متى |كم )/.test(n))return "question";if(/\?$/.test(String(text).trim())||/[؟?]/.test(text))return "question";if(/^عل[ّ]?م رون/.test(n))return "learning";return "statement";};
const topicTokens=text=>tokens(String(text||"")).filter(x=>x.length>2);
const topicOf=()=>{const recent=messages.slice().reverse().find(m=>m.role==="user"&&String(m.text||"").trim().length>=5&&!/^(طيب|طب|وبعدين|وماذا عنه|وماذا عنها|وهل|وضح|اشرح اكثر|كمل|تابع|ليه|لماذا|ازاي|كيف ذلك|لماذا ذلك|ما السبب|ما معنى ذلك|ماذا تقصد|هل هذا صحيح)\s*[؟?!.،]*$/.test(norm(m.text)));return recent?String(recent.text).trim():null;};
const resolveFollowUp=text=>{const n=norm(text);if(!/^(?:اذا\s+)?(?:طيب|طب|وبعدين|وماذا عنه|وماذا عنها|وهل|وضح|اشرح اكثر|كمل|تابع|ابحث عنها|ابحث عن ذلك|دور عليها|دور عن ذلك|نعم علق عليه|ايوه علق عليه|علق عليه|اشرح كلامي|حلل كلامي|ليه|لماذا|ازاي|كيف ذلك|لماذا ذلك|ما السبب|ما معنى ذلك|ماذا تقصد|هل هذا صحيح)(?:\s*[؟?!.،]*)?$/.test(n))return null;const current=String(text||"").trim();const previous=messages.slice(0,-1).slice().reverse().find(m=>m.role==="user"&&String(m.text||"").trim().length>=5&&!/^(طيب|طب|وبعدين|وماذا عنه|وماذا عنها|وهل|وضح|اشرح اكثر|كمل|تابع|ابحث عنها|ابحث عن ذلك|دور عليها|دور عن ذلك|اذا ابحث عنها|اذا ابحث عن ذلك|ليه|لماذا|ازاي|كيف ذلك|لماذا ذلك|ما السبب|ما معنى ذلك|ماذا تقصد|هل هذا صحيح)\s*[؟?!.،]*$/.test(norm(m.text)));return previous?String(previous.text).trim():null;};
const relatedMemories=topic=>{if(!topic)return [];const q=new Set(topicTokens(topic));return lessons.map((x,i)=>{const t=new Set(topicTokens(x.text));let score=0;q.forEach(w=>{if(t.has(w))score++});if(x.key.startsWith("lesson:"))score+=0.1;return {x,score,i};}).filter(x=>x.score>0).sort((a,b)=>b.score-a.score||b.i-a.i).slice(0,8).map(x=>x.x);};
const understanding= text=>{const intent=analyzeIntent(text),follow=resolveFollowUp(text),topic=follow||topicOf()||String(text||"").trim(),related=relatedMemories(topic);return {intent,topic,shortHistory:messages.slice(-12).map(m=>(m.role==="ron"?"رون: ":"المستخدم: ")+m.text),longMemory:related.map(x=>x.key+": "+x.text),followUp:!!follow};};
const conversationHistory=()=>messages.slice(-14).map(m=>({role:m.role==="ron"?"assistant":"user",content:String(m.text||"")}));
if(!messages.length){messages=asMessages(read(LEGACY.chat,[]));}
if(!messages.length)messages=[{role:"ron",text:"مرحبًا. أنا رون. النواة المحلية تعمل، وذاكرتي محفوظة على هذا الجهاز."}];
if(!lessons.length)lessons=asLessons(read(LEGACY.lessons,[]));
const bubble=(role,text)=>{const e=document.createElement("article");e.className="message "+role;const b=document.createElement("b"),p=document.createElement("p");b.textContent=role==="ron"?"رون":"أنت";p.textContent=text;e.append(b,p);chat.appendChild(e);chat.scrollTop=chat.scrollHeight};
const render=()=>{chat.replaceChildren();messages.forEach(m=>bubble(m.role,m.text))};
const cleanValue=s=>s.replace(/^[\s.،,؛;:]+|[\s.!؟?،,؛;:]+$/g,"").trim();
const isSafeProfileName=value=>{
 const raw=String(value??"").trim(),n=norm(cleanValue(raw)),parts=n.split(" ").filter(Boolean);
 return !!n&&parts.length<=4&&!/^(?:لا|ليس|مش|مو|ما|بسالك|اسالك|عن|اسمك|اسمي|لكن|ولكن|وليس|بل)(?:\s|$)/.test(n)&&!/(?:^|\s)(?:وليس|لكن|ولكن|مش|مو|بل|لا|ليس|بسالك|اسالك|عن|اسمي|اسمك)$/.test(n)&&!/(?:^|\s)(?:بسالك|اسالك)\s+عن(?:\s|$)/.test(n)&&!/(?:^|\s)(?:اسمك|اسمي)(?:\s|$)/.test(n)&&!/[؟?]/.test(raw);
};
const PROFILE_CLEAN_V5="ron-profile-clean-v5";
if(localStorage.getItem(PROFILE_CLEAN_V5)!=="1"){
 lessons=lessons.filter(x=>x?.key!=="user.name"||isSafeProfileName(x.text)).filter(x=>x?.key!=="ron.name"||norm(x.text)==="رون");
 try{const raw=JSON.parse(localStorage.getItem("ron-core-data-v2")||"null");if(raw&&Array.isArray(raw.facts)){raw.facts=raw.facts.filter(x=>{if(x?.s==="$user"&&norm(x.p)==="اسم")return isSafeProfileName(x.o);if(x?.s==="$self"&&norm(x.p)==="اسم")return norm(x.o)==="رون";return true;});localStorage.setItem("ron-core-data-v2",JSON.stringify(raw));}}catch{}
 try{const raw=JSON.parse(localStorage.getItem("ron-learning-core-v1")||"null");if(raw&&Array.isArray(raw.knowledge)){raw.knowledge=raw.knowledge.filter(x=>{if(x?.kind!=="fact")return true;const t=norm(x.text||"");let m=t.match(/^اسمك\s+هو\s+(.+)$/);if(m)return isSafeProfileName(m[1]);m=t.match(/^اسمي\s+هو\s+(.+)$/);if(m)return norm(m[1])==="رون";return true;});localStorage.setItem("ron-learning-core-v1",JSON.stringify(raw));}}catch{}
 localStorage.setItem(PROFILE_CLEAN_V5,"1");
 try{localStorage.setItem(S.lessons,JSON.stringify(lessons.slice(-500)))}catch{}
}
const saveFact=(key,text)=>{lessons=lessons.filter(x=>x.key!==key);lessons.push({key,text,at:new Date().toISOString()});save()};
const extractFacts=t=>{
 const n=norm(t).replace(/^\.\s*/,"");
 const facts=[];
 // Questions/meta-conversation are not identity assertions.
 if(/[؟?]/.test(String(t))||/^(?:انا\s+)?(?:بسالك|اسالك)\s+عن\s+اسمي(?=\s|$)/.test(n))return facts;
 // Handle explicit corrections as one user-name fact; never parse the negated clause as Ron's name.
 const corrected=n.match(/^(?:انا\s+)?اسمي\s+(.+?)\s+(?:وليس|لكن|ولكن|مش|مو)\s+(?:اسمك|اسم|انت|انا)(?=\s|$).*$/);
 if(corrected){const value=cleanValue(corrected[1]);if(isSafeProfileName(value))facts.push({key:"user.name",text:value});return facts;}
 let both=n.match(/^اسمي\s+(.+?)\s*[،,]?\s+(?:و)?اسمك\s+(?:هو\s+)?(.+)$/);
 if(both){const userName=cleanValue(both[1]),ronName=cleanValue(both[2]).replace(/^انت\s+/,"").trim();if(!isSafeProfileName(userName))return facts;facts.push({key:"user.name",text:userName});if(isSafeProfileName(ronName)&&norm(ronName)==="رون")facts.push({key:"ron.name",text:"رون"});return facts;}
 let m=n.match(/(?:^|\s)(?:انا\s+)?(?:عمري|سني)\s+(\d{1,3})\s*(?:عام|سنة|سنين)?(?=\s|$)/);
 if(!m)m=n.match(/(?:^|\s)(?:انا\s+)?(\d{1,3})\s*(?:عام|سنة|سنين)(?=\s|$)/);
 if(m){const age=Number(m[1]);if(age>=1&&age<=120)facts.push({key:"user.age",text:String(age)});}
 m=n.match(/^(?:انا\s+)?اسمي\s+(?:هو\s+)?(.+?)\s*[.!؟?،,؛;]*$/);
 if(!m)m=n.match(/^(?:اريدك\s+)?(?:ان\s+)?تعلم\s+(?:ان\s+)?اسمي\s+(?:هو\s+)?(.+?)\s*[.!؟?،,؛;]*$/);
 if(!m)m=n.match(/^تعلم\s+ان\s+اسمي\s+(?:هو\s+)?(.+?)\s*[.!؟?،,؛;]*$/);
 if(m){
  let value=cleanValue(m[1]);
  const correction=value.match(/^(.+?)\s+فقط\s+(?:اما|لكن)\s+(\d{1,3})\s*(?:عام|سنة|سنين)\s*$/);
  if(correction){value=cleanValue(correction[1]);const age=Number(correction[2]);if(age>=1&&age<=120)facts.push({key:"user.age",text:String(age)});}
  value=value.replace(/\s+(?:و)?عمري\s+\d{1,3}\s*(?:عام|سنة|سنين)?$/,"").replace(/\s+\d{1,3}\s*(?:عام|سنة|سنين)$/,"").trim();
  if(isSafeProfileName(value))facts.push({key:"user.name",text:value});
}
 if(!facts.some(x=>x.key==="user.name")){
  m=n.match(/^انا\s+(.+?)\s+وعمري\s+\d{1,3}\s*(?:عام|سنة|سنين)?$/);
  if(m){
   const value=cleanValue(m[1]);
   if(isSafeProfileName(value)&&!/^(?:عمري|سني|احب|لا احب)\b/.test(value))facts.push({key:"user.name",text:value});
  }
 }
 if(!facts.some(x=>x.key==="user.name")){
  m=n.match(/^انا\s+(.+?)$/);
  if(m&&!/^(?:عمري|سني|احب|لا احب|بسالك|اسالك|عايز|اريد|بقولك|اقصد|مش|ليس|عن)(?:\s|$)/.test(m[1])){
   const value=cleanValue(m[1]);
   if(isSafeProfileName(value))facts.push({key:"user.name",text:value});
  }
 }
 return facts;
};
const extractFact=t=>extractFacts(t)[0]||null;
const queryRonLocalModel=async text=>{
 try{
  const configured=typeof window.RON_API_BASE_URL==="string"&&window.RON_API_BASE_URL.trim()?window.RON_API_BASE_URL.trim():"https://accurate-courage-production-8a31.up.railway.app";
  const base=configured.replace(/\/+$/,"");
  const response=await fetch(base+"/api/chat",{
   method:"POST",
   headers:{"Content-Type":"application/json"},
   cache:"no-store",
   body:JSON.stringify({message:text,session_id:nativeApiSessionId,request_id:"r"+Date.now().toString(36)+Math.random().toString(36).slice(2,8)})
  });
  if(!response.ok)return null;
  const payload=await response.json();
  return typeof payload.reply==="string"&&payload.reply.trim()?{reply:payload.reply.trim(),generationStatus:String(payload.generation_status||""),health:payload.health||null}:null;
 }catch(error){console.debug("Ron local Qwen API unavailable; using browser fallback.",error);return null;}
};
const answer=t=>{
 const n=norm(t),facts=extractFacts(t),ronFacts=extractRonFacts(t),fact=facts[0]||null;
 if(/^(?:اذا\\s+)?هل تستطيع ان تشرح لي فكر(?:ه|ة)$/.test(n))return "نعم، بالتأكيد. ما الفكرة التي تريد أن أشرحها لك؟";
 if(/^(?:ماذا تعني|ماذا تقصد|ما معنى ذلك)$/.test(n)){
  const previous=messages.slice(0,-1).slice().reverse().find(m=>m.role==="ron");
  const priorUser=messages.slice(0,-1).slice().reverse().find(m=>m.role==="user");
  if(previous&&!nativeReplyLooksUsable(previous.text,priorUser?.text||""))return "أعتذر، كان ردي السابق غير واضح. ما الفكرة أو الكلمة التي تريد أن أشرحها ببساطة؟";
  if(previous)return "هل تقصد ردي السابق؟ أستطيع توضيحه، أو يمكنك تحديد الكلمة التي لم تكن واضحة.";
  return "ما الكلمة أو الفكرة التي تريد أن أشرحها؟";
 }
 if(isEgyptCapitalPreference(n)){preferences.shortEgyptCapital=true;savePreferences();return "تم. سأجيب عن عاصمة مصر بكلمة «القاهرة» فقط.";}
 if(isReadinessQuestion(n))return "أيوه، جاهز.";
 if(isWithMeQuestion(n))return "أيوه، أنا معاك. قولّي نكمل منين.";
 if(isPreviousReplyQuestion(n))return explainPreviousReply();
 if(isPendingCommentConfirmation(n))return commentOnPreviousUserMessage();
 if(isCommentCommand(n))return commentOnPreviousUserMessage();
 if(isIncompleteMathQuestion(n))return "ما العملية الحسابية التي تريد حسابها؟";
 if(isEgyptCapitalQuestion(n)&&preferences.shortEgyptCapital)return "القاهرة";
 if(/^(?:اسمك|اسمك هو|انت اسمك)\s+(?:رون\s+)?(?:فعلا|حقا)[؟?]*$/.test(n))return "نعم، اسمي رون.";
 if(/^(مرحبا|اهلا|أهلا|السلام عليكم|سلام|هاي|هلا)([!！،,. ]*)$/.test(n))return "مرحبًا. أنا رون، وجاهز لمساعدتك.";
 if(/^(ازيك|إزيك|كيف حالك|عامل ايه|عامل إيه)([؟? !،,.]*)$/.test(n))return "أنا بخير وجاهز للعمل. ماذا تريد أن نفعل؟";
 if(/^(ماذا تستطيع|ايه اللي تقدر تعمله|ماذا يمكنك ان تفعل|ماذا يمكنك|ما الذي تستطيع فعله|ما هي قدراتك|ايه قدراتك|قدراتك)$/.test(n))return "أستطيع فهم المحادثة، حفظ ما تعلّمني إياه، استخدام معرفتي المحلية، إجراء استدلال بسيط، والبحث في الإنترنت عندما أحتاج معلومة غير موجودة لدي. وويمكنني الاعتماد على نواتي المحلية والبحث في الإنترنت عند الحاجة.";
 if(/^(?:(?:هذا|دي|ده|دا)\s+)?(?:عظيم|رائع|ممتاز|جميل|جيد|جيد جدا|جيد جدًا|حلو|كويس|احسنت|أحسنت|تمام|شكرا|شكرًا|شكراً)$/.test(n))return "شكرًا! أنا جاهز نكمل.";
 if(/^عل[ّ]?م رون\s*(?::|،|,|-)/.test(n))return teach(t);
 if(ronFacts.length&&!isRonNameQuestion(n)&&!isRonAgeQuestion(n)&&!isNameQuestion(n)&&!isAgeQuestion(n)){ronFacts.forEach(x=>saveFact(x.key,x.text));const rn=ronFacts.find(x=>x.key==="ron.name"),ra=ronFacts.find(x=>x.key==="ron.age");if(rn&&ra)return "تم. حفظت أن اسمي "+rn.text+" وأن عمري "+ra.text+".";if(ra)return "تم. حفظت أن عمري "+ra.text+".";return "تم. حفظت أن اسمي "+rn.text+".";}
 if(/^(?:نعم\s+)?(?:علق\s+عليه|علّق\s+عليه|اشرح\s+كلامي|حلل\s+كلامي)$/.test(n)){
  const prior=messages.slice(0,-1).slice().reverse().find(m=>m.role==="user"&&!/^(?:مظبوط|صح|تمام|نعم|ايوه|أيوه|اه|اها|ممتاز|جميل|شكرا|شكرًا|نعم علق عليه|علق عليه|اشرح كلامي|حلل كلامي)$/.test(norm(m.text)));
  const savedName=lessons.find(x=>x.key==="user.name");
  if(prior&&/^(?:انا\s+)?اسمي\s+/.test(norm(prior.text))&&savedName)return "تعليقًا على كلامك: عرّفتني باسمك، وقد حفظت الاسم «"+savedName.text+"» لأستخدمه عند الحاجة في المحادثة.";
  if(prior)return "تعليقًا على كلامك «"+prior.text+"»: فهمت أنك تريد مناقشته، لكن أحتاج إلى معرفة أي جانب تريد تحليله تحديدًا.";
  return "بالتأكيد، أستطيع التعليق، لكن لا أجد في سياق المحادثة كلامًا سابقًا واضحًا لأعلّق عليه.";
 }
 if(isBothNamesQuestion(n)){
  const name=lessons.find(x=>x.key==="user.name"),self=lessons.find(x=>x.key==="ron.name");
  return (name?("اسمك "+name.text):"لم تخبرني باسمك بعد.")+"، واسمي "+(self?self.text:"رون")+".";
 }
 if(isProfileQuestion(n)){
  const name=lessons.find(x=>x.key==="user.name"),age=lessons.find(x=>x.key==="user.age");
  return (name?("اسمك "+name.text):"لم تخبرني باسمك بعد.")+"، "+(age?("وعمرك "+age.text+" سنة."): "ولم تخبرني بعمرك بعد.");
 }
 if(isRonNameQuestion(n)){const x=lessons.find(x=>x.key==="ron.name");return x?"اسمي "+x.text+".":"اسمي رون.";}
 if(isRonAgeQuestion(n)){const x=lessons.find(x=>x.key==="ron.age");return x?"عمري المسجل هو "+x.text+".":"ليس لدي عمر بشري؛ أنا برنامج، ولا أملك عمرًا شخصيًا مثل الإنسان."; }
 if(facts.length){
  facts.forEach(x=>saveFact(x.key,x.text));
  const name=facts.find(x=>x.key==="user.name"),age=facts.find(x=>x.key==="user.age");
  if(name&&age)return "تم. سأحفظ أن اسمك "+name.text+" وأن عمرك "+age.text+" سنة.";
  if(age)return "تم. سأحفظ أن عمرك "+age.text+" سنة.";
  return "تم. سأحفظ أن اسمك "+name.text+".";
 }
 if(isNameQuestion(n)){const x=lessons.find(x=>x.key==="user.name");return x?"اسمك "+x.text+".":"لم تخبرني باسمك بعد.";}
 if(isAgeQuestion(n)){const x=lessons.find(x=>x.key==="user.age");return x?"عمرك "+x.text+" سنة.":"لم تخبرني بعمرك بعد.";}
 if(n.includes("ماذا تعلمت")||n.includes("ما الذي تعلمته"))return lessons.length?"هذه آخر تعليماتي المحفوظة:\n\n"+lessons.slice(-10).map((x,i)=>i+1+". "+x.text).join("\n"):"لم تعلّمني شيئًا بعد.";
 if(n.includes("امسح الذاكره")||n.includes("امسح الذاكرة")){lessons=[];save();return"تم مسح الذاكرة التي علّمتني إياها."}
 const stopWords=new Set(["ما","ماذا","ماهي","ماهو","هي","هو","هل","من","في","عن","الى","هذا","هذه","ذلك","تلك","اي","اية","ايه","يا","رون","انا","ان","و","او","ال","هو","هي","الذي","التي","هل"]);
const tokens=s=>norm(s).replace(/[؟?!.,،؛;:()\[\]{}]/g," ").split(" ").filter(w=>w.length>=2&&!stopWords.has(w));
const localLessonAnswer=n=>{
 const query=norm(n);
 const capitalMatch=query.match(/^(?:ما هي|ماهو|ما هو|ايه|اي)\s+(?:عاصمة|عاصمه)\s+(.+?)[؟?]?$/);
 if(capitalMatch){
  const place=cleanValue(capitalMatch[1]);
  const candidates=lessons.filter(x=>x.kind==="relation"&&x.relation==="capital_of"&&norm(x.object)===norm(place));
  const best=candidates[candidates.length-1];
  if(best)return best.subject+" هي عاصمة "+best.object+".";
 }
 return null;
};
 const semantic=null;
 if(semantic?.answer)return semantic.answer;
 const hit=localLessonAnswer(n);
 if(hit)return hit;
 return null;
};
const sendMessage=()=>{const t=input.value.trim();if(!t)return;messages.push({role:"user",text:t});bubble("user",t);input.value="";input.style.height="auto";if(send){send.disabled=true;send.classList?.add("thinking");if(send.dataset)send.dataset.originalText=send.textContent||"إرسال";send.textContent="رون يفكر";}setTimeout(async()=>{try{
 let r=null;
 const n=norm(t);
 const localMemoryCommand=extractFacts(t).length>0||extractRonFacts(t).length>0||/^عل[ّ]?م رون\s*(?::|،|,|-)/.test(n)||isNameQuestion(n)||isAgeQuestion(n)||isRonNameQuestion(n)||isRonAgeQuestion(n)||isBothNamesQuestion(n)||isProfileQuestion(n)||n.includes("ماذا تعلمت")||n.includes("ما الذي تعلمته")||n.includes("امسح الذاكره")||n.includes("امسح الذاكرة")||isEgyptCapitalPreference(n);
 if(localMemoryCommand){r=answer(t);}else{
  const generated=await queryRonLocalModel(t);
  if(generated&&generated.reply)r=generated.reply;
  else r="تعذّر الاتصال بمحرك الاستدلال أو توليد الرد. لم أستبدل إجابة النموذج بردّ من الذاكرة؛ تحقّق من تشغيل خادم رون وتحميل أوزان Qwen.";
 }
 messages.push({role:"ron",text:String(r)});bubble("ron",String(r));save();saveCurrent();
 }catch(err){console.error("Ron inference error",err);const failure="حدث خطأ أثناء تشغيل نموذج رون. لم يتم إنشاء رد بديل من الذاكرة؛ راجع سجل خادم الاستدلال.";messages.push({role:"ron",text:failure});bubble("ron",failure)}finally{send.disabled=false;send.classList?.remove("thinking");send.textContent=send.dataset?.originalText||"إرسال";input.focus?.()}},80)};
 form.addEventListener("submit",e=>{e.preventDefault();sendMessage()});
send.addEventListener("click",e=>{e.preventDefault();sendMessage()});input.addEventListener("input",()=>{input.style.height="auto";input.style.height=Math.min(input.scrollHeight,140)+"px"});input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();sendMessage()}});
$("start-ron")?.addEventListener("click",()=>{localStorage.setItem(S.started,"1");$("startup").classList.add("hidden");$("main-app").classList.remove("hidden");input.focus()});if(localStorage.getItem(S.started)==="1"){$("startup").classList.add("hidden");$("main-app").classList.remove("hidden")};$("menu").addEventListener("click",()=>{renderHistory();$("settings").classList.add("open")});$("close-settings").addEventListener("click",()=>$("settings").classList.remove("open"));const renderHistory=()=>{const box=$("history-list");if(!box)return;box.replaceChildren();if(!conversations.length){const e=document.createElement("div");e.className="history-empty";e.textContent="لا توجد محادثات محفوظة بعد.";box.appendChild(e);return}conversations.slice().reverse().forEach(x=>{const b=document.createElement("button");b.type="button";b.className="history-item";b.textContent=x.title||"محادثة";b.addEventListener("click",()=>{messages=asMessages(x.messages);globalThis.ronConversationId=x.id;save();render();$("settings").classList.remove("open")});box.appendChild(b)})};$("new-chat").addEventListener("click",()=>{saveCurrent();messages=[{role:"ron",text:"بدأنا محادثة جديدة. كيف يمكنني مساعدتك؟"}];globalThis.ronConversationId=makeId();save();render();$("settings").classList.remove("open")});$("history-btn")?.addEventListener("click",()=>{$("history-list")?.classList.toggle("open");renderHistory()});
const exportTraining=$("export-training"),exportApproved=$("export-approved");
$("clear-data").addEventListener("click",()=>{if(confirm("مسح كل بيانات رون المحلية؟")){localStorage.clear();location.reload()}});
$("export").addEventListener("click",()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify({version:5,exportedAt:new Date().toISOString(),messages,lessons,conversations},null,2)],{type:"application/json"}));a.download="ron-backup.json";a.click()});
$("import").addEventListener("change",async e=>{const f=e.target.files[0];if(!f)return;try{const d=JSON.parse(await f.text());if(!Array.isArray(d.messages)||!Array.isArray(d.lessons))throw 0;messages=d.messages;lessons=d.lessons;conversations=Array.isArray(d.conversations)?d.conversations:conversations;save();render();alert("تم استيراد ذاكرة رون.")}catch{alert("ملف النسخ الاحتياطي غير صالح.")}e.target.value=""});render();
})();