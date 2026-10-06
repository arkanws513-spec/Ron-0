(() => {
const S={chat:"ron-chat-v4",lessons:"ron-lessons-v4"},LEGACY={chat:"ron-chat-v3",lessons:"ron-lessons-v3"},$=id=>document.getElementById(id),chat=$("chat"),form=$("composer"),input=$("input"),send=$("send");
const read=(k,d)=>{try{return JSON.parse(localStorage.getItem(k))??d}catch{return d}};
const asMessages=v=>Array.isArray(v)?v.filter(x=>x&&typeof x.role==="string"&&typeof x.text==="string"):[];
const asLessons=v=>Array.isArray(v)?v.filter(x=>x&&typeof x.key==="string"&&typeof x.text==="string"):[];
let messages=asMessages(read(S.chat,null)),lessons=asLessons(read(S.lessons,null));
const save=()=>{try{localStorage.setItem(S.chat,JSON.stringify(messages.slice(-200)));localStorage.setItem(S.lessons,JSON.stringify(lessons.slice(-500)))}catch(err){console.error("Ron storage error",err)}};
if(!messages.length){messages=asMessages(read(LEGACY.chat,[]));}
if(!messages.length)messages=[{role:"ron",text:"مرحبًا. أنا رون. النواة المحلية تعمل، وذاكرتي محفوظة على هذا الجهاز."}];
if(!lessons.length)lessons=asLessons(read(LEGACY.lessons,[]));
const bubble=(role,text)=>{const e=document.createElement("article");e.className="message "+role;const b=document.createElement("b"),p=document.createElement("p");b.textContent=role==="ron"?"رون":"أنت";p.textContent=text;e.append(b,p);chat.appendChild(e);chat.scrollTop=chat.scrollHeight};
const render=()=>{chat.replaceChildren();messages.forEach(m=>bubble(m.role,m.text))};
const norm=s=>s.toLowerCase().replace(/[ًٌٍَُِّْـ]/g,"").replace(/[أإآ]/g,"ا").replace(/ى/g,"ي").replace(/\s+/g," ").trim();
const cleanValue=s=>s.replace(/^[\s.،,؛;:]+|[\s.!؟?،,؛;:]+$/g,"").trim();
const saveFact=(key,text)=>{lessons=lessons.filter(x=>x.key!==key);lessons.push({key,text,at:new Date().toISOString()});save()};
const extractFacts=t=>{
 const n=norm(t).replace(/^\.\s*/,"");
 const facts=[];
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
  if(value)facts.push({key:"user.name",text:value});
}
 if(!facts.length){
  m=n.match(/^انا\s+(.+?)$/);
  if(m&&!/^(?:عمري|سني|احب|لا احب)\b/.test(m[1])){
   let value=cleanValue(m[1]).replace(/\s+وعمري\s+\d{1,3}\s*(?:عام|سنة|سنين)?$/,"").trim();
   if(value)facts.push({key:"user.name",text:value});
  }
 }
 return facts;
};
const extractFact=t=>extractFacts(t)[0]||null;
const isRonNameStatement=n=>/^(?:انت\s+)?(?:اسمك\s+هو|اسمك|انت\s+اسمك)\s+رون$/.test(n)||/^اسمك\s+رون$/.test(n);
const isNameQuestion=n=>n.includes("ما اسمي")||n.includes("ايه اسمي")||n.includes("اي اسمي")||n.includes("هل تتذكر اسمي");
const isRonNameQuestion=n=>n.includes("ما اسمك")||n.includes("ايه اسمك")||n.includes("ما هو اسمك")||n==="وانت"||n==="وانت؟";
const isBothNamesQuestion=n=>n.includes("ما اسمي وما اسمك")||n.includes("اسمي واسمك")||n.includes("ما اسمي واسمك");
const isAgeQuestion=n=>n.includes("كم عمري")||n.includes("ما عمري")||n.includes("ما هو عمري")||n.includes("عندي كام سنة")||n.includes("هل تتذكر عمري");
const isProfileQuestion=n=>(isNameQuestion(n)||isAgeQuestion(n))&&(isNameQuestion(n)&&isAgeQuestion(n));
const teach=t=>{const x=t.replace(/^\s*عل[ّ]?م\s+رون\s*:\s*/i,"").trim();if(!x)return"اكتب المعلومة بعد «علّم رون:».";
 lessons.push({key:"lesson:"+Date.now(),text:x,at:new Date().toISOString()});save();return"تم حفظ التعليم في ذاكرة رون المحلية."};
const answer=t=>{
 const n=norm(t),facts=extractFacts(t),fact=facts[0]||null;
 if(/^عل[ّ]?م رون\s*:/.test(n))return teach(t);
 if(isBothNamesQuestion(n)){
  const name=lessons.find(x=>x.key==="user.name"),age=lessons.find(x=>x.key==="user.age");
  return (name?("اسمك "+name.text):"لم تخبرني باسمك بعد.")+", "+(age?("وعمرك "+age.text+" سنة."): "ولم تخبرني بعمرك بعد.");
 }
 if(isProfileQuestion(n)){
  const name=lessons.find(x=>x.key==="user.name"),age=lessons.find(x=>x.key==="user.age");
  return (name?("اسمك "+name.text):"لم تخبرني باسمك بعد.")+"، "+(age?("وعمرك "+age.text+" سنة."): "ولم تخبرني بعمرك بعد.");
 }
 if(isRonNameQuestion(n))return "اسمي رون.";
 if(isRonNameStatement(n))return "صحيح. اسمي رون، وأنت صاحب الاسم الذي أخبرتني به.";
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
 if(/^(مرحبا|اهلا|السلام عليكم)(\s+رون)?/.test(n))return"أهلًا بك. أنا رون. كيف يمكنني مساعدتك؟";
 if(n.includes("من انت"))return"أنا رون، مشروع مساعد مستقل. اسمي رون.";
 if(n.includes("كيف حالك"))return"أنا بخير وأعمل محليًا. أخبرني بما تريد أن نفعله.";
 const hit=lessons.slice().reverse().find(x=>x.text&&n.includes(norm(x.text).slice(0,Math.min(30,norm(x.text).length))));
 return hit?"أتذكر تعليمك: "+hit.text:"وصلتني رسالتك. النواة المحلية تعمل، وما زالت طبقة النموذج المتقدم قيد البناء.";
};
const sendMessage=()=>{const t=input.value.trim();if(!t)return;messages.push({role:"user",text:t});bubble("user",t);input.value="";input.style.height="auto";send.disabled=true;setTimeout(()=>{try{const r=answer(t);messages.push({role:"ron",text:String(r||"حدث خطأ غير متوقع.")});bubble("ron",String(r||"حدث خطأ غير متوقع."));save()}catch(err){console.error("Ron response error",err);const r="حدث خطأ مؤقت داخل النواة المحلية. أعد إرسال الرسالة.";messages.push({role:"ron",text:r});bubble("ron",r)}finally{send.disabled=false;input.focus()}},50)};
form.addEventListener("submit",e=>{e.preventDefault();sendMessage()});input.addEventListener("input",()=>{input.style.height="auto";input.style.height=Math.min(input.scrollHeight,140)+"px"});input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();sendMessage()}});
$("menu").addEventListener("click",()=>$("settings").classList.add("open"));$("close-settings").addEventListener("click",()=>$("settings").classList.remove("open"));$("new-chat").addEventListener("click",()=>{messages=[{role:"ron",text:"بدأنا محادثة جديدة. كيف يمكنني مساعدتك؟"}];save();render();$("settings").classList.remove("open")});
$("clear-data").addEventListener("click",()=>{if(confirm("مسح كل بيانات رون المحلية؟")){localStorage.clear();location.reload()}});
$("export").addEventListener("click",()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify({version:4,exportedAt:new Date().toISOString(),messages,lessons},null,2)],{type:"application/json"}));a.download="ron-backup.json";a.click()});
$("import").addEventListener("change",async e=>{const f=e.target.files[0];if(!f)return;try{const d=JSON.parse(await f.text());if(!Array.isArray(d.messages)||!Array.isArray(d.lessons))throw 0;messages=d.messages;lessons=d.lessons;save();render();alert("تم استيراد ذاكرة رون.")}catch{alert("ملف النسخ الاحتياطي غير صالح.")}e.target.value=""});render();
})();