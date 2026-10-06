(() => {
const S={chat:"ron-chat-v4",lessons:"ron-lessons-v4"},$=id=>document.getElementById(id),chat=$("chat"),form=$("composer"),input=$("input"),send=$("send");
const read=(k,d)=>{try{return JSON.parse(localStorage.getItem(k))??d}catch{return d}},save=()=>{localStorage.setItem(S.chat,JSON.stringify(messages.slice(-200)));localStorage.setItem(S.lessons,JSON.stringify(lessons.slice(-500)))};
let messages=read(S.chat,[{role:"ron",text:"مرحبًا. أنا رون. النواة المحلية تعمل، وذاكرتي محفوظة على هذا الجهاز."}]),lessons=read(S.lessons,[]);
const bubble=(role,text)=>{const e=document.createElement("article");e.className="message "+role;const b=document.createElement("b"),p=document.createElement("p");b.textContent=role==="ron"?"رون":"أنت";p.textContent=text;e.append(b,p);chat.appendChild(e);chat.scrollTop=chat.scrollHeight};
const render=()=>{chat.replaceChildren();messages.forEach(m=>bubble(m.role,m.text))};
const norm=s=>s.toLowerCase().replace(/[ًٌٍَُِّْـ]/g,"").replace(/[أإآ]/g,"ا").replace(/ى/g,"ي").replace(/\s+/g," ").trim();
const cleanValue=s=>s.replace(/^[\s.،,؛;:]+|[\s.!؟?،,؛;:]+$/g,"").trim();
const saveFact=(key,text)=>{lessons=lessons.filter(x=>x.key!==key);lessons.push({key,text,at:new Date().toISOString()});save()};
const extractFact=t=>{
 const n=norm(t).replace(/^\.+\s*/,"");
 let m=n.match(/^(?:انا\s+)?اسمي\s+(?:هو\s+)?(.+?)\s*[.!؟?،,؛;]*$/);
 if(!m)m=n.match(/^(?:اريدك\s+)?(?:ان\s+)?تعلم\s+(?:ان\s+)?اسمي\s+(?:هو\s+)?(.+?)\s*[.!؟?،,؛;]*$/);
 if(!m)m=n.match(/^تعلم\s+ان\s+اسمي\s+(?:هو\s+)?(.+?)\s*[.!؟?،,؛;]*$/);
 if(m){const value=cleanValue(m[1]);if(value)return {key:"user.name",text:value};}
 return null;
};
const isRonNameStatement=n=>/^(?:انت\s+)?(?:اسمك\s+هو|اسمك|انت\s+اسمك)\s+رون$/.test(n)||/^اسمك\s+رون$/.test(n);
const isNameQuestion=n=>n.includes("ما اسمي")||n.includes("ايه اسمي")||n.includes("اي اسمي")||n.includes("هل تتذكر اسمي");
const isRonNameQuestion=n=>n.includes("ما اسمك")||n.includes("ايه اسمك")||n.includes("ما هو اسمك")||n==="وانت"||n==="وانت؟";
const isBothNamesQuestion=n=>n.includes("ما اسمي وما اسمك")||n.includes("اسمي واسمك");
const teach=t=>{const x=t.replace(/^\s*عل[ّ]?م\s+رون\s*:\s*/i,"").trim();if(!x)return"اكتب المعلومة بعد «علّم رون:».";
 lessons.push({key:"lesson:"+Date.now(),text:x,at:new Date().toISOString()});save();return"تم حفظ التعليم في ذاكرة رون المحلية."};
const answer=t=>{
 const n=norm(t),fact=extractFact(t);
 if(/^عل[ّ]?م رون\s*:/.test(n))return teach(t);
 if(isBothNamesQuestion(n))return "اسمك زيريوس، واسمي رون.";
 if(isRonNameQuestion(n))return "اسمي رون.";
 if(isRonNameStatement(n))return "صحيح. اسمي رون، وأنت زيريوس إذا كان هذا هو الاسم الذي أخبرتني به.";
 if(fact){saveFact(fact.key,fact.text);return "تم. سأحفظ أن اسمك "+fact.text+".";}
 if(isNameQuestion(n)){const x=lessons.find(x=>x.key==="user.name");return x?"اسمك "+x.text+".":"لم تخبرني باسمك بعد.";}
 if(n.includes("ماذا تعلمت")||n.includes("ما الذي تعلمته"))return lessons.length?"هذه آخر تعليماتي المحفوظة:\n\n"+lessons.slice(-10).map((x,i)=>i+1+". "+x.text).join("\n"):"لم تعلّمني شيئًا بعد.";
 if(n.includes("امسح الذاكره")||n.includes("امسح الذاكرة")){lessons=[];save();return"تم مسح الذاكرة التي علّمتني إياها."}
 if(/^(مرحبا|اهلا|السلام عليكم)(\s+رون)?/.test(n))return"أهلًا بك. أنا رون. كيف يمكنني مساعدتك؟";
 if(n.includes("من انت"))return"أنا رون، مشروع مساعد مستقل. اسمي رون.";
 if(n.includes("كيف حالك"))return"أنا بخير وأعمل محليًا. أخبرني بما تريد أن نفعله.";
 const hit=lessons.slice().reverse().find(x=>x.text&&n.includes(norm(x.text).slice(0,Math.min(30,norm(x.text).length))));
 return hit?"أتذكر تعليمك: "+hit.text:"وصلتني رسالتك. النواة المحلية تعمل، وما زالت طبقة النموذج المتقدم قيد البناء.";
};
const sendMessage=()=>{const t=input.value.trim();if(!t)return;messages.push({role:"user",text:t});bubble("user",t);input.value="";input.style.height="auto";send.disabled=true;setTimeout(()=>{const r=answer(t);messages.push({role:"ron",text:r});bubble("ron",r);save();send.disabled=false;input.focus()},100)};
form.addEventListener("submit",e=>{e.preventDefault();sendMessage()});input.addEventListener("input",()=>{input.style.height="auto";input.style.height=Math.min(input.scrollHeight,140)+"px"});input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();sendMessage()}});
$("menu").addEventListener("click",()=>$("settings").classList.add("open"));$("close-settings").addEventListener("click",()=>$("settings").classList.remove("open"));$("new-chat").addEventListener("click",()=>{messages=[{role:"ron",text:"بدأنا محادثة جديدة. كيف يمكنني مساعدتك؟"}];save();render();$("settings").classList.remove("open")});
$("clear-data").addEventListener("click",()=>{if(confirm("مسح كل بيانات رون المحلية؟")){localStorage.clear();location.reload()}});
$("export").addEventListener("click",()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify({version:3,exportedAt:new Date().toISOString(),messages,lessons},null,2)],{type:"application/json"}));a.download="ron-backup.json";a.click()});
$("import").addEventListener("change",async e=>{const f=e.target.files[0];if(!f)return;try{const d=JSON.parse(await f.text());if(!Array.isArray(d.messages)||!Array.isArray(d.lessons))throw 0;messages=d.messages;lessons=d.lessons;save();render();alert("تم استيراد ذاكرة رون.")}catch{alert("ملف النسخ الاحتياطي غير صالح.")}e.target.value=""});render();
})();