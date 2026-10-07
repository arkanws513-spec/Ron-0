/* Ron Agent Core — orchestration layer
 * Independent control layer: intent, memory retrieval, deterministic reasoning,
 * uncertainty handling and safe routing between local knowledge, web search and
 * deterministic local reasoning. Ron remains one core.
 */
(()=>{"use strict";
const norm=s=>String(s||"").toLowerCase().replace(/[ًٌٍَُِّْـ]/g,"").replace(/[أإآ]/g,"ا").replace(/ى/g,"ي").replace(/\s+/g," ").trim();
const stop=new Set(["ما","ماذا","ماهو","ماهي","ما","هل","هو","هي","من","في","عن","الى","إلى","هذا","هذه","ذلك","تلك","اي","ايه","اية","يا","رون","انا","ان","و","او","ال","التي","الذي"]);
const words=s=>norm(s).replace(/[؟?!.,،؛;:()\[\]{}]/g," ").split(" ").filter(w=>w.length>2&&!stop.has(w));
const score=(q,t)=>{const a=words(q),b=new Set(words(t));return a.reduce((n,w)=>n+(b.has(w)?1:0),0)};
const memory=(q,limit=8)=>{try{return globalThis.RonLearning?.search?.(q,limit)||[]}catch{return[]}};
const arithmetic=s=>{const q=norm(s).replace(/×/g,"*").replace(/÷/g,"/").replace(/[^0-9+\-*/().%\s]/g,"");if(!q||!/[+\-*/%]/.test(q)||!/\d/.test(q))return null;try{if(!/^[0-9+\-*/().%\s]+$/.test(q))return null;const v=Function('"use strict";return ('+q+')')();if(Number.isFinite(v))return String(v)}catch{}return null};
const localAnswer=q=>{
 const hits=memory(q,10).filter(x=>x?.text);
 if(!hits.length)return null;
 const ranked=hits.map((x,i)=>({x,s:score(q,x.text)+(x.confidence||0)*.2-i*.001})).sort((a,b)=>b.s-a.s);
 const best=ranked[0]; if(!best||best.s<1.1)return null;
 const text=String(best.x.text);
 const n=norm(q);
 let m=n.match(/^(?:ما هي|ما هو|ماهو|ايه|اي)\s+عاصمة\s+(.+)$/);
 if(m){
   const place=m[1].trim(), c=norm(text);
   let z=c.match(new RegExp("^(.+?)\\s+(?:هي\\s+)?عاصمة\\s+"+place+"$"));
   if(z)return z[1]+" هي عاصمة "+place+".";
 }
 if(/^(?:ما هي|ما هو|ماهو)\s+/.test(n)){
   const subject=n.replace(/^(?:ما هي|ما هو|ماهو)\s+/,"").trim();
   const z=norm(text).match(/^(.+?)\s+(?:هي|هو)\s+(.+)$/);
   if(z&&norm(z[1])===subject)return z[2]+".";
 }
 return null;
};
const api={
 normalize:norm,
 classify(q){
   const n=norm(q);
   if(/^(مرحبا|اهلا|السلام عليكم|هاي|هلا)/.test(n))return"greeting";
   if(/^(شكرا|عظيم|رائع|ممتاز|جميل|تمام|احسنت)/.test(n))return"social";
   if(/^عل[ّ]?م رون/.test(n))return"learning";
   if(/^(ليه|لماذا|ازاي|كيف|ماذا|ما هو|ما هي|هل|كم|اين|متى|من هو|من هي|عايز|اريد|ممكن)/.test(n)||/[؟?]/.test(q))return"question";
   return"statement";
 },
 plan(q){
   const intent=this.classify(q), calc=arithmetic(q);
   return {intent,steps:calc?["local-arithmetic"]:intent==="question"?["local-memory","local-reasoning","web-search"]:["local"]};
 },
 answer(q){
   const calc=arithmetic(q); if(calc!==null)return"النتيجة: "+calc;
   return localAnswer(q);
 },
 retrieve(q){return memory(q,12)},
 explainUncertainty(){return"لا أملك ثقة كافية في هذه المعلومة بعد."}
};
globalThis.RonAgent=api;
})();