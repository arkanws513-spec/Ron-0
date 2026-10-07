/* Ron Agent Core — unified cognition/orchestration layer
 * Ron remains one brain: understand -> retrieve -> reason -> answer.
 * Other modules are capabilities used by this core, not separate agents.
 */
(()=>{"use strict";
const norm=s=>String(s||"").toLowerCase().replace(/[ًٌٍَُِّْـ]/g,"").replace(/[أإآ]/g,"ا").replace(/ى/g,"ي").replace(/\s+/g," ").trim();
const stop=new Set(["ما","ماذا","ماهو","ماهي","هل","هو","هي","من","في","عن","الى","إلى","هذا","هذه","ذلك","تلك","اي","ايه","اية","يا","رون","انا","ان","و","او","ال","التي","الذي","كان","كانت","مع","على"]);
const words=s=>norm(s).replace(/[؟?!.,،؛;:()\[\]{}]/g," ").split(" ").filter(w=>w.length>2&&!stop.has(w));
const overlap=(q,t)=>{const a=new Set(words(q)),b=words(t);return a.size?b.filter(w=>a.has(w)).length/a.size:0};
const memory=(q,limit=8)=>{try{return globalThis.RonLearning?.search?.(q,limit)||[]}catch{return[]}};
const arithmetic=s=>{const q=norm(s).replace(/×/g,"*").replace(/÷/g,"/").replace(/[^0-9+\-*/().%\s]/g,"");if(!q||!/[+\-*/%]/.test(q)||!/\d/.test(q))return null;try{if(!/^[0-9+\-*/().%\s]+$/.test(q))return null;const v=Function('"use strict";return ('+q+')')();if(Number.isFinite(v))return String(v)}catch{}return null};
const intent=s=>{const n=norm(s);if(/^(مرحبا|اهلا|السلام عليكم|هاي|هلا)/.test(n))return"greeting";if(/^(شكرا|عظيم|رائع|ممتاز|جميل|تمام|احسنت)/.test(n))return"social";if(/^عل[ّ]?م رون/.test(n))return"learning";if(/^(ليه|لماذا|ازاي|كيف|ماذا|ما هو|ما هي|هل|كم|اين|متى|من هو|من هي|عايز|اريد|ممكن)/.test(n)||/[؟?]/.test(s))return"question";return"statement"};
const localKnowledgeAnswer=q=>{
 const k=globalThis.RonKnowledge;
 if(k?.retrieve){const hits=k.retrieve(q,10);const best=hits[0];if(best){
   const n=norm(q),t=norm(best.text);
   let m=n.match(/^(?:ما هي|ما هو|ماهو|ماهي|ايه|اي)\s+(?:عاصمة|عاصمه)\s+(.+)$/);
   if(m){const place=m[1].trim(),z=t.match(new RegExp("^(.+?)\\s+(?:هي\\s+)?عاصمة\\s+"+place+"$"));if(z)return{answer:z[1]+" هي عاصمة "+place+".",confidence:Math.min(1,best.score||.8),source:best.source||"ron-knowledge"}}
   m=n.match(/^(?:ما هو|ما هي|ماهو|ماهي)\s+(.+)$/);
   if(m){const subject=m[1].trim(),z=t.match(/^(.+?)\s+(?:هي|هو)\s+(.+)$/);if(z&&norm(z[1])===subject)return{answer:z[2]+".",confidence:best.confidence??.8,source:best.source||"ron-knowledge"}}
   if((best.relevance??0)>=.5)return{answer:String(best.text).replace(/^سؤال:\s*[^|]+\|\s*(?:معلومة|إجابة):\s*/,""),confidence:best.confidence??.7,source:best.source||"ron-knowledge"};
 }}
 return null;
};
const reason=q=>{
 try{
  const r=globalThis.RonReasoner;
  if(r?.answer){
   const learned=memory(q,30),knowledge=globalThis.RonKnowledge?.retrieve?.(q,20)||[];
   const all=[...learned,...knowledge].map(x=>x?.text?x.text:x).filter(Boolean);
   const seen=new Set(),facts=all.filter(x=>{const n=norm(x);if(seen.has(n))return false;seen.add(n);return true});
   const result=r.answer(q,r.parseAll(facts));
   if(result?.answer)return result;
  }
 }catch{}
 return null;
};
const api={
 normalize:norm,
 classify:intent,
 plan(q){const i=intent(q),calc=arithmetic(q);return{intent:i,steps:calc?["local-arithmetic"]:i==="question"?["memory","knowledge","reasoning","web-search-if-needed"]:["local"]}},
 retrieve(q){return memory(q,12)},
 answer(q){
   const calc=arithmetic(q);if(calc!==null)return"النتيجة: "+calc;
   const reasoned=reason(q);if(reasoned?.answer)return reasoned.answer;
   const known=localKnowledgeAnswer(q);if(known?.answer)return known.answer;
   return null;
 },
 async think(q){
   const calc=arithmetic(q);if(calc!==null)return{answer:"النتيجة: "+calc,stage:"local-arithmetic",confidence:1};
   const reasoned=reason(q);if(reasoned?.answer)return{...reasoned,stage:"reasoning"};
   const known=localKnowledgeAnswer(q);if(known?.answer)return{...known,stage:"knowledge"};
   return{answer:null,stage:"needs-web-or-model",confidence:0};
 },
 explainUncertainty(){return"لا أملك ثقة كافية في هذه المعلومة بعد."}
};
globalThis.RonAgent=api;
})();