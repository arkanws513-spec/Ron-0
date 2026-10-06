/* Ron Learning Core
 * Local, persistent learning/experience layer.
 * It does not change model weights; it stores validated knowledge, experiences,
 * confidence and provenance so Ron can improve across conversations.
 */
(()=>{"use strict";
const KEY="ron-learning-core-v1",MAX=600;
const read=()=>{try{const v=JSON.parse(localStorage.getItem(KEY));return v&&typeof v==="object"?v:{knowledge:[],experiences:[],stats:{}}}catch{return{knowledge:[],experiences:[],stats:{}}}};
const write=v=>{try{localStorage.setItem(KEY,JSON.stringify(v))}catch{}};
const clean=s=>String(s||"").trim().replace(/\s+/g," ");
const api={
 get(){return read()},
 addKnowledge(item){const v=read(),x={...item,text:clean(item.text),at:new Date().toISOString()};if(!x.text)return false;
   const sig=clean(x.text).toLowerCase();v.knowledge=v.knowledge.filter(k=>clean(k.text).toLowerCase()!==sig);
   v.knowledge.push(x);v.knowledge=v.knowledge.slice(-MAX);write(v);return true},
 addExperience(input,output,source="conversation",confidence=.5){const v=read();const i=clean(input),o=clean(output);if(!i||!o)return false;
   v.experiences.push({input:i,output:o,source,confidence,at:new Date().toISOString()});v.experiences=v.experiences.slice(-MAX);
   v.stats.interactions=(v.stats.interactions||0)+1;write(v);return true},
 reinforce(text,delta=.05){const v=read(),q=clean(text).toLowerCase();const k=v.knowledge.find(x=>clean(x.text).toLowerCase()===q);if(k)k.confidence=Math.max(0,Math.min(1,(k.confidence??.5)+delta));write(v);return !!k},
 search(query,limit=12){const v=read(),q=clean(query).toLowerCase();if(!q)return[];
   const words=q.split(" ").filter(x=>x.length>2);return v.knowledge.map((x,i)=>{const t=clean(x.text).toLowerCase();let s=t===q?20:(t.includes(q)?10:0);for(const w of words)if(t.includes(w))s++;return{x,s,i}}).filter(z=>z.s>0).sort((a,b)=>b.s-a.s||b.i-a.i).slice(0,limit).map(z=>z.x)},
 summary(){const v=read();return{knowledge:v.knowledge.length,experiences:v.experiences.length,interactions:v.stats.interactions||0}}
};
globalThis.RonLearning=api;
})();