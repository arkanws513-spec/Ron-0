/* Ron Knowledge Core v1
 * Permanent knowledge/retrieval layer of Ron's brain.
 * It does not replace the language model; it decides what stored knowledge
 * is relevant, fresh, trustworthy, and worth placing in Ron's reasoning path.
 */
(()=>{"use strict";
const KEY="ron-knowledge-core-v1",MAX=5000;
const clean=s=>String(s||"").replace(/\s+/g," ").trim();
const norm=s=>clean(s).toLowerCase().replace(/[ًٌٍَُِّْـ]/g,"").replace(/[أإآ]/g,"ا").replace(/ى/g,"ي");
const stop=new Set(["ما","ماذا","ماهو","ماهي","هل","هو","هي","من","في","عن","الى","هذا","هذه","ذلك","تلك","اي","ايه","يا","رون","انا","ان","و","او","ال","التي","الذي","ما","كان","كانت","مع","على"]);
const tokens=s=>norm(s).replace(/[؟?!.,،؛;:()\[\]{}]/g," ").split(" ").filter(w=>w.length>2&&!stop.has(w));
const read=()=>{try{const x=JSON.parse(localStorage.getItem(KEY));return x&&typeof x==="object"?x:{documents:[],facts:[],stats:{}}}catch{return{documents:[],facts:[],stats:{}}}};
const write=x=>{try{localStorage.setItem(KEY,JSON.stringify(x))}catch{}};
const domain=q=>{const n=norm(q);
 if(/طب|مرض|دواء|علاج|اعراض|تشخيص|جسم|صحة/.test(n))return"medical";
 if(/دين|اسلام|مسلم|قران|حديث|فقه|عقيدة|مسيحي|يهودي/.test(n))return"religion";
 if(/فلسف|منطق|وجود|اخلاق|معرف/.test(n))return"philosophy";
 if(/برمج|كود|تقنية|تكنولوجيا|حاسوب|ذكاء اصطناعي|نموذج|api/.test(n))return"technology";
 if(/فيزياء|كيمياء|احياء|رياضيات|فلك|علم|علوم/.test(n))return"science";
 if(/تاريخ|حضارة|ثقافة|لغة|ادب/.test(n))return"culture";
 return"general";
};
const sourceScore=x=>{const s=norm(x.source||"");if(/official|government|edu|university|who|nih|nasa|arxiv/.test(s))return .25;if(/wikipedia/.test(s))return .1;return 0};
const freshness=x=>{const t=Date.parse(x.at||x.publishedAt||"");if(!Number.isFinite(t))return 0;const days=Math.max(0,(Date.now()-t)/86400000);return Math.max(0,Math.min(.15, .15*Math.exp(-days/365)))};
const relevance=(q,t)=>{const a=new Set(tokens(q)),b=tokens(t);let n=0;for(const w of b)if(a.has(w))n++;return n/(Math.max(1,a.size))};
const add=(item={})=>{const text=clean(item.text);if(!text)return false;const v=read(),sig=norm(text);v.documents=v.documents.filter(x=>norm(x.text)!==sig);v.documents.push({...item,text,domain:item.domain||domain(text),at:item.at||new Date().toISOString()});v.documents=v.documents.slice(-MAX);write(v);return true};
const retrieve=(q,limit=8)=>{const v=read();const d=domain(q);const local=globalThis.RonLearning?.get?.()?.knowledge||[];const all=[...v.documents,...local.map(x=>({...x,domain:x.domain||domain(x.text)}))];const seen=new Set();return all.map(x=>{const text=clean(x.text);const sig=norm(text);if(!text||seen.has(sig))return null;seen.add(sig);const r=relevance(q,text);const ds=x.domain===d?.12:0;return r?{...x,relevance:r,score:r+ds+sourceScore(x)+freshness(x)}:null}).filter(Boolean).sort((a,b)=>b.score-a.score).slice(0,limit)};
const stats=()=>{const v=read();return{documents:v.documents.length,domainCounts:v.documents.reduce((a,x)=>(a[x.domain||"general"]=(a[x.domain||"general"]||0)+1,a),{}),retrievals:v.stats.retrievals||0}};
const api={domain,add,retrieve,stats,rememberSearchResult:(q,r)=>{if(!r?.answer)return false;return add({text:"سؤال: "+clean(q)+" | معلومة: "+clean(r.answer),source:r.source||"web",url:r.url||"",title:r.title||"",confidence:.65})}};
globalThis.RonKnowledge=api;
const previous=globalThis.RonAgent;
if(previous){
 const oldRetrieve=previous.retrieve?.bind(previous),oldAnswer=previous.answer?.bind(previous);
 previous.retrieve=q=>api.retrieve(q,12);
 previous.answer=q=>{
   const direct=oldAnswer?oldAnswer(q):null;
   if(direct)return direct;
   const hits=api.retrieve(q,6);
   const best=hits[0];
   if(best && best.relevance>=.34){
     return clean(best.text).replace(/^سؤال:\s*[^|]+\|\s*معلومة:\s*/,"");
   }
   return null;
 };
 previous.knowledge=api;
}
})();