/* Ron Browser Reasoner — one local inference engine
 * Deterministic, explainable, browser-safe reasoning.
 */
(()=>{"use strict";
const norm=s=>String(s||"").toLowerCase().replace(/[ًٌٍَُِّْـ]/g,"").replace(/[أإآ]/g,"ا").replace(/ى/g,"ي").replace(/إلى/g,"الى").replace(/\s+/g," ").trim();
const clean=s=>String(s||"").replace(/^[\s.،,؛;:]+|[\s.!؟?،,؛;:]+$/g,"").replace(/\s+/g," ").trim();
const inverse={precedes:"follows",follows:"precedes"};
const patterns=[
 [/^(.+?)\s+(هي|هو)\s+(.+)$/,"is"],
 [/^(.+?)\s+(يسبب|تسبب|يؤدي الى|يؤدي الي)\s+(.+)$/,"causes"],
 [/^(.+?)\s+(يدعم|تدعم)\s+(.+)$/,"supports"],
 [/^(.+?)\s+(قبل|يسبق)\s+(.+)$/,"precedes"],
 [/^(.+?)\s+(بعد|يتبع)\s+(.+)$/,"follows"],
 [/^(.+?)\s+(يحتاج|يحتاج الى|يحتاج الي)\s+(.+)$/,"needs"],
 [/^(.+?)\s+(جزء من|ضمن|ينتمي الى)\s+(.+)$/,"part_of"]
];
const parse=s=>{const original=clean(s),n=norm(original);
 let m=n.match(/^عاصمة\s+(.+?)\s+(?:هي|هو)\s+(.+)$/);if(m)return{subject:clean(m[2]),relation:"is",object:clean(m[1]),surface:"هي عاصمة",source:"local",confidence:.9,text:original,derived:false};
 m=n.match(/^(.+?)\s+(?:هي|هو)\s+عاصمة\s+(.+)$/);if(m)return{subject:clean(m[1]),relation:"is",object:clean(m[2]),surface:"هي عاصمة",source:"local",confidence:.9,text:original,derived:false};
 for(const [p,r] of patterns){m=n.match(p);if(m)return{subject:clean(m[1]),relation:r,object:clean(m[3]),surface:m[2],source:"local",confidence:.8,text:original,derived:false}}return null};
const parseAll=items=>items.flatMap(x=>{const f=parse(x?.text||x);return f?[f]:[]});
const key=f=>norm(f.subject)+"|"+f.relation+"|"+norm(f.object);
const derive=facts=>{
 const out=facts.map(x=>({...x})),seen=new Set(out.map(key));
 const add=(s,r,o,base,rule)=>{const f={subject:s,relation:r,object:o,source:base.source||"ron-core",confidence:Math.max(.5,(base.confidence??.8)*.92),derived:true,rule};const k=key(f);if(!seen.has(k)){seen.add(k);out.push(f);return true}return false};
 for(let round=0;round<4;round++){
  let changed=false;
  for(const f of [...out])if(inverse[f.relation])changed=add(f.object,inverse[f.relation],f.subject,f,"inverse")||changed;
  for(const rel of ["precedes","follows","causes","supports","needs","part_of"]){
   const same=out.filter(x=>x.relation===rel);
   for(const a of same)for(const b of same)if(norm(a.object)===norm(b.subject)&&norm(a.subject)!==norm(b.object))changed=add(a.subject,rel,b.object,a,"chain")||changed;
  }
  if(!changed)break;
 }
 return out;
};
const relMap={"يسبب":"causes","تسبب":"causes","يدعم":"supports","تدعم":"supports","قبل":"precedes","يسبق":"precedes","بعد":"follows","يتبع":"follows","يحتاج":"needs","يحتاج الى":"needs","جزء من":"part_of","ضمن":"part_of"};
const question=q=>{
 const n=norm(q).replace(/[؟?]+$/,"").trim();let m=n.match(/^(?:ما هي|ما هو|ماهي|ماهو|ايه|اي)\s+(?:عاصمة|عاصمه)\s+(.+)$/);
 if(m)return{type:"capital",relation:"is",object:clean(m[1])};
 m=n.match(/^(?:ما هي|ما هو|ماهي|ماهو|ايه|اي)\s+(.+?)\s+(يسبب|تسبب|يدعم|تدعم|قبل|يسبق|بعد|يتبع|يحتاج|يحتاج الى|جزء من|ضمن)\s+(.+)$/);
 if(m)return{type:"relation",relation:relMap[m[2]],subject:clean(m[1]),object:clean(m[3]),word:m[2]};
 m=n.match(/^هل\s+(.+?)\s+(يسبب|تسبب|يدعم|تدعم|قبل|يسبق|بعد|يتبع|يحتاج|يحتاج الى|جزء من|ضمن)\s+(.+)$/);
 if(m)return{type:"yesno",relation:relMap[m[2]],subject:clean(m[1]),object:clean(m[3])};
 m=n.match(/^(?:لماذا|ليه|لما)\s+(.+)$/);if(m)return{type:"why",subject:clean(m[1])};
 return{type:"unknown"};
};
const answer=(q,rawFacts)=>{
 const facts=derive(rawFacts||[]),ask=question(q);
 if(ask.type==="capital"){const f=facts.find(x=>x.relation==="is"&&norm(x.object)===norm(ask.object));if(f)return{answer:f.subject+" هي عاصمة "+f.object+".",confidence:f.confidence,source:f.source,trace:f}}
 if(ask.type==="relation"){const f=facts.find(x=>x.relation===ask.relation&&norm(x.subject)===norm(ask.subject)&&norm(x.object)===norm(ask.object));if(f)return{answer:f.subject+" "+ask.word+" "+f.object+".",confidence:f.confidence,source:f.source,trace:f}}
 if(ask.type==="yesno"){const f=facts.find(x=>x.relation===ask.relation&&norm(x.subject)===norm(ask.subject)&&norm(x.object)===norm(ask.object));if(f)return{answer:"نعم، لدي أساس محلي لهذا الاستنتاج.",confidence:f.confidence,source:f.source,trace:f}}
 if(ask.type==="why"){const f=facts.find(x=>x.relation==="causes"&&norm(x.object)===norm(ask.subject));if(f)return{answer:"لأن "+f.subject+" يسبب "+f.object+".",confidence:f.confidence,source:f.source,trace:f}}
 const n=norm(q);if(/^(?:ما هو|ما هي|ماهو|ماهي)\s+/.test(n)){const subject=clean(n.replace(/^(?:ما هو|ما هي|ماهو|ماهي)\s+/,""));const f=facts.find(x=>x.relation==="is"&&norm(x.subject)===norm(subject));if(f)return{answer:f.object+".",confidence:f.confidence,source:f.source,trace:f}}
 return null;
};
globalThis.RonReasoner={normalize:norm,parse,parseAll,reason:derive,derive,answer};
})();