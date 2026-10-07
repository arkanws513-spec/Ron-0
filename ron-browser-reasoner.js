(()=>{"use strict";
const norm=s=>String(s||"").toLowerCase().replace(/[ًٌٍَُِّْـ]/g,"").replace(/[أإآ]/g,"ا").replace(/ى/g,"ي").replace(/\s+/g," ").trim();
const clean=s=>String(s||"").replace(/^[\s.،,؛;:]+|[\s.!؟?،,؛;:]+$/g,"").trim();
const patterns=[
[/^(.+?)\s+(?:هي|هو)\s+(.+)$/,"is"],[/^(.+?)\s+(?:يسبب|تسبب|يؤدي الى|يؤدي الي|يؤدي إلى)\s+(.+)$/,"causes"],
[/^(.+?)\s+(?:يدعم|تدعم)\s+(.+)$/,"supports"],[/^(.+?)\s+(?:قبل|يسبق)\s+(.+)$/,"precedes"],
[/^(.+?)\s+(?:بعد|يتبع)\s+(.+)$/,"follows"],[/^(.+?)\s+(?:يحتاج الى|يحتاج الي|يحتاج إلى|يحتاج)\s+(.+)$/,"needs"]];
const parse=s=>{const n=norm(s).replace(/[.،,؛;؟?]+$/,"").trim();for(const [p,r] of patterns){const m=n.match(p);if(m)return{subject:clean(m[1]),relation:r,object:clean(m[2]),source:"local",confidence:.8,text:s}}return null};
const parseAll=items=>items.flatMap(x=>{const f=parse(x.text||x);return f?[f]:[]});
const reason=facts=>facts;
const answer=(q,facts)=>{
 const n=norm(q).replace(/[؟?]+$/,"").trim();
 let m=n.match(/^(?:ما هي|ما هو|ماهو|ايه|اي)\s+(.+?)\s+(.+)$/);
 if(m){const rel={"عاصمة":"is","عاصمه":"is","يسبب":"causes","تسبب":"causes","يدعم":"supports","تدعم":"supports","قبل":"precedes","يسبق":"precedes","بعد":"follows","يتبع":"follows","يحتاج":"needs"}[m[1]];
  if(rel){const f=facts.filter(x=>x.relation===rel&&norm(x.object)===norm(m[2])).sort((a,b)=>b.confidence-a.confidence)[0];if(f)return{answer:f.subject+" "+m[1]+" "+f.object+".",confidence:f.confidence,source:f.source}}
 }
 m=n.match(/^هل\s+(.+?)\s+(يسبب|تسبب|يدعم|تدعم|قبل|يسبق|بعد|يتبع|يحتاج)\s+(.+)$/);
 if(m){const rel={"يسبب":"causes","تسبب":"causes","يدعم":"supports","تدعم":"supports","قبل":"precedes","يسبق":"precedes","بعد":"follows","يتبع":"follows","يحتاج":"needs"}[m[2]];
  if(facts.some(x=>x.relation===rel&&norm(x.subject)===norm(m[1])&&norm(x.object)===norm(m[3])))return{answer:"نعم، لدي معلومة محلية تدعم ذلك.",confidence:.8,source:"ron-core"}}
 return null;
};
globalThis.RonReasoner={parse,parseAll,reason,answer};
})();