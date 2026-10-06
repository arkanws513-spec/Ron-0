(()=>{"use strict";
const norm=s=>String(s||"").toLowerCase().replace(/[ًٌٍَُِّْـ]/g,"").replace(/[أإآ]/g,"ا").replace(/ى/g,"ي").replace(/\s+/g," ").trim();
const clean=s=>String(s||"").replace(/^[\s.،,؛;:]+|[\s.!؟?،,؛;:]+$/g,"").trim();
const relationPatterns=[
 [/^(.+?)\s+(?:هي|هو)\s+(.+)$/,"is"],
 [/^(.+?)\s+(?:يسبب|تسبب|يؤدي الى|يؤدي الي)\s+(.+)$/,"causes"],
 [/^(.+?)\s+(?:يدعم|تدعم)\s+(.+)$/,"supports"],
 [/^(.+?)\s+(?:قبل|يسبق)\s+(.+)$/,"precedes"],
 [/^(.+?)\s+(?:بعد|يتبع)\s+(.+)$/,"follows"],
 [/^(.+?)\s+(?:يحتاج الى|يحتاج الي|يحتاج)\s+(.+)$/,"needs"]
];
const parse=s=>{const n=norm(s).replace(/[.،,؛;؟?]+$/,"").trim();for(const [p,r] of relationPatterns){const m=n.match(p);if(m)return{subject:clean(m[1]),relation:r,object:clean(m[2]),source:"local",confidence:.8,text:s}}return null};
const parseAll=items=>items.flatMap(x=>{const f=parse(x.text||x);return f?[f]:[]});
const reason=(facts)=>{
 const out=[...facts],seen=new Set(facts.map(f=>[norm(f.subject),f.relation,norm(f.object)].join("|")));
 const rules=[["causes","causes","causes"],["precedes","precedes","precedes"],["supports","supports","supports"]];
 for(let pass=0;pass<4;pass++){let added=false;for(const [a,b,c] of rules)for(const x of out)for(const y of out)if(x.relation===a&&y.relation===b&&norm(x.object)===norm(y.subject)){const z={subject:x.subject,relation:c,object:y.object,source:"inference",confidence:Math.min(x.confidence,y.confidence)*.84,text:x.subject+" "+c+" "+y.object};const k=[norm(z.subject),z.relation,norm(z.object)].join("|");if(!seen.has(k)){seen.add(k);out.push(z);added=true}}if(!added)break}
 return out;
};
const answer=(q,facts)=>{
 const n=norm(q).replace(/[؟?]+$/,"").trim();
 let m=n.match(/^(?:ما هي|ما هو|ماهو|ايه|اي)\s+(.+?)\s+(.+)$/);
 if(m){const rel={"عاصمة":"is","عاصمه":"is","يسبب":"causes","تسبب":"causes","يدعم":"supports","تدعم":"supports","قبل":"precedes","يسبق":"precedes","بعد":"follows","يتبع":"follows","يحتاج":"needs"}[m[1]];if(rel){const f=facts.filter(x=>x.relation===rel&&norm(x.object)===norm(m[2])).sort((a,b)=>b.confidence-a.confidence)[0];if(f)return{answer:f.subject+" "+m[1]+" "+f.object+".",confidence:f.confidence,source:f.source}}}
 m=n.match(/^هل\s+(.+?)\s+(يسبب|تسبب|يدعم|تدعم|قبل|يسبق|بعد|يتبع|يحتاج)\s+(.+)$/);
 if(m){const rel={"يسبب":"causes","تسبب":"causes","يدعم":"supports","تدعم":"supports","قبل":"precedes","يسبق":"precedes","بعد":"follows","يتبع":"follows","يحتاج":"needs"}[m[2]],a=norm(m[1]),b=norm(m[3]);if(facts.some(x=>x.relation===rel&&norm(x.subject)===a&&norm(x.object)===b))return{answer:"نعم، لدي دليل يدعم ذلك.",confidence:.8,source:"reasoning"}}
 return null
};
globalThis.RonReasoner={parse,parseAll,reason,answer};
})();