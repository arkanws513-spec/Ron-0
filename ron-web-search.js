/* Ron Web Search — retrieve answer-bearing facts, not whole pages */
(()=>{"use strict";
const clean=s=>String(s||"").replace(/\s+/g," ").trim();
const norm=s=>clean(s).toLowerCase().replace(/[ًٌٍَُِّْـ]/g,"").replace(/[أإآ]/g,"ا").replace(/ى/g,"ي");
const enc=s=>encodeURIComponent(clean(s));
const timeout=async(url,ms=9000)=>{const c=new AbortController(),t=setTimeout(()=>c.abort(),ms);try{const r=await fetch(url,{signal:c.signal,headers:{Accept:"application/json"}});if(!r.ok)throw new Error("HTTP "+r.status);return await r.json()}finally{clearTimeout(t)}};
const sentences=s=>clean(s).split(/(?<=[.!؟?؛])\s+|(?<=،)\s+(?=[^،]{20,})/).map(clean).filter(x=>x.length>15);
const queryWords=q=>norm(q).replace(/[؟?!.,،؛;:()\[\]{}]/g," ").split(" ").filter(w=>w.length>2&&!new Set(["ما","ماذا","ماهو","ماهي","هل","هو","هي","من","في","عن","الى","هذا","هذه","ذلك","تلك","اي","ايه","اين","متى","كيف","لماذا","عاصمة","اسم","اخبار"]).has(w));
const scoreSentence=(q,s)=>{const ws=queryWords(q),t=norm(s);let score=0;for(const w of ws)if(t.includes(w))score+=2;return score+(s.length<=220?1:0)};
const concise=(q,text)=>{
 const ss=sentences(text);if(!ss.length)return clean(text).slice(0,420);
 const ranked=ss.map((s,i)=>({s,score:scoreSentence(q,s),i})).sort((a,b)=>b.score-a.score||a.s.length-b.s.length||a.i-b.i);
 const useful=ranked.filter(x=>x.score>0).slice(0,2).sort((a,b)=>a.i-b.i).map(x=>x.s);
 return (useful.length?useful:ranked.slice(0,2).map(x=>x.s)).join(" ").slice(0,520);
};
const wiki=async(q,lang)=>{const url="https://"+lang+".wikipedia.org/w/api.php?action=query&list=search&srsearch="+enc(q)+"&format=json&origin=*&utf8=1&srlimit=5";const d=await timeout(url);const hits=d?.query?.search||[];if(!hits.length)return[];const ids=hits.map(x=>x.pageid).join("|");const eurl="https://"+lang+".wikipedia.org/w/api.php?action=query&pageids="+ids+"&prop=extracts|info&exintro=1&explaintext=1&inprop=url&format=json&origin=*";const e=await timeout(eurl);return Object.values(e?.query?.pages||{}).map(x=>({title:x.title,extract:clean(x.extract),url:x.fullurl||("https://"+lang+".wikipedia.org/wiki/"+encodeURIComponent(x.title.replace(/ /g,"_"))),source:"Wikipedia"})).filter(x=>x.extract)};
const ddg=async q=>{const d=await timeout("https://api.duckduckgo.com/?q="+enc(q)+"&format=json&no_html=1&skip_disambig=1");const out=[];if(d?.AbstractText)out.push({title:d.Heading||q,extract:clean(d.AbstractText),url:d.AbstractURL||"https://duckduckgo.com/?q="+enc(q),source:"DuckDuckGo"});for(const x of (d?.RelatedTopics||[]).slice(0,5))if(x?.Text)out.push({title:x.Text.split(" - ")[0]||q,extract:clean(x.Text),url:x.FirstURL||"https://duckduckgo.com/?q="+enc(q),source:"DuckDuckGo"});return out};
const answer=async query=>{
 const q=clean(query);if(!q)return null;
 // For capital questions, search for the country itself and prefer extracts that explicitly identify its capital.
 const cap=q.match(/^(?:ما\s+(?:هي|هو)|ماهي|ماهو|ايه|اي)\s+(?:عاصمة|عاصمه)\s+(.+?)[؟?]?$/i);
 const searchQ=cap?clean(cap[1]):q;
 const settled=await Promise.allSettled([wiki(searchQ,"ar"),wiki(searchQ,"en"),ddg(searchQ)]);
 const results=settled.flatMap(x=>x.status==="fulfilled"?x.value:[]);
 const unique=[],seen=new Set();
 for(const r of results){const key=(r.url||r.title).toLowerCase();if(!seen.has(key)){seen.add(key);unique.push(r)}}
 if(!unique.length)return null;
 const ranked=unique.map((r,i)=>{const answer=concise(q,r.extract);let score=scoreSentence(q,r.extract)+(r.source==="Wikipedia"?1:0)-i*.01;if(cap&&/(عاصمة|capital)\s*(?:هي|is)?\s*|عاصمة\s+.+\s+هي|capital\s+of/i.test(r.extract))score+=5;return {...r,answer,score};}).filter(r=>r.answer);
 ranked.sort((a,b)=>b.score-a.score||a.answer.length-b.answer.length);
 const best=ranked[0];
 return best?{answer:best.answer,source:best.source,title:best.title,url:best.url,results:ranked.slice(0,8).map(({answer,source,title,url})=>({answer,source,title,url}))}:null;
};
const CACHE_KEY="ron-web-cache-v3";
const remember=(query,result)=>{
 try{
  if(!result?.answer)return false;
  const a=JSON.parse(localStorage.getItem(CACHE_KEY)||"[]"),q=norm(query),next=a.filter(x=>norm(x.query)!==q);
  const item={query:clean(query),answer:clean(result.answer),source:clean(result.source||"web"),title:clean(result.title||""),url:result.url||"",at:new Date().toISOString()};
  next.push(item);localStorage.setItem(CACHE_KEY,JSON.stringify(next.slice(-200)));
  globalThis.RonKnowledge?.rememberSearchResult?.(query,{answer:item.answer,source:item.source,title:item.title,url:item.url});
  globalThis.RonLearning?.addKnowledge?.({text:"سؤال: "+clean(query)+" | معلومة: "+item.answer,kind:"web-fact",source:item.source,confidence:.65,url:item.url,title:item.title});
  return true;
 }catch{return false}
};
const cached=query=>{try{const q=norm(query),a=JSON.parse(localStorage.getItem(CACHE_KEY)||"[]");return a.slice().reverse().find(v=>norm(v.query)===q)||null}catch{return null}};
globalThis.RonWebSearch={answer,remember,cached};
})();