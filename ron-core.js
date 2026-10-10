'use strict';

/* Ron Core v2: one brain, memory + reasoning + web orchestration. Node 18+, zero dependencies. */
const fs=typeof require==='function'?require('fs'):null;

const DIACRITICS=/[\u064B-\u065F\u0670\u0640]/g;
const MAP={'أ':'ا','إ':'ا','آ':'ا','ى':'ي','ة':'ه'};
const PUNCT=/[.!?؟,،;؛:"«»()\[\]{}]/g;
const clean=s=>String(s??'').replace(DIACRITICS,'').replace(/\s+/g,' ').trim();
const normalize=s=>clean(s).replace(PUNCT,' ').replace(/[٠-٩]/g,c=>String('٠١٢٣٤٥٦٧٨٩'.indexOf(c))).replace(/[\s\S]/g,c=>MAP[c]||c.toLowerCase()).replace(/\bت\s+علم\b/g,'تعلم').replace(/\s+/g,' ').trim();
const key=normalize;
const STOP=new Set('ما ماذا ماهو ماهي هل هو هي من في عن الى هذا هذه ذلك تلك اي ايه يا رون انا انت ان و او ال التي الذي كان كانت مع على ثم قد لقد لم لن'.split(' '));
const CONTEXT_WORDS=new Set('فوق قبل سابق السابق السابقه سابقا ذلك ذلكك هذه هذا عنه عنها فيه فيها منهم منه بها به قول كلام رسالة سؤال اجابة'.split(' '));
const normalizeContextText=s=>normalize(s).replace(/\s+/g,' ').trim();
const words=s=>normalize(s).split(' ').filter(x=>x&&x.length>1&&!STOP.has(x));
const overlap=(a,b)=>{const A=new Set(words(a));if(!A.size)return 0;let n=0;for(const x of words(b))if(A.has(x))n++;return n/A.size;};

const DEFAULTS={
 selfName:'رون', personal:['اسم','عمر','لقب','عمل'], valueTokens:{default:4,'اسم':2},
 bareNames:true, searchTimeoutMs:8000, autoSearch:true, minSearchConfidence:.68,
 maxAnswerChars:1400
};
const STATE=new Set('بخير جيد تمام طيب جائع تعبان سعيد حزين متعب مشغول افكر افكرُ افكر فيه'.split(' '));
const QSTART=/^(?:ما|ماذا|ماهو|ماهي|من|كيف|هل|اين|متى|كم|لماذا|ليه|ازاي)(?:\s|$)/;

function createRuleNLU(cfg=DEFAULTS){
 cfg={...DEFAULTS,...cfg}; const pers=cfg.personal.map(key).join('|');
 const fillers=new Set(['يا','طيب','اذا','ان','تعلم','اعلم','تذكر','اوكي','حسنا',key(cfg.selfName)]);
 const parse=text=>{
  const c=clean(text),n0=normalize(c); let n=n0,t=c;
  // Direct conversational forms must be recognized before filler removal.
  if(/^(?:انت\s+)?(?:ما\s+هو|ماهو|ماهي|ما)\s+اسمك$/.test(n0)||/^(?:وانت\s+)?(?:ما\s+هو|ماهو|ماهي|ما)\s+اسمك$/.test(n0))return[{type:'ask',s:'$self',p:'اسم',sd:'',pd:'اسم',text:c}];
  if(/^(?:انت\s+)?(?:ما\s+هو|ماهو|ماهي|ما)\s+اسمي$/.test(n0))return[{type:'ask',s:'$user',p:'اسم',sd:'',pd:'اسم',text:c}];
  while(true){const m=/^(\S+)\s+/.exec(n);if(!m||!fillers.has(m[1]))break;n=n.slice(m[0].length);t=t.slice(m[0].length);}
  if(!n)return[{type:'unknown',text:c}];
  if(/^(رون|يا\s+رون|رون\s*[!،,.؟?]*)$/.test(n0))return[{type:'smalltalk',kind:'call'}];
  if(/^(?:انظر|بص|شوف)\s+(?:لما|ما)\s+(?:قلته|قلت\ه|قولته)\s+(?:فوق|قبل)$/.test(n0)||/^(?:ماذا|ما)\s+(?:قلت|قلته)\s+(?:فوق|قبل)$/.test(n0))return[{type:'context',kind:'previous'}];
  if(/^(?:ماذا|ما)\s+(?:تعلم|تعلمت|تعلمه|تعرفه)(?:\s+حتى\s+الان|\s+لحد\s+دلوقتي)?$/.test(n0)||/^هل\s+تعلمت\s+(?:ذلك|كل\s+ذلك|هذا)$/.test(n0))return[{type:'context',kind:'learned'}];
  if(/^(?:كيف\s+(?:حالك|الحال)|كيفك|شلونك|شخبارك)(?:\s+\S+)?$/.test(n))return[{type:'smalltalk',kind:'howareyou'}];
  if(/^(?:مرحبا|اهلا|هلا|سلام|السلام\s+عليكم|صباح\s+الخير|مساء\s+الخير)(?:\s+\S+)?$/.test(n))return[{type:'smalltalk',kind:'greet'}];
  let m=/^(?:ابحث|دور|فتش)(?:\s+لي)?(?:\s+عن(ها|هم|ه)?(?:\s+(.+))?)?$/.exec(n);
  if(m)return[{type:'search',anaphor:!!m[1],query:m[2]?m[2]:null}];
  if(QSTART.test(n)){
   // Resolve capital-city questions before the generic subject/property parser.
   // normalize() maps Arabic taa marbuta to haa, so match "عاصمه" here.
   const capitalQuestion=/^(?:ما|ماذا)\s+(?:هي\s+)?عاصمه\s+(.+)$/.exec(n);
   if(capitalQuestion)return[{type:'ask',s:key(capitalQuestion[1]),p:key('عاصمة'),sd:capitalQuestion[1],pd:'عاصمة',text:c}];
   if(/^من\s+انا$/.test(n))return[{type:'ask',s:'$user',p:'اسم',sd:'',pd:'اسم',text:c}];
   if(/^من\s+انت$/.test(n))return[{type:'ask',s:'$self',p:'اسم',sd:'',pd:'اسم',text:c}];
   m=/^(?:ما|ماذا|ماهو|ماهي)\s+(?:هو\s+|هي\s+)?(\S+?)(ي|ك)$/.exec(n);
   if(m&&cfg.personal.map(key).includes(m[1]))return[{type:'ask',s:m[2]==='ي'?'$user':'$self',p:m[1],sd:'',pd:m[1],text:c}];
   m=/^(?:ما|ماذا|ماهو|ماهي)\s+(?:هو\s+|هي\s+)?(\S+)\s+(.+)$/.exec(n);
   if(m)return[{type:'ask',s:key(m[2]),p:m[1],sd:m[2],pd:m[1],text:c}];
   return[{type:'unknown',text:c,question:true}];
  }
  const frames=[];
  const re=new RegExp('(?:^|\\s)و?('+pers+')(ي|ك)(?:\\s+(?:هو|هي|انت|انا))?\\s+','g');
  let hit;
  while((hit=re.exec(n))){
   const valueStart=hit.index+hit[0].length;
   const rest=n.slice(valueStart);
   const next=new RegExp('\\s+و?(?:'+pers+')(?:ي|ك)(?:\\s|$)');
   const stop=next.exec(rest);
   const raw=(stop?rest.slice(0,stop.index):rest).trim();
   const max=cfg.valueTokens[hit[1]] ?? cfg.valueTokens.default;
   const parts=raw.split(/\\s+/).slice(0,max);
   const o=key(parts.join(' '));
   if(o&&!['انت','انا','هو','هي'].includes(o))frames.push({type:'assert',s:hit[2]==='ي'?'$user':'$self',p:hit[1],o,sd:'',pd:hit[1],od:parts.join(' ')});
   if(stop)re.lastIndex=valueStart+stop.index;
  }
  if(frames.length)return frames;
  m=/^(\S+)\s+(.+?)\s+(?:هي|هو)\s+(.+)$/.exec(n);
  if(m)return[{type:'assert',s:key(m[2]),p:m[1],o:key(m[3]),sd:m[2],pd:m[1],od:m[3]}];
  m=/^(انا|انت)\s+(\S+)$/.exec(n);
  if(cfg.bareNames&&m&&!STATE.has(m[2])&&m[2].length>=2)return[{type:'assert',s:m[1]==='انا'?'$user':'$self',p:'اسم',o:key(m[2]),sd:'',pd:'اسم',od:m[2]}];
  return[{type:'unknown',text:c}];
 };
 return{parse};
}

class MemoryAdapter{load(){return null;}save(){}}
class JsonFileAdapter{
 constructor(path){this.path=path;}
 load(){try{return fs?JSON.parse(fs.readFileSync(this.path,'utf8')):null;}catch{return null;}}
 save(data){if(!fs)return;fs.mkdirSync(require('path').dirname(this.path),{recursive:true});const tmp=this.path+'.tmp';fs.writeFileSync(tmp,JSON.stringify(data,null,2));fs.renameSync(tmp,this.path);}
}
class LocalStorageAdapter{
 constructor(k='ron-core-data-v2'){this.k=k;}
 load(){try{return JSON.parse(localStorage.getItem(this.k)||'null');}catch{return null;}}
 save(data){try{localStorage.setItem(this.k,JSON.stringify(data));}catch{}}
}

const RANK={search:0,seed:1,model:1,user:2,official:3};
class FactStore{
 constructor(adapter=new MemoryAdapter(),{multi=[]}={}){this.adapter=adapter;this.multi=new Set(multi.map(key));const d=adapter.load()||{};this.facts=d.facts||[];this.disp=d.disp||{};this.unparsed=d.unparsed||[];this.events=d.events||[];}
 display(k){return k==='$user'?'أنت':k==='$self'?'أنا':this.disp[k]??k;}
 get(s,p){const subject=s[0]==='
 set(s,p,o,{source='user',confidence=1,sd,pd,od,persist=true}={}){
  const i=this.facts.findIndex(f=>f.s===s&&f.p===p&&(!this.multi.has(p)||f.o===o));let status,prev;
  if(i<0){this.facts.push({s,p,o,source,confidence,ts:Date.now(),history:[]});status='added';}
  else{const f=this.facts[i];prev=f.o;if(f.o===o)status='unchanged';else if((RANK[source]??0)>=(RANK[f.source]??0)){f.history.push({o:f.o,source:f.source,confidence:f.confidence,ts:f.ts});Object.assign(f,{o,source,confidence,ts:Date.now()});status='replaced';}else status='kept';}
  if(status==='added'||status==='replaced'){if(sd&&s[0]!=='$')this.disp[s]=sd;if(pd)this.disp[p]=pd;if(od)this.disp[o]=od;}
  if(persist)this.save();return{status,prev};
 }
 logUnparsed(text){this.unparsed.push({text,ts:Date.now()});if(this.unparsed.length>1000)this.unparsed.shift();this.save();}
 event(type,data={}){this.events.push({type,ts:Date.now(),...data});if(this.events.length>2000)this.events.shift();this.save();}
 save(){this.adapter.save({facts:this.facts,disp:this.disp,unparsed:this.unparsed,events:this.events});}
}

const DEFAULT_RULES=[
 {name:'transitive:جزء_من',if:[['?a','جزء_من','?b'],['?b','جزء_من','?c']],then:['?a','جزء_من','?c']},
 {name:'symmetric:متزوج_من',if:[['?a','متزوج_من','?b']],then:['?b','متزوج_من','?a']}
];
const isVar=x=>typeof x==='string'&&x[0]==='?';
function unify(p,t,b){const o={...b};for(let i=0;i<3;i++){if(isVar(p[i])){if(p[i] in o&&o[p[i]]!==t[i])return null;o[p[i]]=t[i];}else if(p[i]!==t[i])return null;}return o;}
function* solve(ps,ts,b={}){if(!ps.length){yield b;return;}const [p,...r]=ps;for(const t of ts){const x=unify(p,t,b);if(x)yield* solve(r,ts,x);}}
class Reasoner{
 constructor(store,rules=DEFAULT_RULES,maxRounds=8){this.store=store;this.rules=rules;this.maxRounds=maxRounds;}
 infer(){const k=new Map(),id=t=>t.join('\u0001');for(const f of this.store.facts)k.set(id([f.s,f.p,f.o]),{t:[f.s,f.p,f.o],via:f.source});for(let r=0;r<this.maxRounds;r++){let ch=false,ts=[...k.values()].map(x=>x.t);for(const rule of this.rules)for(const b of solve(rule.if,ts)){const t=rule.then.map(x=>isVar(x)?b[x]:x);if(t.every(x=>x!=null)&&!k.has(id(t))){k.set(id(t),{t,via:rule.name});ch=true;}}if(!ch)break;}return k;}
 ask(s,p){const d=this.store.get(s,p);if(d)return{o:d.o,via:d.source,confidence:d.confidence};for(const x of this.infer().values())if(x.t[0]===s&&x.t[1]===p)return{o:x.t[2],via:x.via,confidence:.7};return null;}
}

function arithmetic(text){
 let s=normalize(text).replace(/×/g,'*').replace(/÷/g,'/').replace(/[^0-9+\-*/().%\s]/g,'').trim();
 if(!s||!/[+\-*/%]/.test(s)||!/\d/.test(s))return null;
 const ts=s.match(/\d+(?:\.\d+)?|[()+\-*/%]/g)||[];if(ts.join('')!==s.replace(/\s+/g,''))return null;let i=0;
 const primary=()=>{if(ts[i]==='('){i++;const v=expr();if(ts[i]!==')')throw 0;i++;return v;}if(ts[i]==='-'){i++;return-primary();}if(!/^\d/.test(ts[i]||''))throw 0;return Number(ts[i++]);};
 const term=()=>{let v=primary();while(['*','/','%'].includes(ts[i])){const op=ts[i++],b=primary();if(op==='*')v*=b;else if(op==='/'){if(b===0)throw 0;v/=b;}else v%=b;}return v;};
 const expr=()=>{let v=term();while(['+','-'].includes(ts[i])){const op=ts[i++],b=term();v=op==='+'?v+b:v-b;}return v;};
 try{const v=expr();return i===ts.length&&Number.isFinite(v)?String(v):null;}catch{return null;}
}

function evaluateResult(r,q){
 if(!r?.answer)return null;const answer=clean(r.answer).slice(0,1800),source=clean(r.source||'web');
 const st=normalize(source+' '+(r.url||''));let sc=.45;if(/official|gov|government|edu|university|who|nih|nasa|arxiv/.test(st))sc=.9;else if(/wikipedia/.test(st))sc=.68;else if(/duckduckgo/.test(st))sc=.5;
 const rel=overlap(q,answer);return{...r,answer,source,confidence:Math.max(r.confidence??0,sc),relevance:rel,score:rel*.65+sc*.35};
}

const SEED=[['مصر','عاصمة','القاهرة'],['السعودية','عاصمة','الرياض'],['الإمارات','عاصمة','أبوظبي'],['الأردن','عاصمة','عمان'],['العراق','عاصمة','بغداد']];

class RonCore{
 constructor(opts={}){
  this.cfg={...DEFAULTS,...opts};this.store=new FactStore(opts.adapter||new MemoryAdapter(),{multi:opts.multi||[]});this.reasoner=new Reasoner(this.store,opts.rules||DEFAULT_RULES,opts.maxReasoningRounds||8);this.nlu=opts.nlu||createRuleNLU(this.cfg);this.searchTool=opts.searchTool||null;this.answerModel=opts.answerModel||null;this.ctx={pending:null,history:[]};
  this.store.set('$self','اسم',key(this.cfg.selfName),{source:'seed',confidence:1,pd:'اسم',od:this.cfg.selfName,persist:false});
  if(opts.seed!==false)for(const x of SEED)this.store.set(key(x[0]),key(x[1]),key(x[2]),{source:'seed',confidence:.9,sd:x[0],pd:x[1],od:x[2],persist:false});
  this.store.save();
 }
 async handle(text){const input=clean(text);if(!input)return{reply:'اكتب رسالة أولًا.',frames:[]};this.ctx.history.push({role:'user',text:input,ts:Date.now()});if(this.ctx.history.length>30)this.ctx.history.shift();const frames=this.nlu.parse(input),out=[];for(const f of frames)out.push(await this.exec(f));const reply=out.filter(Boolean).join('\n');this.ctx.history.push({role:'assistant',text:reply,ts:Date.now()});
  globalThis.RonLearning?.addExperience?.(input,reply,'conversation',.65);
  return{reply,frames};}
 phrase(s,p,o){const pd=this.store.display(p),od=this.store.display(o);if(s==='$user')return pd+'ك هو '+od;if(s==='$self')return pd+'ي هو '+od;return pd+' '+this.store.display(s)+' '+(/[هة]$/.test(pd)?'هي':'هو')+' '+od;}
 async exec(f){
  if(f.type==='smalltalk')return f.kind==='howareyou'?'أنا بخير وجاهز للعمل. ماذا تريد أن نفعل؟':f.kind==='call'?'نعم، أنا معك.':'مرحبًا، كيف أساعدك؟';
  if(f.type==='context'){
   if(f.kind==='previous'){
    const h=this.ctx.history.filter(x=>x.role==='user');
    const prev=h.length>1?h[h.length-2]:null;
    return prev?`آخر شيء قلته قبل رسالتك الحالية كان: «${prev.text}»`:'لا توجد رسالة سابقة أستطيع الرجوع إليها بعد.';
   }
   const facts=this.store.facts.filter(x=>x.source!=='seed');
   const learned=[];
   for(const fct of facts)learned.push(this.phrase(fct.s,fct.p,fct.o));
   const knowledge=globalThis.RonLearning?.get?.()?.knowledge||[];
   for(const x of knowledge.slice(-20))if(x?.text)learned.push(clean(x.text));
   const unique=[...new Set(learned)].slice(-12);
   return unique.length?'نعم، لدي معرفة محفوظة من محادثاتك وتعليماتك. من أمثلتها:\n'+unique.map((x,i)=>`${i+1}. ${x}`).join('\n'):'لم أتعلم معلومات شخصية أو تعليمات جديدة بعد.';
  }
  if(f.type==='assert'){const r=this.store.set(f.s,f.p,f.o,{sd:f.sd,pd:f.pd,od:f.od});globalThis.RonLearning?.addKnowledge?.({text:this.phrase(f.s,f.p,f.o),kind:'fact',source:'conversation',confidence:1});const line=this.phrase(f.s,f.p,f.o);if(r.status==='unchanged')return'أعرف ذلك بالفعل: '+line+'.';if(r.status==='replaced')return'تم، حدّثتها: '+line+' (كانت: '+this.store.display(r.prev)+').';return'تم. '+line+'.';}
  if(f.type==='ask'){
   const calc=arithmetic(f.text);if(calc!==null)return'النتيجة: '+calc;
   const local=this.reasoner.ask(f.s,f.p);if(local){this.ctx.pending=null;return this.phrase(f.s,f.p,local.o)+'.';}
   this.ctx.pending={...f};
   if(this.cfg.autoSearch&&this.searchTool){const r=await this.runSearch(f.text);const a=await this.integrateSearch(f,r);if(a)return a;}
   if(f.s==='$user')return'لا أعرف '+f.pd+'ك بعد. قل لي: «'+f.pd+'ي ...»';
   if(f.s==='$self')return'لا أعرف '+f.pd+'ي بعد.';
   return this.searchTool?'لا أعرف الإجابة بعد. يمكنك أن تقول «ابحث عنها» لإعادة البحث.':'فهمت سؤالك لكن لا أعرف الإجابة بعد. يمكنك تعليمي: «'+f.pd+' '+f.sd+' هي ...» أو تفعيل البحث.';
  }
  if(f.type==='search'){const q=f.query||this.ctx.pending?.text;if(!q)return'عن ماذا تريد أن أبحث؟';if(!this.searchTool)return'أداة البحث غير مفعّلة حاليًا في نواة رون.';const r=await this.runSearch(q);if(!r)return'بحثت ولم أجد نتيجة مفيدة.';return(await this.integrateSearch(this.ctx.pending,r))||r.answer;}
  this.store.logUnparsed(f.text);return f.question?'فهمت أنه سؤال، لكن صياغته خارج ما أستطيع تحليله بعد.':'لم أفهم الجملة. جرّب صياغة أخرى.';
 }
 async integrateSearch(p,r){
  const e=evaluateResult(r,this.ctx.pending?.text||this.ctx.history.at(-1)?.text);if(!e)return null;
  if(this.answerModel)try{const m=await this.answerModel({question:this.ctx.pending?.text,evidence:[e],history:this.ctx.history.slice(-8)});if(m?.answer){this.ctx.pending=null;return clean(m.answer).slice(0,this.cfg.maxAnswerChars);}}catch(err){this.store.event('model_error',{message:String(err?.message||err)});}
  if(this.ctx.pending?.s&&(e.score>=this.cfg.minSearchConfidence||e.confidence>=.85)){const p=this.ctx.pending;this.store.set(p.s,p.p,key(e.answer),{source:e.confidence>=.85?'official':'search',confidence:e.confidence,sd:p.sd,pd:p.pd,od:e.answer});this.ctx.pending=null;}
  this.store.event('search',{query:this.ctx.pending?.text||this.ctx.history.at(-1)?.text,source:e.source,confidence:e.confidence});
  return e.answer+'\n(المصدر: '+e.source+')';
 }
 async runSearch(q){let timer;const timeout=new Promise(r=>timer=setTimeout(()=>r(null),this.cfg.searchTimeoutMs));try{return await Promise.race([Promise.resolve().then(()=>this.searchTool(q)),timeout]);}catch{return null;}finally{clearTimeout(timer);}}
}

if(typeof module!=='undefined'&&module.exports)module.exports={RonCore,FactStore,Reasoner,createRuleNLU,DEFAULT_RULES,MemoryAdapter,JsonFileAdapter,LocalStorageAdapter,clean,normalize,key,arithmetic,evaluateResult};
if(typeof globalThis!=='undefined'){globalThis.RonCore=RonCore;globalThis.RonCoreAdapters={MemoryAdapter,JsonFileAdapter,LocalStorageAdapter};}

if(typeof require==='function'&&typeof module!=='undefined'&&require.main===module){const readline=require('readline');const core=new RonCore({adapter:new JsonFileAdapter(process.argv[2]||'./ron-data.json')});const rl=readline.createInterface({input:process.stdin,output:process.stdout,prompt:'أنت> '});console.log('رون جاهز. (Ctrl+C للخروج)');rl.prompt();(async()=>{for await(const line of rl){if(line.trim())console.log('رون> '+(await core.handle(line)).reply);rl.prompt();}})();}
?s:key(s),property=key(p);return this.facts.find(f=>f.s===subject&&f.p===property);}
 set(s,p,o,{source='user',confidence=1,sd,pd,od,persist=true}={}){
  const i=this.facts.findIndex(f=>f.s===s&&f.p===p&&(!this.multi.has(p)||f.o===o));let status,prev;
  if(i<0){this.facts.push({s,p,o,source,confidence,ts:Date.now(),history:[]});status='added';}
  else{const f=this.facts[i];prev=f.o;if(f.o===o)status='unchanged';else if((RANK[source]??0)>=(RANK[f.source]??0)){f.history.push({o:f.o,source:f.source,confidence:f.confidence,ts:f.ts});Object.assign(f,{o,source,confidence,ts:Date.now()});status='replaced';}else status='kept';}
  if(status==='added'||status==='replaced'){if(sd&&s[0]!=='$')this.disp[s]=sd;if(pd)this.disp[p]=pd;if(od)this.disp[o]=od;}
  if(persist)this.save();return{status,prev};
 }
 logUnparsed(text){this.unparsed.push({text,ts:Date.now()});if(this.unparsed.length>1000)this.unparsed.shift();this.save();}
 event(type,data={}){this.events.push({type,ts:Date.now(),...data});if(this.events.length>2000)this.events.shift();this.save();}
 save(){this.adapter.save({facts:this.facts,disp:this.disp,unparsed:this.unparsed,events:this.events});}
}

const DEFAULT_RULES=[
 {name:'transitive:جزء_من',if:[['?a','جزء_من','?b'],['?b','جزء_من','?c']],then:['?a','جزء_من','?c']},
 {name:'symmetric:متزوج_من',if:[['?a','متزوج_من','?b']],then:['?b','متزوج_من','?a']}
];
const isVar=x=>typeof x==='string'&&x[0]==='?';
function unify(p,t,b){const o={...b};for(let i=0;i<3;i++){if(isVar(p[i])){if(p[i] in o&&o[p[i]]!==t[i])return null;o[p[i]]=t[i];}else if(p[i]!==t[i])return null;}return o;}
function* solve(ps,ts,b={}){if(!ps.length){yield b;return;}const [p,...r]=ps;for(const t of ts){const x=unify(p,t,b);if(x)yield* solve(r,ts,x);}}
class Reasoner{
 constructor(store,rules=DEFAULT_RULES,maxRounds=8){this.store=store;this.rules=rules;this.maxRounds=maxRounds;}
 infer(){const k=new Map(),id=t=>t.join('\u0001');for(const f of this.store.facts)k.set(id([f.s,f.p,f.o]),{t:[f.s,f.p,f.o],via:f.source});for(let r=0;r<this.maxRounds;r++){let ch=false,ts=[...k.values()].map(x=>x.t);for(const rule of this.rules)for(const b of solve(rule.if,ts)){const t=rule.then.map(x=>isVar(x)?b[x]:x);if(t.every(x=>x!=null)&&!k.has(id(t))){k.set(id(t),{t,via:rule.name});ch=true;}}if(!ch)break;}return k;}
 ask(s,p){const d=this.store.get(s,p);if(d)return{o:d.o,via:d.source,confidence:d.confidence};for(const x of this.infer().values())if(x.t[0]===s&&x.t[1]===p)return{o:x.t[2],via:x.via,confidence:.7};return null;}
}

function arithmetic(text){
 let s=normalize(text).replace(/×/g,'*').replace(/÷/g,'/').replace(/[^0-9+\-*/().%\s]/g,'').trim();
 if(!s||!/[+\-*/%]/.test(s)||!/\d/.test(s))return null;
 const ts=s.match(/\d+(?:\.\d+)?|[()+\-*/%]/g)||[];if(ts.join('')!==s.replace(/\s+/g,''))return null;let i=0;
 const primary=()=>{if(ts[i]==='('){i++;const v=expr();if(ts[i]!==')')throw 0;i++;return v;}if(ts[i]==='-'){i++;return-primary();}if(!/^\d/.test(ts[i]||''))throw 0;return Number(ts[i++]);};
 const term=()=>{let v=primary();while(['*','/','%'].includes(ts[i])){const op=ts[i++],b=primary();if(op==='*')v*=b;else if(op==='/'){if(b===0)throw 0;v/=b;}else v%=b;}return v;};
 const expr=()=>{let v=term();while(['+','-'].includes(ts[i])){const op=ts[i++],b=term();v=op==='+'?v+b:v-b;}return v;};
 try{const v=expr();return i===ts.length&&Number.isFinite(v)?String(v):null;}catch{return null;}
}

function evaluateResult(r,q){
 if(!r?.answer)return null;const answer=clean(r.answer).slice(0,1800),source=clean(r.source||'web');
 const st=normalize(source+' '+(r.url||''));let sc=.45;if(/official|gov|government|edu|university|who|nih|nasa|arxiv/.test(st))sc=.9;else if(/wikipedia/.test(st))sc=.68;else if(/duckduckgo/.test(st))sc=.5;
 const rel=overlap(q,answer);return{...r,answer,source,confidence:Math.max(r.confidence??0,sc),relevance:rel,score:rel*.65+sc*.35};
}

const SEED=[['مصر','عاصمة','القاهرة'],['السعودية','عاصمة','الرياض'],['الإمارات','عاصمة','أبوظبي'],['الأردن','عاصمة','عمان'],['العراق','عاصمة','بغداد']];

class RonCore{
 constructor(opts={}){
  this.cfg={...DEFAULTS,...opts};this.store=new FactStore(opts.adapter||new MemoryAdapter(),{multi:opts.multi||[]});this.reasoner=new Reasoner(this.store,opts.rules||DEFAULT_RULES,opts.maxReasoningRounds||8);this.nlu=opts.nlu||createRuleNLU(this.cfg);this.searchTool=opts.searchTool||null;this.answerModel=opts.answerModel||null;this.ctx={pending:null,history:[]};
  this.store.set('$self','اسم',key(this.cfg.selfName),{source:'seed',confidence:1,pd:'اسم',od:this.cfg.selfName,persist:false});
  if(opts.seed!==false)for(const x of SEED)this.store.set(key(x[0]),key(x[1]),key(x[2]),{source:'seed',confidence:.9,sd:x[0],pd:x[1],od:x[2],persist:false});
  this.store.save();
 }
 async handle(text){const input=clean(text);if(!input)return{reply:'اكتب رسالة أولًا.',frames:[]};this.ctx.history.push({role:'user',text:input,ts:Date.now()});if(this.ctx.history.length>30)this.ctx.history.shift();const frames=this.nlu.parse(input),out=[];for(const f of frames)out.push(await this.exec(f));const reply=out.filter(Boolean).join('\n');this.ctx.history.push({role:'assistant',text:reply,ts:Date.now()});
  globalThis.RonLearning?.addExperience?.(input,reply,'conversation',.65);
  return{reply,frames};}
 phrase(s,p,o){const pd=this.store.display(p),od=this.store.display(o);if(s==='$user')return pd+'ك هو '+od;if(s==='$self')return pd+'ي هو '+od;return pd+' '+this.store.display(s)+' '+(/[هة]$/.test(pd)?'هي':'هو')+' '+od;}
 async exec(f){
  if(f.type==='smalltalk')return f.kind==='howareyou'?'أنا بخير وجاهز للعمل. ماذا تريد أن نفعل؟':f.kind==='call'?'نعم، أنا معك.':'مرحبًا، كيف أساعدك؟';
  if(f.type==='context'){
   if(f.kind==='previous'){
    const h=this.ctx.history.filter(x=>x.role==='user');
    const prev=h.length>1?h[h.length-2]:null;
    return prev?`آخر شيء قلته قبل رسالتك الحالية كان: «${prev.text}»`:'لا توجد رسالة سابقة أستطيع الرجوع إليها بعد.';
   }
   const facts=this.store.facts.filter(x=>x.source!=='seed');
   const learned=[];
   for(const fct of facts)learned.push(this.phrase(fct.s,fct.p,fct.o));
   const knowledge=globalThis.RonLearning?.get?.()?.knowledge||[];
   for(const x of knowledge.slice(-20))if(x?.text)learned.push(clean(x.text));
   const unique=[...new Set(learned)].slice(-12);
   return unique.length?'نعم، لدي معرفة محفوظة من محادثاتك وتعليماتك. من أمثلتها:\n'+unique.map((x,i)=>`${i+1}. ${x}`).join('\n'):'لم أتعلم معلومات شخصية أو تعليمات جديدة بعد.';
  }
  if(f.type==='assert'){const r=this.store.set(f.s,f.p,f.o,{sd:f.sd,pd:f.pd,od:f.od});globalThis.RonLearning?.addKnowledge?.({text:this.phrase(f.s,f.p,f.o),kind:'fact',source:'conversation',confidence:1});const line=this.phrase(f.s,f.p,f.o);if(r.status==='unchanged')return'أعرف ذلك بالفعل: '+line+'.';if(r.status==='replaced')return'تم، حدّثتها: '+line+' (كانت: '+this.store.display(r.prev)+').';return'تم. '+line+'.';}
  if(f.type==='ask'){
   const calc=arithmetic(f.text);if(calc!==null)return'النتيجة: '+calc;
   const local=this.reasoner.ask(f.s,f.p);if(local){this.ctx.pending=null;return this.phrase(f.s,f.p,local.o)+'.';}
   this.ctx.pending={...f};
   if(this.cfg.autoSearch&&this.searchTool){const r=await this.runSearch(f.text);const a=await this.integrateSearch(f,r);if(a)return a;}
   if(f.s==='$user')return'لا أعرف '+f.pd+'ك بعد. قل لي: «'+f.pd+'ي ...»';
   if(f.s==='$self')return'لا أعرف '+f.pd+'ي بعد.';
   return this.searchTool?'لا أعرف الإجابة بعد. يمكنك أن تقول «ابحث عنها» لإعادة البحث.':'فهمت سؤالك لكن لا أعرف الإجابة بعد. يمكنك تعليمي: «'+f.pd+' '+f.sd+' هي ...» أو تفعيل البحث.';
  }
  if(f.type==='search'){const q=f.query||this.ctx.pending?.text;if(!q)return'عن ماذا تريد أن أبحث؟';if(!this.searchTool)return'أداة البحث غير مفعّلة حاليًا في نواة رون.';const r=await this.runSearch(q);if(!r)return'بحثت ولم أجد نتيجة مفيدة.';return(await this.integrateSearch(this.ctx.pending,r))||r.answer;}
  this.store.logUnparsed(f.text);return f.question?'فهمت أنه سؤال، لكن صياغته خارج ما أستطيع تحليله بعد.':'لم أفهم الجملة. جرّب صياغة أخرى.';
 }
 async integrateSearch(p,r){
  const e=evaluateResult(r,this.ctx.pending?.text||this.ctx.history.at(-1)?.text);if(!e)return null;
  if(this.answerModel)try{const m=await this.answerModel({question:this.ctx.pending?.text,evidence:[e],history:this.ctx.history.slice(-8)});if(m?.answer){this.ctx.pending=null;return clean(m.answer).slice(0,this.cfg.maxAnswerChars);}}catch(err){this.store.event('model_error',{message:String(err?.message||err)});}
  if(this.ctx.pending?.s&&(e.score>=this.cfg.minSearchConfidence||e.confidence>=.85)){const p=this.ctx.pending;this.store.set(p.s,p.p,key(e.answer),{source:e.confidence>=.85?'official':'search',confidence:e.confidence,sd:p.sd,pd:p.pd,od:e.answer});this.ctx.pending=null;}
  this.store.event('search',{query:this.ctx.pending?.text||this.ctx.history.at(-1)?.text,source:e.source,confidence:e.confidence});
  return e.answer+'\n(المصدر: '+e.source+')';
 }
 async runSearch(q){let timer;const timeout=new Promise(r=>timer=setTimeout(()=>r(null),this.cfg.searchTimeoutMs));try{return await Promise.race([Promise.resolve().then(()=>this.searchTool(q)),timeout]);}catch{return null;}finally{clearTimeout(timer);}}
}

if(typeof module!=='undefined'&&module.exports)module.exports={RonCore,FactStore,Reasoner,createRuleNLU,DEFAULT_RULES,MemoryAdapter,JsonFileAdapter,LocalStorageAdapter,clean,normalize,key,arithmetic,evaluateResult};
if(typeof globalThis!=='undefined'){globalThis.RonCore=RonCore;globalThis.RonCoreAdapters={MemoryAdapter,JsonFileAdapter,LocalStorageAdapter};}

if(typeof require==='function'&&typeof module!=='undefined'&&require.main===module){const readline=require('readline');const core=new RonCore({adapter:new JsonFileAdapter(process.argv[2]||'./ron-data.json')});const rl=readline.createInterface({input:process.stdin,output:process.stdout,prompt:'أنت> '});console.log('رون جاهز. (Ctrl+C للخروج)');rl.prompt();(async()=>{for await(const line of rl){if(line.trim())console.log('رون> '+(await core.handle(line)).reply);rl.prompt();}})();}
