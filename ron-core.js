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

const INVALID_NAME_PREFIX=/^(?:لا|ليس|مش|مو|ما|بسالك|اسالك|عن|اسمك|اسمي|لكن|ولكن|وليس|بل)(?:\s|$)/;
const INVALID_NAME_SUFFIX=/(?:^|\s)(?:وليس|لكن|ولكن|مش|مو|بل|لا|ليس|بسالك|اسالك|عن|اسمي|اسمك)$/;
function isSafeNameValue(value){
 const raw=clean(value),n=normalize(raw),parts=n.split(' ').filter(Boolean);
 return !!n&&parts.length<=4&&!INVALID_NAME_PREFIX.test(n)&&!INVALID_NAME_SUFFIX.test(n)&&!/(?:^|\s)(?:بسالك|اسالك)\s+عن(?:\s|$)/.test(n)&&!/[؟?]/.test(raw);
}
function createRuleNLU(cfg=DEFAULTS){
 cfg={...DEFAULTS,...cfg}; const pers=cfg.personal.map(key).join('|');
 const fillers=new Set(['يا','طيب','اذا','ان','تعلم','اعلم','تذكر','اوكي','حسنا',key(cfg.selfName)]);
 const parse=text=>{
  const c=clean(text),n0=normalize(c); let n=n0,t=c;
  // Direct conversational forms must be recognized before filler removal.
  if(/^(?:و\s*)?(?:ما|ايه|اي)\s+اسمي\s+و\s*(?:ما|ايه|اي)\s+اسمك$/.test(n0)||/^(?:ما|ايه|اي)\s+اسمي\s+و?\s*(?:ما|ايه|اي)\s+اسمك$/.test(n0))return[{type:'ask-both-names',text:c}];
  // Parse profile assertions before the generic relation parser.
  if(/^(?:اسمك|اسمك\s+هو|انت\s+اسمك)\s+رون\s+(?:فعلا|حقا)$/.test(n0))return[{type:'identity-confirm',text:c}];
  if(/^(?:انا\s+)?(?:بسالك|اسالك)\s+عن\s+اسمي(?=\s|$)/.test(n0))return[{type:'unknown',text:c}];
  let correction=n0.match(/^(?:انا\s+)?اسمي\s+(.+?)\s+(?:وليس|لكن|ولكن|مش|مو)\s+(?:اسمك|اسم|انت|انا)(?=\s|$).*$/);
  if(correction){const value=correction[1].trim();if(isSafeNameValue(value))return[{type:'assert',s:'$user',p:'اسم',o:key(value),sd:'',pd:'اسم',od:value}];return[{type:'memory-clarify',text:c}];}
  let namedBoth=n0.match(/^(?:انا\s+)?اسمي\s+(.+?)\s+(?:و)?اسمك\s+(?:هو\s+)?(.+)$/);
  if(namedBoth){
   const userName=namedBoth[1].trim(),selfName=namedBoth[2].trim();
   if(!isSafeNameValue(userName))return[{type:'memory-clarify',text:c}];
   const frames=[{type:'assert',s:'$user',p:'اسم',o:key(userName),sd:'',pd:'اسم',od:userName}];
   if(isSafeNameValue(selfName)&&key(selfName)===key(cfg.selfName))frames.push({type:'assert',s:'$self',p:'اسم',o:key(selfName),sd:'',pd:'اسم',od:selfName});
   else frames.push({type:'identity-protected',text:c});
   return frames;
  }
  let explicitName=n0.match(/^(?:انا\s+)?اسمي\s+(?:هو\s+)?(.+?)$/);
  if(explicitName){const value=explicitName[1].trim();if(isSafeNameValue(value))return[{type:'assert',s:'$user',p:'اسم',o:key(value),sd:'',pd:'اسم',od:value}];return[{type:'memory-clarify',text:c}];}
  if(/^(?:انت\s+)?(?:ما\s+هو|ماهو|ماهي|ما)\s+اسمك$/.test(n0)||/^(?:و\s*)?(?:وانت\s+)?(?:ما\s+هو|ماهو|ماهي|ما)\s+اسمك$/.test(n0))return[{type:'ask',s:'$self',p:'اسم',sd:'',pd:'اسم',text:c}];
  if(/^(?:و\s*)?(?:انت\s+)?(?:ما\s+هو|ماهو|ماهي|ما)\s+اسمي$/.test(n0))return[{type:'ask',s:'$user',p:'اسم',sd:'',pd:'اسم',text:c}];
  if(/^(?:و\s*)?(?:انت\s+)?(?:كم\s+عمرك|ما\s+عمرك|ما\s+هو\s+عمرك)$/.test(n0))return[{type:'self-age',text:c}];
  while(true){const m=/^(\S+)\s+/.exec(n);if(!m||!fillers.has(m[1]))break;n=n.slice(m[0].length);t=t.slice(m[0].length);}
  if(!n)return[{type:'unknown',text:c}];
  if(/^(رون|يا\s+رون|رون\s*[!،,.؟?]*)$/.test(n0))return[{type:'smalltalk',kind:'call'}];
  if(/^(?:ليه|لماذا|ازاي|كيف ذلك|لماذا ذلك|ما السبب|ما معنى ذلك|ماذا تقصد|كيف يعني)$/.test(n0))return[{type:'context',kind:'why-previous',text:c}];
  if(/^(?:اذا\s+)?(?:لماذا|ليه|ازاي)\s+(?:قلت|قولت|كتبت|ذكرت|حفظت)(?=\s|$)/.test(n0))return[{type:'context',kind:'why-previous',text:c}];
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
  if(frames.length){
   const blocked=frames.some(f=>f.s==='$self'&&key(f.p)==='اسم'&&key(f.o)!==key(cfg.selfName));
   const safeFrames=frames.filter(f=>!(f.s==='$self'&&key(f.p)==='اسم'&&key(f.o)!==key(cfg.selfName)));
   if(blocked)safeFrames.push({type:'identity-protected',text:c});
   return safeFrames;
  }
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
 constructor(adapter=new MemoryAdapter(),{multi=[]}={}){this.adapter=adapter;this.multi=new Set(multi.map(key));const d=adapter.load()||{};this.facts=(d.facts||[]).filter(f=>!(f&&f.s==='$user'&&key(f.p)==='اسم'&&/اسالك عن اسمك|اسالك|ما اسمي|ما اسمك|اسمي وما اسمك/.test(key(f.o))));this.disp=d.disp||{};this.unparsed=d.unparsed||[];this.events=d.events||[];}
 display(k){return k==='$user'?'أنت':k==='$self'?'أنا':this.disp[k]??k;}
 get(s,p){const subject=key(s),property=key(p);return this.facts.find(f=>key(f.s)===subject&&key(f.p)===property);}
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
 // Parse only an expression (optionally preceded by a small, explicit Arabic math prompt).
 // Never strip arbitrary letters: "hello 2+2" must not silently become a calculation.
 if(typeof text!=='string'||text.length>256)return null;
 let s=clean(text)
  .replace(/[أإآ]/g,'ا')
  .replace(/[٠-٩]/g,c=>String('٠١٢٣٤٥٦٧٨٩'.indexOf(c)))
  .replace(/[۰-۹]/g,c=>String('۰۱۲۳۴۵۶۷۸۹'.indexOf(c)))
  .replace(/×/g,'*').replace(/÷/g,'/').replace(/[−–]/g,'-');
 s=s.replace(/^(?:و\s*)?(?:(?:احسب(?:\s+لي)?|كم\s+(?:تساوي|يساوي)|ما\s+الناتج(?:\s+عن)?|ما\s+ناتج)\s*[:：؟?]?\s*)/,'').trim();
 if(!s||!/[+\-*/%]/.test(s)||!/[0-9]/.test(s)||/[^0-9+\-*/().%\s]/.test(s))return null;
 const compact=s.replace(/\s+/g,'');
 const ts=s.match(/(?:\d+(?:\.\d*)?|\.\d+|[()+\-*/%])/g)||[];
 if(!ts.length||ts.length>160||ts.join('')!==compact)return null;
 let i=0;
 const primary=()=>{
  if(ts[i]==='('){i++;const v=expr();if(ts[i]!==')')throw 0;i++;return v;}
  if(ts[i]==='-'||ts[i]==='+'){const sign=ts[i++];const v=primary();return sign==='-'?-v:v;}
  if(!/^(?:\d+(?:\.\d*)?|\.\d+)$/.test(ts[i]||''))throw 0;
  return Number(ts[i++]);
 };
 const term=()=>{let v=primary();while(['*','/','%'].includes(ts[i])){const op=ts[i++],b=primary();if((op==='/'||op==='%')&&b===0)throw 0;if(op==='*')v*=b;else if(op==='/')v/=b;else v%=b;if(!Number.isFinite(v))throw 0;}return v;};
 const expr=()=>{let v=term();while(['+','-'].includes(ts[i])){const op=ts[i++],b=term();v=op==='+'?v+b:v-b;if(!Number.isFinite(v))throw 0;}return v;};
 try{const v=expr();return i===ts.length&&Number.isFinite(v)?String(Object.is(v,-0)?0:v):null;}catch{return null;}
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
 async handle(text){const input=clean(text);if(!input)return{reply:'اكتب رسالة أولًا.',frames:[]};this.ctx.history.push({role:'user',text:input,ts:Date.now()});if(this.ctx.history.length>30)this.ctx.history.shift();const calc=arithmetic(input);if(calc!==null){const reply='النتيجة: '+calc;this.ctx.history.push({role:'assistant',text:reply,ts:Date.now()});if(this.ctx.history.length>30)this.ctx.history.shift();globalThis.RonLearning?.addExperience?.(input,reply,'conversation',.65);return{reply,frames:[{type:'calculation',text:input,result:calc}]};}
  // RonCore is the sole orchestrator. Language understanding, retrieval and inference
  // are invoked as internal capabilities; they do not create a second agent or bypass this core.
  const cognition=globalThis.RonAgent;
  if(cognition&&typeof cognition.think==='function'){
   try{
    const thought=await cognition.think(input);
    if(thought&&typeof thought.answer==='string'&&thought.answer.trim()&&thought.stage!=='needs-web-or-model'){
     const reply=clean(thought.answer).slice(0,this.cfg.maxAnswerChars);
     this.ctx.history.push({role:'assistant',text:reply,ts:Date.now()});
     if(this.ctx.history.length>30)this.ctx.history.shift();
     this.store.event('integrated-cognition',{stage:thought.stage||'local',confidence:thought.confidence??null});
     globalThis.RonLearning?.addExperience?.(input,reply,'integrated-cognition',thought.confidence??.5);
     return{reply,frames:[{type:'integrated-cognition',stage:thought.stage||'local',confidence:thought.confidence??null}]};
    }
   }catch(err){this.store.event('cognition_error',{message:String(err?.message||err)});}
  }
  const frames=this.nlu.parse(input),out=[];for(const f of frames)out.push(await this.exec(f));const reply=out.filter(Boolean).join('\n');this.ctx.history.push({role:'assistant',text:reply,ts:Date.now()});
  globalThis.RonLearning?.addExperience?.(input,reply,'conversation',.65);
  return{reply,frames};}
 phrase(s,p,o){const pd=this.store.display(p),od=this.store.display(o);if(s==='$user')return pd+'ك هو '+od;if(s==='$self')return pd+'ي هو '+od;return pd+' '+this.store.display(s)+' '+(/[هة]$/.test(pd)?'هي':'هو')+' '+od;}
 async exec(f){
  if(f.type==='memory-clarify')return 'لم أغيّر الذاكرة لأن صياغة الاسم غير واضحة أو تحتوي على نفي أو تكملة. اكتب الاسم وحده، مثل: «اسمي أركانوس».';
  if(f.type==='identity-protected'){const self=this.store.get('$self','اسم');return 'هويتي المسجلة هي '+(self?this.store.display(self.o):this.cfg.selfName)+'، ولن أغيّرها من عبارة ملتبسة. إذا كنت تقصد تغيير الإعداد فعلًا، وضّح الأمر مباشرة.';}
  if(f.type==='identity-confirm')return 'نعم، اسمي '+this.cfg.selfName+'.';
  if(f.type==='smalltalk')return f.kind==='howareyou'?'أنا بخير وجاهز للعمل. ماذا تريد أن نفعل؟':f.kind==='call'?'نعم، أنا معك.':'مرحبًا، كيف أساعدك؟';
  if(f.type==='self-age')return this.store.get('$self','عمر')?'عمري المسجل هو '+this.store.display(this.store.get('$self','عمر').o)+'.':'ليس لدي عمر بشري؛ أنا برنامج، ولا أملك عمرًا شخصيًا مثل الإنسان.';
  if(f.type==='ask-both-names'){const user=this.store.get('$user','اسم'),self=this.store.get('$self','اسم');return (user?'اسمك '+this.store.display(user.o):'لم تخبرني باسمك بعد')+'، واسمي '+(self?this.store.display(self.o):this.cfg.selfName)+'.';}
  if(f.type==='context'){
   if(f.kind==='why-previous'){
    const previous=this.ctx.history.filter(x=>x.role==='assistant').at(-1);
    const query=normalize(f.text);
    if(query.includes('اسم')){
     const user=this.store.get('$user','اسم'),self=this.store.get('$self','اسم');
     if(user)return 'لأنني أخطأت في فهم عبارتك السابقة وخلطت بين سؤالك عن الاسم والاسم نفسه. الصحيح أن اسمك '+this.store.display(user.o)+' واسمي '+(self?this.store.display(self.o):this.cfg.selfName)+'. أعتذر عن الخلط.';
     return 'لأنني أخطأت في تحليل عبارتك، ولا ينبغي أن أحوّل سؤالك عن الاسم إلى حقيقة محفوظة. لا أملك اسمك الصحيح مسجلًا الآن؛ أخبرني به وسأحفظه بوضوح.';
    }
    return previous?'قد يكون ردي السابق غير دقيق. كان ردي: «'+clean(previous.text).slice(0,240)+'». سأراجعه بدل أن أخمّن سببًا غير مؤكد.':'لا أجد ردًا سابقًا واضحًا أراجعه، لذلك لن أخمّن سببًا.';
   }
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

/* === RonCore integrated capabilities: cognition, knowledge, reasoning, and learning === */

/* Integrated capability: RonAgent cognition/orchestration */
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
const arithmetic=s=>{const q=norm(s).replace(/×/g,"*").replace(/÷/g,"/").replace(/[^0-9+\-*/().%\s]/g,"").trim();if(!q||!/[+\-*/%]/.test(q)||!/\d/.test(q))return null;const ts=q.match(/\d+(?:\.\d+)?|[()+\-*/%]/g)||[];if(ts.join("")!==q.replace(/\s+/g,""))return null;let i=0;const primary=()=>{if(ts[i]==="("){i++;const v=expr();if(ts[i]!==")")throw 0;i++;return v;}if(ts[i]==="-"){i++;return-primary();}if(!/^\d/.test(ts[i]||""))throw 0;return Number(ts[i++]);};const term=()=>{let v=primary();while(["*","/","%"].includes(ts[i])){const op=ts[i++],b=primary();if(op==="*")v*=b;else if(op==="/"){if(b===0)throw 0;v/=b;}else v%=b;}return v;};const expr=()=>{let v=term();while(["+","-"].includes(ts[i])){const op=ts[i++],b=term();v=op==="+"?v+b:v-b;}return v;};try{const v=expr();return i===ts.length&&Number.isFinite(v)?String(v):null;}catch{return null;}};
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
/* Integrated capability: RonKnowledge local retrieval */
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
/* Integrated capability: RonReasoner symbolic inference */
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
 m=n.match(/^(.+?)\s+ي(?:ا)?تي\s+(بعد|قبل)\s+(.+)$/);if(m)return{subject:clean(m[1]),relation:m[2]==="بعد"?"follows":"precedes",object:clean(m[3]),surface:m[2],source:"local",confidence:.85,text:original,derived:false};
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
   const same=rel==="precedes"||rel==="follows"?out.filter(x=>x.relation===rel):[];
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
 m=n.match(/^(?:ما الذي|من الذي)\s+ي(?:ا)?تي\s+(بعد|قبل)\s+(.+)$/);if(m)return{type:"relation-object",relation:m[1]==="بعد"?"follows":"precedes",object:clean(m[2]),word:m[1]};
 m=n.match(/^(?:ما هي|ما هو|ماهي|ماهو|ما الذي|ايه|اي)\s+(.+?)\s+(يسبب|تسبب|يدعم|تدعم|قبل|يسبق|بعد|يتبع|يحتاج|يحتاج الى|جزء من|ضمن)\s+(.+)$/);
 if(m)return{type:"relation",relation:relMap[m[2]],subject:clean(m[1]),object:clean(m[3]),word:m[2]};
 m=n.match(/^هل\s+(.+?)\s+(يسبب|تسبب|يدعم|تدعم|قبل|يسبق|بعد|يتبع|يحتاج|يحتاج الى|جزء من|ضمن)\s+(.+)$/);
 if(m)return{type:"yesno",relation:relMap[m[2]],subject:clean(m[1]),object:clean(m[3])};
 m=n.match(/^(?:لماذا|ليه|لما)\s+(.+)$/);if(m)return{type:"why",subject:clean(m[1])};
 return{type:"unknown"};
};
const answer=(q,rawFacts)=>{
 const facts=derive(rawFacts||[]),ask=question(q);
 if(ask.type==="capital"){const f=facts.find(x=>x.relation==="is"&&norm(x.object)===norm(ask.object));if(f)return{answer:f.subject+" هي عاصمة "+f.object+".",confidence:f.confidence,source:f.source,trace:f}}
 if(ask.type==="relation-object"){const f=facts.find(x=>x.relation===ask.relation&&norm(x.object)===norm(ask.object));if(f)return{answer:f.subject+" يأتي "+ask.word+" "+f.object+".",confidence:f.confidence,source:f.source,trace:f}}
 if(ask.type==="relation"){const f=facts.find(x=>x.relation===ask.relation&&norm(x.subject)===norm(ask.subject)&&norm(x.object)===norm(ask.object));if(f)return{answer:f.subject+" "+ask.word+" "+f.object+".",confidence:f.confidence,source:f.source,trace:f}}
 if(ask.type==="yesno"){const f=facts.find(x=>x.relation===ask.relation&&norm(x.subject)===norm(ask.subject)&&norm(x.object)===norm(ask.object));if(f)return{answer:"نعم، لدي أساس محلي لهذا الاستنتاج.",confidence:f.confidence,source:f.source,trace:f}}
 if(ask.type==="why"){const f=facts.find(x=>x.relation==="causes"&&norm(x.object)===norm(ask.subject));if(f)return{answer:"لأن "+f.subject+" يسبب "+f.object+".",confidence:f.confidence,source:f.source,trace:f}}
 const n=norm(q);if(/^(?:ما هو|ما هي|ماهو|ماهي)\s+/.test(n)){const subject=clean(n.replace(/^(?:ما هو|ما هي|ماهو|ماهي)\s+/,""));const f=facts.find(x=>x.relation==="is"&&norm(x.subject)===norm(subject));if(f)return{answer:f.object+".",confidence:f.confidence,source:f.source,trace:f}}
 return null;
};
globalThis.RonReasoner={normalize:norm,parse,parseAll,reason:derive,derive,answer};
})();
/* Integrated capability: RonLearning local experience/knowledge memory */
/* Ron Learning Core
 * Local, persistent learning/experience layer.
 * It does not change model weights; it stores validated knowledge, experiences,
 * confidence and provenance so Ron can improve across conversations.
 */
(()=>{"use strict";
const KEY="ron-learning-core-v1",MAX=600;
const BOOTSTRAP_KEY="ron-bootstrap-knowledge-v1";
const BOOTSTRAP=[
["مصر دولة في شمال شرق أفريقيا.","geography"],["عاصمة مصر هي القاهرة.","geography"],["القاهرة عاصمة مصر.","geography"],["أفريقيا قارة.","geography"],["آسيا قارة.","geography"],["أوروبا قارة.","geography"],["أمريكا الشمالية قارة.","geography"],["أمريكا الجنوبية قارة.","geography"],["أستراليا قارة.","geography"],["القارة القطبية الجنوبية قارة.","geography"],["المحيط الأطلسي محيط.","geography"],["المحيط الهادئ محيط.","geography"],["المحيط الهندي محيط.","geography"],["البحر المتوسط بحر.","geography"],["نهر النيل يمر عبر مصر.","geography"],["الأرض كوكب.","astronomy"],["الشمس نجم.","astronomy"],["القمر تابع طبيعي للأرض.","astronomy"],["الأرض تدور حول نفسها.","astronomy"],["الأرض تدور حول الشمس.","astronomy"],["دوران الأرض حول نفسها يرتبط بتعاقب الليل والنهار.","astronomy"],["دوران الأرض حول الشمس يرتبط بالسنة.","astronomy"],["المريخ كوكب في المجموعة الشمسية.","astronomy"],["المشتري كوكب في المجموعة الشمسية.","astronomy"],["زحل كوكب في المجموعة الشمسية.","astronomy"],["عطارد أقرب كواكب المجموعة الشمسية إلى الشمس.","astronomy"],["الزهرة كوكب في المجموعة الشمسية.","astronomy"],["نبتون كوكب في المجموعة الشمسية.","astronomy"],["أورانوس كوكب في المجموعة الشمسية.","astronomy"],["الماء مركب يتكون من الهيدروجين والأكسجين.","science"],["الهواء خليط من عدة غازات.","science"],["الأكسجين غاز.","science"],["ثاني أكسيد الكربون غاز.","science"],["النباتات تستخدم الضوء في عملية البناء الضوئي.","biology"],["البناء الضوئي يساعد النباتات على إنتاج الغذاء.","biology"],["الإنسان من الثدييات.","biology"],["الإنسان يحتاج إلى الماء والغذاء والأكسجين للبقاء.","biology"],["القلب عضو يضخ الدم في الجسم.","biology"],["الرئتان عضوان أساسيان للتنفس.","biology"],["الدم ينقل الأكسجين والمواد المختلفة داخل الجسم.","biology"],["الدماغ جزء رئيسي من الجهاز العصبي.","biology"],["العظام تدعم الجسم وتحمي أعضاء مهمة.","biology"],["النباتات كائنات حية.","biology"],["الحيوانات كائنات حية.","biology"],["الماء يتجمد عند نحو 0 درجة مئوية عند الضغط الجوي المعتاد.","physics"],["الماء يغلي عند نحو 100 درجة مئوية عند مستوى سطح البحر.","physics"],["الحرارة تنتقل من الجسم الأعلى حرارة إلى الجسم الأقل حرارة تلقائيًا.","physics"],["الجاذبية تؤثر في الأجسام ذات الكتلة.","physics"],["السرعة هي معدل تغير المسافة بالنسبة للزمن.","physics"],["الضوء ينتقل أسرع بكثير من الصوت في الهواء.","physics"],["اللغة العربية لغة سامية.","language"],["اللغة الإنجليزية من اللغات الجرمانية.","language"],["الجملة يمكن أن تتكون من كلمات تؤدي معنى.","language"],["الاسم والفعل والحرف من أقسام الكلمة في النحو العربي.","language"],["الصفر عدد.","mathematics"],["الواحد عدد.","mathematics"],["الجمع عملية حسابية.","mathematics"],["الطرح عملية حسابية.","mathematics"],["الضرب عملية حسابية.","mathematics"],["القسمة عملية حسابية.","mathematics"],["المثلث شكل له ثلاثة أضلاع.","mathematics"],["المربع شكل له أربعة أضلاع متساوية وأربع زوايا قائمة.","mathematics"],["المستطيل له أربعة أضلاع وأربع زوايا قائمة.","mathematics"],["الدائرة شكل هندسي مغلق.","mathematics"],["اليوم يتكون من 24 ساعة تقريبًا.","time"],["الساعة تتكون من 60 دقيقة.","time"],["الدقيقة تتكون من 60 ثانية.","time"],["الأسبوع يتكون من سبعة أيام.","time"],["السنة الميلادية المعتادة تتكون من 365 يومًا.","time"],["السنة الكبيسة تتكون من 366 يومًا.","time"],["الكمبيوتر جهاز إلكتروني يعالج البيانات.","technology"],["الإنترنت شبكة عالمية تربط شبكات وأجهزة.","technology"],["البرمجيات هي البرامج والتعليمات التي تشغل الحاسوب.","technology"],["HTML تستخدم لوصف بنية صفحات الويب.","technology"],["CSS تستخدم لتنسيق صفحات الويب.","technology"],["JavaScript لغة برمجة تستخدم كثيرًا في الويب.","technology"],["قاعدة البيانات تستخدم لتخزين وتنظيم البيانات.","technology"],["التاريخ يدرس أحداث الماضي.","history"],["الجغرافيا تدرس الأماكن والظواهر المرتبطة بسطح الأرض.","geography"],["العلم يعتمد على الملاحظة والاختبار والأدلة.","knowledge"],["المعلومة تحتاج إلى مصدر موثوق عندما تكون قابلة للتحقق.","knowledge"],["إذا لم تتوفر أدلة كافية يجب على رون أن يذكر عدم اليقين بدل اختلاق معلومة.","reasoning"],["عند عدم معرفة الإجابة يمكن لرون البحث في مصادر الإنترنت ثم مقارنة النتائج.","reasoning"],["رون لا يعتبر إجابة نموذج لغوي حقيقة تلقائيًا.","reasoning"],["رون يحتفظ بالمعرفة التي يتعلمها محليًا في ذاكرته.","learning"],["المعرفة الجديدة يمكن مراجعتها وتعزيز الثقة فيها مع تكرار المصادر الموثوقة.","learning"]
];
const clean=s=>String(s||"").trim().replace(/\s+/g," ");
const read=()=>{try{const v=JSON.parse(localStorage.getItem(KEY));return v&&typeof v==="object"?v:{knowledge:[],experiences:[],stats:{}}}catch{return{knowledge:[],experiences:[],stats:{}}}};
const write=v=>{try{localStorage.setItem(KEY,JSON.stringify(v))}catch{}};
const bootstrap=()=>{try{if(localStorage.getItem(BOOTSTRAP_KEY))return;const v=read();for(const [text,domain] of BOOTSTRAP){if(!v.knowledge.some(k=>clean(k.text).toLowerCase()===text.toLowerCase()))v.knowledge.push({text,domain,source:"ron-bootstrap",confidence:.85,at:new Date().toISOString()});}v.knowledge=v.knowledge.slice(-MAX);write(v);localStorage.setItem(BOOTSTRAP_KEY,"1")}catch{}};
bootstrap();
const api={
 get(){return read()},
 addKnowledge(item){const v=read(),x={...item,text:clean(item.text),at:new Date().toISOString()};if(!x.text)return false;const sig=x.text.toLowerCase();v.knowledge=v.knowledge.filter(k=>clean(k.text).toLowerCase()!==sig);v.knowledge.push(x);v.knowledge=v.knowledge.slice(-MAX);write(v);return true},
 addExperience(input,output,source="conversation",confidence=.5){const v=read(),i=clean(input),o=clean(output);if(!i||!o)return false;v.experiences.push({input:i,output:o,source,confidence,at:new Date().toISOString()});v.experiences=v.experiences.slice(-MAX);v.stats.interactions=(v.stats.interactions||0)+1;write(v);return true},
 reinforce(text,delta=.05){const v=read(),q=clean(text).toLowerCase(),k=v.knowledge.find(x=>clean(x.text).toLowerCase()===q);if(k)k.confidence=Math.max(0,Math.min(1,(k.confidence??.5)+delta));write(v);return !!k},
 search(query,limit=12){bootstrap();const v=read(),q=clean(query).toLowerCase();if(!q)return[];const words=q.split(" ").filter(x=>x.length>2);return v.knowledge.map((x,i)=>{const t=clean(x.text).toLowerCase();let s=t===q?20:(t.includes(q)?10:0);for(const w of words)if(t.includes(w))s++;return{x,s,i}}).filter(z=>z.s>0).sort((a,b)=>b.s-a.s||b.i-a.i).slice(0,limit).map(z=>z.x)},
 summary(){const v=read();return{knowledge:v.knowledge.length,experiences:v.experiences.length,interactions:v.stats.interactions||0}}
};
globalThis.RonLearning=api;
})();