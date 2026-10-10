'use strict';
const test=require('node:test');
const assert=require('node:assert');
const os=require('node:os');
const path=require('node:path');
const fs=require('node:fs');
const {RonCore,JsonFileAdapter,Reasoner,FactStore,arithmetic,evaluateResult}=require('./ron-core');
const say=async(c,t)=>(await c.handle(t)).reply;

test('identity',async()=>{const r=new RonCore();assert.match(await say(r,'ما اسمك'),/رون/);await say(r,'اسمي زيريوس، واسمك رون');assert.strictEqual(r.store.get('$user','اسم').o,'زيريوس');assert.strictEqual(r.store.get('$self','اسم').o,'رون');assert.match(await say(r,'ما اسمي'),/زيريوس/);});
test('unknown identity is explicit',async()=>{assert.match(await say(new RonCore(),'ما اسمي'),/لا أعرف اسمك بعد/);});
test('update keeps history',async()=>{const r=new RonCore();await say(r,'اسمي زيريوس');assert.match(await say(r,'اسمي علي'),/حدّثتها.*زيريوس/);assert.ok(r.store.get('$user','اسم').history.length);});
test('seed facts',async()=>{const r=new RonCore();assert.match(await say(r,'ما عاصمه مصر'),/القاهرة/);});
test('automatic search and learning',async()=>{const calls=[];const r=new RonCore({searchTool:async q=>{calls.push(q);return{answer:'باريس',source:'official',url:'https://example.gov/france'}}});assert.match(await say(r,'ما عاصمة فرنسا'),/باريس/);assert.deepStrictEqual(calls,['ما عاصمة فرنسا']);assert.strictEqual(r.store.get('فرنسا','عاصمة').o,'باريس');});
test('explicit search works when automatic search is disabled',async()=>{const calls=[];const r=new RonCore({autoSearch:false,searchTool:async q=>{calls.push(q);return{answer:'باريس',source:'test'}}});assert.match(await say(r,'ما عاصمة فرنسا'),/لا أعرف/);assert.match(await say(r,'اذا ابحث عنها'),/باريس/);assert.deepStrictEqual(calls,['ما عاصمة فرنسا']);});
test('timeout never hangs',async()=>{const r=new RonCore({searchTimeoutMs:25,searchTool:()=>new Promise(()=>{})});assert.match(await say(r,'ابحث عن شيء'),/لم أجد/);});
test('persistence',async()=>{const file=path.join(fs.mkdtempSync(path.join(os.tmpdir(),'ron-')),'data.json');const a=new RonCore({adapter:new JsonFileAdapter(file)});await say(a,'عاصمة اليابان هي طوكيو');const b=new RonCore({adapter:new JsonFileAdapter(file)});assert.match(await say(b,'ما عاصمة اليابان'),/طوكيو/);});
test('trust ranking',()=>{const r=new RonCore();r.store.set('x','y','صحيح',{source:'user'});assert.strictEqual(r.store.set('x','y','خطأ',{source:'search'}).status,'kept');});
test('unparsed logged',async()=>{const r=new RonCore();assert.match(await say(r,'بلا بلا بلا'),/لم أفهم/);assert.strictEqual(r.store.unparsed.length,1);});
test('smalltalk',async()=>{const r=new RonCore();assert.match(await say(r,'كيف حالك'),/بخير/);assert.match(await say(r,'مرحبا'),/مرحبًا/);});
test('multi-hop reasoning',()=>{const s=new FactStore();s.set('القاهرة','جزء_من','مصر');s.set('مصر','جزء_من','افريقيا');assert.ok([...new Reasoner(s).infer().values()].some(v=>v.t.join() === 'القاهرة,جزء_من,افريقيا'));});
test('safe arithmetic',()=>{assert.strictEqual(arithmetic('2 + 3 * 4'),'14');assert.strictEqual(arithmetic('process.exit()'),null);assert.strictEqual(arithmetic('2 / 0'),null);});
test('source scoring',()=>{const a=evaluateResult({answer:'معلومة',source:'official',url:'https://example.gov'},'معلومة');const b=evaluateResult({answer:'معلومة',source:'unknown'},'معلومة');assert.ok(a.confidence>b.confidence);});

test('name meta-question never becomes a profile fact',async()=>{
 const r=new RonCore();
 await say(r,'اسمي اي');
 await say(r,'انا بسألك عن اسمي');
 assert.strictEqual(r.store.get('$user','اسم').o,'اي');
 assert.strictEqual(r.store.get('$self','اسم').o,'رون');
});
test('explicit name correction is a single user update',async()=>{
 const r=new RonCore();
 await say(r,'اسمي اي');
 await say(r,'اسمي اركانوس وليس اسمك بسألك عن اسمي');
 assert.strictEqual(r.store.get('$user','اسم').o,'اركانوس');
 assert.strictEqual(r.store.get('$self','اسم').o,'رون');
 assert.match(await say(r,'ما اسمي'),/اركانوس/);
});
test('confirming Ron identity does not overwrite self name',async()=>{
 const r=new RonCore();
 assert.match(await say(r,'اسمك رون فعلا'),/اسمي رون/);
 assert.strictEqual(r.store.get('$self','اسم').o,'رون');
});
