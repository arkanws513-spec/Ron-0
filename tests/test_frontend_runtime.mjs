import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const ids = ["chat","composer","input","send","menu","settings","close-settings","new-chat","clear-data","export","import"];
const nodes = new Map();
const listeners = new Map();
for (const id of ids) {
  nodes.set(id, {
    id, value: "", disabled: false, scrollTop: 0, scrollHeight: 0, style: {},
    children: [], className: "", textContent: "",
    append(...items){ this.children.push(...items); },
    appendChild(item){ this.children.push(item); },
    replaceChildren(...items){ this.children = [...items]; },
    addEventListener(type, fn){ listeners.set(id + ":" + type, fn); },
    classList: { add(){}, remove(){} }
  });
}
const document = {
  getElementById(id){ return nodes.get(id) ?? null; },
  createElement(tag){ return {
    tag, className:"", textContent:"", children:[],
    append(...items){this.children.push(...items);},
    click(){},
    style:{}
  }; }
};
const storage = new Map();
const context = {
  document,
  localStorage: {
    getItem:k=>storage.has(k)?storage.get(k):null,
    setItem:(k,v)=>storage.set(k,v),
    clear:()=>storage.clear(),
    removeItem:k=>storage.delete(k)
  },
  console,
  setTimeout: fn => { fn(); return 1; },
  Blob: class { constructor(parts){this.parts=parts;} },
  URL: {createObjectURL:()=> "blob:test"},
  confirm:()=>true,
  alert:()=>{},
  Date,
  JSON,
  Math
};
vm.runInNewContext(fs.readFileSync("app.js","utf8"), context);

const submit = listeners.get("composer:submit");
assert.equal(typeof submit, "function");
const input = nodes.get("input");

input.value = "انا زيريوس وعمري 34 سنة";
submit({preventDefault(){}});
assert.match(nodes.get("chat").children.at(-1).children.at(-1).textContent, /زيريوس/);

input.value = "ما اسمي";
submit({preventDefault(){}});
assert.match(nodes.get("chat").children.at(-1).children.at(-1).textContent, /زيريوس/);

input.value = "كم عمري";
submit({preventDefault(){}});
assert.match(nodes.get("chat").children.at(-1).children.at(-1).textContent, /34/);

input.value = "مرحبا";
submit({preventDefault(){}});
assert.ok(nodes.get("chat").children.at(-1).children.at(-1).textContent.trim().length > 0);

console.log("frontend runtime smoke test passed");

input.value = "اسمي أركانوس";
submit({preventDefault(){}});
assert.match(nodes.get("chat").children.at(-1).children.at(-1).textContent, /اركانوس/);
input.value = "اسمي أركانوس وليس";
submit({preventDefault(){}});
input.value = "ما اسمي";
submit({preventDefault(){}});
assert.match(nodes.get("chat").children.at(-1).children.at(-1).textContent, /اركانوس/);
const savedLessons = JSON.parse(storage.get("ron-lessons-v5") || "[]");
assert.equal(savedLessons.find(x => x.key === "user.name")?.text, "اركانوس");

input.value = "اسمك كوين";
submit({preventDefault(){}});
input.value = "ما اسمك";
submit({preventDefault(){}});
assert.match(nodes.get("chat").children.at(-1).children.at(-1).textContent, /رون/);
assert.equal(JSON.parse(storage.get("ron-lessons-v5") || "[]").some(x => x.key === "ron.name" && x.text !== "رون"), false);


// Existing bad profile values are removed during the one-time v5 migration.
const nodes2 = new Map();
const listeners2 = new Map();
for (const id of ids) {
  nodes2.set(id, {
    id, value:"", disabled:false, scrollTop:0, scrollHeight:0, style:{},
    children:[], className:"", textContent:"",
    append(...items){this.children.push(...items);},
    appendChild(item){this.children.push(item);},
    replaceChildren(...items){this.children=[...items];},
    addEventListener(type,fn){listeners2.set(id+":"+type,fn);},
    classList:{add(){},remove(){}}
  });
}
const storage2 = new Map([
  ["ron-lessons-v5", JSON.stringify([
    {key:"user.name",text:"اركانوس وليس"},
    {key:"user.name",text:"اركانوس"},
    {key:"ron.name",text:"كوين"}
  ])],
  ["ron-facts-migration-v1","1"],
  ["ron-core-data-v2", JSON.stringify({facts:[
    {s:"$user",p:"اسم",o:"اركانوس وليس"},
    {s:"$user",p:"اسم",o:"اركانوس"},
    {s:"$self",p:"اسم",o:"كوين"},
    {s:"$self",p:"اسم",o:"رون"}
  ]})],
  ["ron-learning-core-v1", JSON.stringify({knowledge:[
    {kind:"fact",text:"اسمك هو اركانوس وليس"},
    {kind:"fact",text:"اسمي هو كوين"},
    {kind:"fact",text:"اسمك هو اركانوس"}
  ]})]
]);
const context2 = {
  document:{
    getElementById(id){return nodes2.get(id)??null;},
    createElement(tag){return {tag,className:"",textContent:"",children:[],append(...items){this.children.push(...items);},click(){},style:{}};}
  },
  localStorage:{
    getItem:k=>storage2.has(k)?storage2.get(k):null,
    setItem:(k,v)=>storage2.set(k,v),
    clear:()=>storage2.clear(),
    removeItem:k=>storage2.delete(k)
  },
  console,
  setTimeout:fn=>{fn();return 1;},
  Blob:class{constructor(parts){this.parts=parts;}},
  URL:{createObjectURL:()=>"blob:test"},
  confirm:()=>true,alert:()=>{},Date,JSON,Math
};
vm.runInNewContext(fs.readFileSync("app.js","utf8"),context2);
const migratedLessons=JSON.parse(storage2.get("ron-lessons-v5")||"[]");
assert.deepEqual(migratedLessons.filter(x=>x.key==="user.name").map(x=>x.text),["اركانوس"]);
assert.equal(migratedLessons.some(x=>x.key==="ron.name"&&x.text!=="رون"),false);
const migratedCore=JSON.parse(storage2.get("ron-core-data-v2")||"{}");
assert.deepEqual(migratedCore.facts.filter(x=>x.s==="$user"&&x.p==="اسم").map(x=>x.o),["اركانوس"]);
assert.deepEqual(migratedCore.facts.filter(x=>x.s==="$self"&&x.p==="اسم").map(x=>x.o),["رون"]);
const migratedKnowledge=JSON.parse(storage2.get("ron-learning-core-v1")||"{}");
assert.equal(migratedKnowledge.knowledge.some(x=>x.kind==="fact"&&x.text==="اسمك هو اركانوس وليس"),false);
assert.equal(migratedKnowledge.knowledge.some(x=>x.kind==="fact"&&x.text==="اسمي هو كوين"),false);
assert.equal(migratedKnowledge.knowledge.some(x=>x.kind==="fact"&&x.text==="اسمك هو اركانوس"),true);


// Regression: deterministic routing beats unrelated retrieval; concise preference survives a fresh runtime.
input.value = "هل أنت جاهز؟";
submit({preventDefault(){}});
assert.equal(nodes.get("chat").children.at(-1).children.at(-1).textContent, "أيوه، جاهز.");

input.value = "حين أسألك ما هي عاصمة مصر، قول القاهرة فقط";
submit({preventDefault(){}});
assert.match(nodes.get("chat").children.at(-1).children.at(-1).textContent, /القاهرة/);
input.value = "ما هي عاصمة مصر؟";
submit({preventDefault(){}});
assert.equal(nodes.get("chat").children.at(-1).children.at(-1).textContent, "القاهرة");
assert.equal(JSON.parse(storage.get("ron-preferences-v1") || "{}").shortEgyptCapital, true);

const nodes3 = new Map();
const listeners3 = new Map();
for (const id of ids) {
  nodes3.set(id, {
    id, value:"", disabled:false, scrollTop:0, scrollHeight:0, style:{},
    children:[], className:"", textContent:"",
    append(...items){this.children.push(...items);},
    appendChild(item){this.children.push(item);},
    replaceChildren(...items){this.children=[...items];},
    addEventListener(type,fn){listeners3.set(id+":"+type,fn);},
    classList:{add(){},remove(){}}
  });
}
const context3 = {
  document:{
    getElementById(id){return nodes3.get(id)??null;},
    createElement(tag){return {tag,className:"",textContent:"",children:[],append(...items){this.children.push(...items);},click(){},style:{}};}
  },
  localStorage:{
    getItem:k=>storage.has(k)?storage.get(k):null,
    setItem:(k,v)=>storage.set(k,v),
    clear:()=>storage.clear(),
    removeItem:k=>storage.delete(k)
  },
  console,setTimeout:fn=>{fn();return 1;},Blob:class{constructor(parts){this.parts=parts;}},
  URL:{createObjectURL:()=>"blob:test"},confirm:()=>true,alert:()=>{},Date,JSON,Math
};
vm.runInNewContext(fs.readFileSync("app.js","utf8"),context3);
nodes3.get("input").value = "ما هي عاصمة مصر؟";
listeners3.get("composer:submit")({preventDefault(){}});
assert.equal(nodes3.get("chat").children.at(-1).children.at(-1).textContent, "القاهرة");


 // An incomplete arithmetic follow-up must ask for the missing expression, not guess or retrieve unrelated text.
input.value = "وكم تساوي؟";
submit({preventDefault(){}});
assert.equal(nodes.get("chat").children.at(-1).children.at(-1).textContent, "ما العملية الحسابية التي تريد حسابها؟");
