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
