import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const storage=new Map();
const context={
 console,
 localStorage:{
  getItem:k=>storage.get(k)??null,
  setItem:(k,v)=>storage.set(k,v),
  removeItem:k=>storage.delete(k)
 },
 Date, JSON, Math, setTimeout, clearTimeout, AbortController
};
vm.runInNewContext(fs.readFileSync("ron-core.js","utf8"),context);
assert.ok(context.RonCore);
assert.ok(context.RonCoreAdapters?.LocalStorageAdapter);
const ron=new context.RonCore({adapter:new context.RonCoreAdapters.LocalStorageAdapter("test-core")});
assert.match((await ron.handle("ما اسمك")).reply,/رون/);
assert.match((await ron.handle("اسمي زيريوس")).reply,/زيريوس/);
assert.match((await ron.handle("ما اسمي")).reply,/زيريوس/);
assert.match((await ron.handle("ما عاصمة مصر")).reply,/القاهرة/);
console.log("browser Ron Core runtime smoke test passed");

assert.match((await ron.handle("وما اسمي")).reply,/زيريوس/);
assert.match((await ron.handle("ما اسمي وما اسمك")).reply,/زيريوس.*رون/);
assert.match((await ron.handle("وانت كم عمرك")).reply,/ليس لدي عمر بشري/);

const followupRon = new context.RonCore({
  adapter: new context.RonCoreAdapters.LocalStorageAdapter("followup-context-test")
});
await followupRon.handle("ما اسمك");
for (const prompt of ["ليه؟", "لماذا؟", "ازاي؟", "كيف ذلك؟", "ما السبب؟", "ماذا تقصد؟"]) {
  const response = await followupRon.handle(prompt);
  assert.equal(response.frames[0]?.type, "context", prompt);
  assert.equal(response.frames[0]?.kind, "why-previous", prompt);
  assert.match(response.reply, /ردي السابق|أخطأت|لا أجد/, prompt);
}
const mathRon = new context.RonCore({
  adapter: new context.RonCoreAdapters.LocalStorageAdapter("arabic-conjunction-math-test")
});
for (const [prompt, expected] of [["كم يساوي 5*5", "25"], ["وكم يساوي 5*5", "25"], ["كم يساوي 425*525", "223125"], ["وكم يساوي 425*525", "223125"]]) {
  const response = await mathRon.handle(prompt);
  assert.equal(response.reply, "النتيجة: " + expected, prompt);
  assert.equal(response.frames[0]?.type, "calculation", prompt);
}
